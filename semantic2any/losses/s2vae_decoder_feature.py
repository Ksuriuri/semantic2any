"""Frozen dots AudioVAE feature matching, uniform over t and sample rate."""
from __future__ import annotations
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def decoder_features(audio_vae, latent, layers=(2, 3, 4, 5)):
    """AMP-averaged outputs; input is normalized [B,128,T]. No hooks."""
    v = audio_vae.vocoder
    # Frozen SLSTM has no dropout; cuDNN needs training reserve for input grad.
    v.dec_mi_layer.train(True)
    x = v.post_proj(audio_vae.denormalize(latent.float()))
    x = v.dec_mi_layer(x.transpose(1, 2)).transpose(1, 2)
    d = v.decoder
    if min(layers) < 0 or max(layers) >= d.num_upsamples:
        raise ValueError("requested feature stage does not exist")
    x = d.conv_pre(x)
    outputs = []
    for i in range(d.num_upsamples):
        for up in d.ups[i]:
            x = up(x)
        xs = None
        for j in range(d.num_kernels):
            branch = d.resblocks[i * d.num_kernels + j](x)
            xs = branch if xs is None else xs + branch
        x = xs / d.num_kernels
        if i in layers:
            outputs.append(x)
        if i == max(layers):
            break
    return tuple(outputs)


class S2VAEDecoderFeatureLoss(nn.Module):
    def __init__(self, layers=(2, 3, 4, 5), weights=(.15, .25, .35, .25),
                 chunk_frames=32, max_samples=2, eps=1e-4, use_checkpoint=True,
                 full_batch_full_length=False, all_batch_random_crop=False,
                 context_frames=16, micro_batch_size=1):
        super().__init__()
        if len(layers) != len(weights) or not layers or tuple(sorted(set(layers))) != tuple(layers):
            raise ValueError('layers must be ascending, distinct, with matching weights')
        if chunk_frames < 1 or max_samples < 1 or eps <= 0 or min(weights) < 0 or sum(weights) <= 0:
            raise ValueError('invalid feature-loss parameters')
        self.layers, self.weights = tuple(layers), tuple(weights)
        self.chunk_frames, self.max_samples = chunk_frames, max_samples
        self.eps, self.use_checkpoint = eps, use_checkpoint
        self.full_batch_full_length = bool(full_batch_full_length)
        self.all_batch_random_crop = bool(all_batch_random_crop)
        self.context_frames, self.micro_batch_size = int(context_frames), int(micro_batch_size)
        if self.context_frames < 0 or self.micro_batch_size < 1:
            raise ValueError('context_frames must be nonnegative and micro_batch_size positive')
        if self.full_batch_full_length and self.all_batch_random_crop:
            raise ValueError('full-length and random-crop modes are mutually exclusive')
        self.last_bounds = None
        self.last_components = {}
        self.last_indices = None

    @property
    def component_keys(self):
        keys = ("total", *(f"stage{i}" for i in self.layers), "samples", "frames")
        if self.full_batch_full_length or self.all_batch_random_crop:
            keys += ("sample_coverage", "frame_coverage", "target_frames", "supervised_frames")
        if self.all_batch_random_crop:
            keys += ("decoded_frames",)
        return keys

    def describe(self):
        if self.all_batch_random_crop:
            return (f"[S2VAE-feature] stages={self.layers}, weights={self.weights}, ALL valid samples, "
                    f"independent random target crops up to {self.chunk_frames} frames, "
                    f"context up to {self.context_frames} frames/side (target only), "
                    f"loss on central crop, decoder microbatch={self.micro_batch_size}, "
                    f"fine-grained checkpoint={self.use_checkpoint}, fp32, all-t/all-SR; "
                    "frames metric=mean supervised center length")
        if self.full_batch_full_length:
            return (f"[S2VAE-feature] stages={self.layers}, weights={self.weights}, "
                    "ALL valid samples, FULL target length, prompt/padding excluded, "
                    f"uniform all-t/all-SR, fp32, sequential latent gradients, "
                    f"fine-grained checkpoint={self.use_checkpoint}; frames metric=mean target length")
        return f"[S2VAE-feature] stages={self.layers}, weights={self.weights}, uniform all-t/all-SR, max_samples={self.max_samples}, chunk_frames={self.chunk_frames}, eps={self.eps}, fp32, checkpoint={self.use_checkpoint}"

    def forward(self, predicted, target, lengths, prompt_lengths, times, audio_vae):
        if predicted.shape != target.shape or predicted.ndim != 3:
            raise ValueError('expected matching [B,C,T] latent tensors')
        available = lengths - prompt_lengths
        indices = torch.nonzero(available > 0, as_tuple=False).flatten()
        if not indices.numel():
            raise ValueError('no valid target frames')
        if self.full_batch_full_length:
            return self._forward_full(predicted, target, lengths, prompt_lengths, indices, audio_vae)
        if self.all_batch_random_crop:
            return self._forward_cropped(predicted, target, lengths, prompt_lengths, indices, audio_vae)
        # Uniform choice, not top-t: every sample is eligible, with no SR gate.
        indices = indices[torch.randperm(indices.numel(), device=indices.device)[:self.max_samples]]
        chunk = min(self.chunk_frames, int(available[indices].min()))
        pred, gt = [], []
        for i in indices.tolist():
            offset = int(torch.randint(int(available[i]) - chunk + 1, (), device=predicted.device))
            start = int(prompt_lengths[i]) + offset
            pred.append(predicted[i, :, start:start + chunk])
            gt.append(target[i, :, start:start + chunk])
        pred, gt = torch.stack(pred).float(), torch.stack(gt).float()
        fn = lambda z: decoder_features(audio_vae, z, self.layers)
        # Decoder always fp32, including when DiT runs under bf16 autocast.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            with torch.no_grad():
                refs = fn(gt)
            feats = checkpoint(fn, pred, use_reentrant=False) if self.use_checkpoint else fn(pred)
            layer_loss = [((p-r).abs().mean(-1) / r.abs().mean(-1).clamp_min(self.eps)).mean()
                          for p, r in zip(feats, refs)]
            total = sum(w*l for w,l in zip(self.weights, layer_loss))
        self.last_indices = indices.detach()
        self.last_components = {f'stage{i}': l.detach() for i,l in zip(self.layers,layer_loss)}
        self.last_components.update(total=total.detach(), samples=total.new_tensor(len(indices)).detach(),
                                    frames=total.new_tensor(chunk).detach())
        return total

    def _forward_full(self, predicted, target, lengths, prompt_lengths, indices, audio_vae):
        """Exact first-order chain rule, without retaining B decoder graphs.

        Every valid sample uses its entire target interval independently, so a
        short sample never truncates another sample. No prompt/padding features
        contribute. Decoder state is frozen and its operations are batch-separable.
        """
        needs_grad = torch.is_grad_enabled() and predicted.requires_grad
        latent_grad = torch.zeros_like(predicted) if needs_grad else None
        values = predicted.new_zeros(len(self.layers), dtype=torch.float32)
        bounds = [(i, int(prompt_lengths[i]), int(lengths[i])) for i in indices.tolist()]
        count = len(bounds)
        with torch.autocast(device_type=predicted.device.type, enabled=False):
            for i, start, end in bounds:
                if not 0 <= start < end <= predicted.size(-1):
                    raise ValueError('invalid prompt/target length')
                z = predicted[i:i+1, :, start:end].detach().float()
                z.requires_grad_(needs_grad)
                gt = target[i:i+1, :, start:end].detach().float()
                with torch.no_grad():
                    refs = decoder_features(audio_vae, gt, self.layers)
                feats = decoder_features_fine(audio_vae, z, self.layers, self.use_checkpoint)
                per_layer = torch.stack([
                    ((p-r).abs().mean(-1) / r.abs().mean(-1).clamp_min(self.eps)).mean()
                    for p, r in zip(feats, refs)
                ])
                values += per_layer.detach() / count
                if needs_grad:
                    sample_loss = sum(w*l for w, l in zip(self.weights, per_layer)) / count
                    grad = torch.autograd.grad(sample_loss, z)[0]
                    latent_grad[i:i+1, :, start:end] = grad
                    del sample_loss, grad
                del z, gt, refs, feats, per_layer
        value = sum(w*l for w, l in zip(self.weights, values))
        total_frames = sum(end-start for _, start, end in bounds)
        self.last_indices = indices.detach()
        self.last_components = {f'stage{i}': v.detach() for i, v in zip(self.layers, values)}
        self.last_components.update(
            total=value.detach(), samples=value.new_tensor(count),
            frames=value.new_tensor(total_frames / count),
            sample_coverage=value.new_tensor(1.), frame_coverage=value.new_tensor(1.),
            target_frames=value.new_tensor(total_frames), supervised_frames=value.new_tensor(total_frames),
        )
        if needs_grad:
            # Select valid targets instead of multiplying prompt/padding by
            # zero: an unconstrained NaN outside the target must not poison loss.
            surrogate = sum(
                (predicted[i:i+1, :, start:end] * latent_grad[i:i+1, :, start:end]).sum()
                for i, start, end in bounds
            )
            return value + (surrogate - surrogate.detach())
        return value


    def _forward_cropped(self, predicted, target, lengths, prompt_lengths, indices, audio_vae):
        """All samples, independent random centers, bounded target context.

        Decoder microbatches share exact input and center sizes; no padding is
        decoded. Immediate latent gradients preserve the first-order chain rule.
        Context can receive gradients; prompt and padding cannot.
        """
        needs_grad = torch.is_grad_enabled() and predicted.requires_grad
        latent_grad = torch.zeros_like(predicted) if needs_grad else None
        values = predicted.new_zeros(len(self.layers), dtype=torch.float32)
        groups, bounds = {}, []
        for i in indices.tolist():
            lo, hi = int(prompt_lengths[i]), int(lengths[i])
            if not 0 <= lo < hi <= predicted.size(-1):
                raise ValueError('invalid prompt/target length')
            size = min(self.chunk_frames, hi-lo)
            start = lo + int(torch.randint(hi-lo-size+1, (), device=predicted.device))
            end = start+size
            left, right = max(lo, start-self.context_frames), min(hi, end+self.context_frames)
            bound = (i, left, right, start, end)
            bounds.append(bound)
            groups.setdefault((right-left, start-left, size), []).append(bound)
        count = len(bounds)
        with torch.autocast(device_type=predicted.device.type, enabled=False):
            for (decoded_size, offset, size), members in groups.items():
                for k in range(0, len(members), self.micro_batch_size):
                    batch = members[k:k+self.micro_batch_size]
                    z = torch.stack([predicted[i, :, left:right] for i,left,right,_,_ in batch]).detach().float()
                    z.requires_grad_(needs_grad)
                    gt = torch.stack([target[i, :, left:right] for i,left,right,_,_ in batch]).detach().float()
                    with torch.no_grad():
                        refs = decoder_features(audio_vae, gt, self.layers)
                    feats = decoder_features_fine(audio_vae, z, self.layers, self.use_checkpoint)
                    losses = []
                    for pred_feat, ref in zip(feats, refs):
                        if pred_feat.size(-1) % decoded_size:
                            raise ValueError('decoder stage must preserve integer temporal scaling')
                        scale = pred_feat.size(-1) // decoded_size
                        a, b = offset*scale, (offset+size)*scale
                        pred_center, ref_center = pred_feat[..., a:b], ref[..., a:b]
                        # Normalize per sample/channel using only the center.
                        losses.append(((pred_center-ref_center).abs().mean(-1)
                                       / ref_center.abs().mean(-1).clamp_min(self.eps)).mean(-1))
                    per_layer = torch.stack(losses)  # [stage, microbatch]
                    values += per_layer.detach().sum(-1) / count
                    if needs_grad:
                        sample_loss = sum(w*l.sum() for w,l in zip(self.weights, per_layer)) / count
                        grad = torch.autograd.grad(sample_loss, z)[0]
                        for j, (i,left,right,_,_) in enumerate(batch):
                            latent_grad[i, :, left:right] = grad[j]
                        del sample_loss, grad
                    del z, gt, refs, feats, losses, per_layer, pred_feat, ref, pred_center, ref_center
        value = sum(w*l for w,l in zip(self.weights, values))
        supervised = sum(end-start for _,_,_,start,end in bounds)
        decoded = sum(right-left for _,left,right,_,_ in bounds)
        total_frames = int((lengths[indices]-prompt_lengths[indices]).sum())
        self.last_bounds = bounds
        self.last_indices = indices.detach()
        self.last_components = {f'stage{i}': v.detach() for i,v in zip(self.layers, values)}
        self.last_components.update(total=value.detach(), samples=value.new_tensor(count),
            frames=value.new_tensor(supervised/count), sample_coverage=value.new_tensor(1.),
            frame_coverage=value.new_tensor(supervised/total_frames),
            target_frames=value.new_tensor(total_frames), supervised_frames=value.new_tensor(supervised),
            decoded_frames=value.new_tensor(decoded))
        if needs_grad:
            surrogate = sum((predicted[i:i+1,:,left:right]*latent_grad[i:i+1,:,left:right]).sum()
                            for i,left,right,_,_ in bounds)
            return value + (surrogate-surrogate.detach())
        return value


def decoder_features_fine(audio_vae, latent, layers=(2, 3, 4, 5), enabled=True):
    """Same decoder features with stage/branch checkpointing for long targets."""
    def run(fn, x):
        if enabled and torch.is_grad_enabled() and x.requires_grad:
            return checkpoint(fn, x, use_reentrant=False)
        return fn(x)

    v = audio_vae.vocoder
    v.dec_mi_layer.train(True)
    x = run(v.post_proj, audio_vae.denormalize(latent.float()))
    x = run(v.dec_mi_layer, x.transpose(1, 2)).transpose(1, 2)
    d = v.decoder
    x = run(d.conv_pre, x)
    outputs = []
    for i in range(d.num_upsamples):
        def stage(z, i=i):
            for up in d.ups[i]:
                z = run(up, z)
            branches = None
            for j in range(d.num_kernels):
                branch = run(d.resblocks[i * d.num_kernels + j], z)
                branches = branch if branches is None else branches + branch
            return branches / d.num_kernels
        x = run(stage, x)
        if i in layers:
            outputs.append(x)
        if i == max(layers):
            break
    return tuple(outputs)
