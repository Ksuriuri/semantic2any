# IndexFish C: reference prefix and aligned local Index input

This is a separate architecture selected by `architecture: prefix_aligned`.
Use `configs/index_fish_prefix_c.yaml` and a new run directory. Existing configs
and inference checkpoints default to the original cross-attention architecture.
Old trained weights are not a trained C model; strict loading intentionally fails
across architectures. No production training switch is performed by this change.

## Exact clock packing

Both streams describe the same audio segment. Use an integer 44,100 Hz clock:
Index interval i = [1764i,1764(i+1)); Fish interval t = [2048t,2048(t+1)).
For Fish t, enumerate floor(2048t/1764) + {0,1,2}, retaining positive overlap
and valid source length. Each Fish interval intersects at most three Index
intervals. Index IDs remain distinct ordered slots, including tokens that cross
Fish boundaries (intentional duplication). Each slot includes its embedding,
relative center offset, overlap fraction, and presence bit. Concatenate the
three slots and project to model width; concatenate with the normalized previous
Fish frame embedding and project again. Overlap does not multiply embedding
amplitude. The first target frame uses BOS, not current-frame labels.

Packing preserves token identity/order/clock coverage, not an information-theoretic
guarantee for the learned fixed-width projection or generated audio. Padding IDs
are masked before embedding. Never interpolate categorical IDs or stretch each
stream to normalized utterance length. A source tail whose start lies beyond
all Fish frames is rejected in training, offline generation, and stream
finalization instead of squeezed into the final frame. Existing dataset duration
checks still apply. The constructor requires >=40 ms code lookahead (or None);
three-slot overlap needs <40 ms beyond the Fish interval end. The existing
80 ms emission schedule remains conservative. Upstream Index encoder context
and latency are separate; this is not an end-to-end audio latency guarantee.

## Reference prefix and masks

Sequence: [reference Fish embeddings + reference positions/type] [separator]
[target input embeddings + target-relative time/type]. Target time starts at
zero independently of reference duration. Known reference tokens have
bidirectional reference attention, but cannot attend to any target token in
any layer. Targets attend to the valid reference/ separator and causal target
prefix only. Padding reference positions do not change target clocks. This
lets reference order matter, unlike the unencoded bag of frames in the original
reference cross-attention. Loss covers target frames only.

Keep the original 10-book dependency order: c0 precedes shared enrichment;
head k may use current-frame books <k, never book k or subsequent books.
Reference tokens remain known context through every layer. All Index/reference
cross-attention modules are absent in C. `RawCodeStream` uses this same forward
path, with no KV cache; this is correctness-first, not a throughput claim.

## Entry points

- Training: `trainers/train_index_fish_mtp.py --config configs/index_fish_prefix_c.yaml`
- Offline inference: `scripts/infer_index_fish_mtp.py --checkpoint .../inference.pt ...`
- Streaming: `RawCodeStream(model, reference_codes, reference_lengths)`
- Strict model reconstruction: `build_index_fish_model(config['model'], config.get('architecture', 'cross_attention'))`

Fill manifest paths and choose a new output directory before training. Reference
prefixes lengthen attention matrices. Do not inherit the old batch32 capacity
assumption: GPU memory/throughput and mixed-precision production canaries remain
to be measured. No quality/convergence improvement is claimed from unit tests.

## Validation

`python -m pytest tests/test_index_fish_prefix.py tests/test_index_fish_mtp.py tests/test_index_fish_shared_index.py tests/test_index_fish_block_sampler.py -q`

C tests cover long integer-clock coverage, tail rejection, padding invariance,
reference order sensitivity, all ten heads' current/future-label leakage,
teacher-forced/prefix agreement, offline/streaming equivalence, all-parameter
finite gradients, tiny-set overfit and architecture roundtrip. CPU DDP smoke
exercises actual trainer, two-rank checkpoint and resume. Dataset clock preflight
sampled 2,000 training rows and all 939 validation rows: no out-of-range Index
tails observed; this is not a scan of all 231 million training records.
