import copy
import json
import tempfile
import unittest
from pathlib import Path
from semantic2any.utils.checkpoint_retention import check_resume_config, retention_plan, prune_checkpoints

class RetentionTests(unittest.TestCase):
    def test_keeps_latest_and_archives_leaves_incomplete_and_unrelated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for s in [1000,2000,20000,40000,54000,55000,56000]:
                p=root/f'checkpoint-{s}';p.mkdir()
                (p/'trainer.json').write_text(json.dumps({'step':s,'world_size':2}))
                for n in ['model.safetensors','optimizer.bin','inference.pt','random_states_0.pkl','random_states_1.pkl']:(p/n).write_bytes(b'x')
            (root/'checkpoint-56000'/'random_states_1.pkl').unlink()
            other=root/'unrelated';other.mkdir();(other/'data').write_text('preserve')
            (root/'checkpoint-60000').symlink_to(other,target_is_directory=True)
            expected={'keep':[20000,40000,54000,55000],'delete':[1000,2000]}
            self.assertEqual(retention_plan(root,2,20000),expected)
            self.assertEqual(prune_checkpoints(root,2,20000),[1000,2000])
            self.assertTrue((root/'checkpoint-56000').exists())
            self.assertTrue((other/'data').exists())
            self.assertEqual(prune_checkpoints(root,2,20000),[])
    def test_archive_in_latest_two_is_not_double_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for s in [18000,20000,22000]:
                p=root/f'checkpoint-{s}';p.mkdir();(p/'trainer.json').write_text(json.dumps({'step':s,'world_size':1}))
                for n in ['model.safetensors','optimizer.bin','inference.pt','random_states_0.pkl']:(p/n).write_bytes(b'x')
            self.assertEqual(retention_plan(root,2,20000),{'keep':[20000,22000],'delete':[18000]})
    def test_policy_only_resume(self):
        old={'model':{'dim':1024},'data':{'path':'a'},'train':{'learning_rate':1e-4,'save_every':1000}}
        new=copy.deepcopy(old);new['train'].update(save_every=2000,keep_last_checkpoints=2,keep_every_steps=20000)
        self.assertEqual(len(check_resume_config(old,new)),3)
        self.assertEqual(old['train']['save_every'],1000)
        for section,key,value in [('train','learning_rate',2e-4),('model','dim',1280),('data','path','b')]:
            bad=copy.deepcopy(new);bad[section][key]=value
            with self.assertRaises(ValueError):check_resume_config(old,bad)
    def test_invalid_policy(self):
        for a,b in [(0,20000),(2,0),(-1,20000)]:
            with self.assertRaises(ValueError):retention_plan('/nonexistent',a,b)

if __name__=='__main__':unittest.main()
