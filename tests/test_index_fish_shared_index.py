import importlib.util,json
from pathlib import Path
import numpy as np
import pytest
from semantic2any.data.index_fish_codes import IndexFishDataset
spec=importlib.util.spec_from_file_location('builder',Path(__file__).parents[1]/'scripts/build_index_fish_index.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
def manifest(tmp_path):
 p=tmp_path/'data.jsonl';rows=[{'target':{'id':str(i)},'reference':{},'index':{},'text':'中文'*i} for i in range(1,8)]
 p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));return p

def test_mmap_offsets_match_sequential(tmp_path):
 p=manifest(tmp_path);expected=IndexFishDataset(p).offsets
 builder.build(p,7);ds=IndexFishDataset(p)
 assert isinstance(ds.offsets,np.memmap);assert list(ds.offsets)==expected
 builder.build(p,7)
 with p.open('rb') as f:
  for i,x in enumerate(ds.offsets):f.seek(int(x));assert json.loads(f.readline())['target']['id']==str(i+1)

def test_stale_manifest_rejected(tmp_path):
 p=manifest(tmp_path);builder.build(p,7)
 with p.open('a') as f:f.write('{}\n')
 with pytest.raises(ValueError,match='Stale'):IndexFishDataset(p)

def test_truncated_index_rejected(tmp_path):
 p=manifest(tmp_path);builder.build(p,7);Path(str(p)+'.idx/offsets.u64').write_bytes(b'')
 with pytest.raises(ValueError,match='Invalid'):IndexFishDataset(p)

def test_wrong_count_not_published(tmp_path):
 p=manifest(tmp_path)
 with pytest.raises(AssertionError):builder.build(p,8)
 assert not Path(str(p)+'.idx/meta.json').exists()
