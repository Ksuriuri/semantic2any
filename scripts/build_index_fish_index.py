"""One sequential scan, atomic fixed-width offsets shared by all training ranks."""
import argparse,fcntl,json,os,time
from pathlib import Path
import numpy as np

def signature(p):
 s=p.stat();return dict(path=str(p.resolve()),size=s.st_size,mtime_ns=s.st_mtime_ns)
def build(path,expected):
 path=Path(path).resolve();root=Path(str(path)+'.idx');root.mkdir(exist_ok=True)
 lock=(root/'build.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 before=signature(path);meta=root/'meta.json'
 if meta.exists():
  m=json.loads(meta.read_text());assert m['source']==before and m['rows']==expected
  assert (root/'offsets.u64').stat().st_size==8*expected
  print('ALREADY_COMPLETE',str(path),expected,flush=True);return
 tmp=root/'offsets.u64.partial';base=rows=0;started=last=time.time();lastbyte=None
 with path.open('rb',buffering=0) as src,tmp.open('wb') as dst:
  dst.write(np.array([0],dtype='<u8').tobytes())
  while True:
   data=src.read(64*1024*1024)
   if not data:break
   lastbyte=data[-1];ends=np.flatnonzero(np.frombuffer(data,dtype=np.uint8)==10).astype(np.uint64)
   rows+=len(ends);starts=ends+base+1;starts=starts[starts<before['size']]
   dst.write(starts.astype('<u8',copy=False).tobytes());base+=len(data)
   if time.time()-last>=30:
    print(json.dumps(dict(path=path.name,bytes=base,total=before['size'],rows=rows,MBps=base/(time.time()-started)/1e6)),flush=True);last=time.time()
  dst.flush();os.fsync(dst.fileno())
 assert lastbyte==10 and rows==expected and tmp.stat().st_size==rows*8,(rows,expected)
 assert signature(path)==before
 os.replace(tmp,root/'offsets.u64')
 offsets=np.memmap(root/'offsets.u64',dtype='<u8',mode='r')
 with path.open('rb') as f:
  for i in sorted(set([0,rows-1]+np.linspace(0,rows-1,min(100,rows),dtype=np.int64).tolist())):
   f.seek(int(offsets[i]));row=json.loads(f.readline());assert all(k in row for k in ['target','reference','index'])
 m=dict(version=1,source=before,rows=rows,dtype='<u8',finished=time.time(),elapsed=time.time()-started)
 t=meta.with_suffix('.tmp');t.write_text(json.dumps(m,indent=2));os.replace(t,meta)
 print('INDEX_COMPLETE',json.dumps(m),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();root=Path(a.root)
 complete=json.loads((root/'_LOCAL_COMPLETE.json').read_text())
 for split in ['valid','train']:build(root/(split+'.jsonl'),complete[split+'_rows'])
