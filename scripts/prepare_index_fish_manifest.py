"""Join exact-segment Index records to existing Fish pairs using an on-disk index.

Index JSONL rows: id,codes,num_frames,duration_seconds,[offset_bytes,generation].
No fuzzy filename/speaker matching, no code resampling and no silent missing rows.
"""
import argparse
import json
import math
import sqlite3
import tempfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from semantic2any.data.index_fish_codes import FISH_HZ

def resolve(record, base):
    record=dict(record)
    key='codes' if 'codes' in record else 'fish_code_path'
    path=record[key]
    if not path.startswith(('/', 'gs://')):record[key]=str(base).rstrip('/')+'/'+path
    return record

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fish-pairs',required=True);p.add_argument('--index-records',required=True)
    p.add_argument('--output',required=True);p.add_argument('--work-dir',required=True)
    args=p.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    Path(args.work_dir).mkdir(parents=True,exist_ok=True)
    count=0
    # Database and partial output are removed on any failed validation.
    with tempfile.TemporaryDirectory(dir=args.work_dir,prefix='index-fish-join-') as temp:
        db=sqlite3.connect(str(Path(temp)/'records.sqlite'))
        db.execute('CREATE TABLE records(id TEXT PRIMARY KEY, value TEXT NOT NULL)')
        with open(args.index_records) as f:
            for line in f:
                row=json.loads(line);rec=resolve(row,row.get('codes_root',Path(args.index_records).resolve().parent))
                db.execute('INSERT INTO records VALUES(?,?)',(rec['id'],json.dumps(rec)))
        db.commit()
        partial=output.with_name(output.name+'.partial')
        try:
            with partial.open('x') as out,open(args.fish_pairs) as f:
                for line in f:
                    row=json.loads(line);base=row.get('codes_root',Path(args.fish_pairs).resolve().parent)
                    target=resolve(row['target'],base);reference=resolve(row['reference'],base)
                    found=db.execute('SELECT value FROM records WHERE id=?',(target['id'],)).fetchone()
                    if found is None:raise ValueError(f'Missing exact Index segment: {target["id"]}')
                    index=json.loads(found[0]);duration=float(index['duration_seconds'])
                    if not math.isfinite(duration) or duration<=0 or abs(index['num_frames']/25-duration)>.081 or abs(target['num_frames']/FISH_HZ-duration)>2/FISH_HZ:
                        raise ValueError(f'Duration mismatch: {target["id"]}')
                    if float(index.get('start_seconds',0))!=float(target.get('start_seconds',0)):
                        raise ValueError(f'Segment start mismatch: {target["id"]}')
                    out.write(json.dumps({'target':target,'reference':reference,'index':index,'duration_seconds':duration})+'\n');count+=1
            if not count:raise ValueError('Empty paired output')
            partial.replace(output)
        finally:
            if partial.exists():partial.unlink()
            db.close()
    print(json.dumps({'rows':count,'output':str(output)}))
if __name__=='__main__':main()
