#!/usr/bin/env python3
"""Deterministic non-overlapping Riemann windows for TERJ Cross-Spectral.

Creates finite-size matched level windows from already-converted Odlyzko tables.
The strict 1e22 holdout is never touched by this program.

Author: Paweł Majsterek
"""

from __future__ import annotations
import argparse, json, os, tempfile
from pathlib import Path
import numpy as np

WINDOWS=(48,128,256,512,1024)
SOURCES=("first_2m","1e12","1e21")


def atomic_npz(path:Path, **arrays):
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+".",suffix=".partial",dir=path.parent); os.close(fd)
    p=Path(tmp)
    try:
        with p.open("wb") as f:
            np.savez(f,**arrays); f.flush(); os.fsync(f.fileno())
        os.replace(p,path)
    finally:
        if p.exists(): p.unlink()


def load_src(root:Path,name:str):
    p=root/"raw"/"riemann"/"odlyzko"/"converted"/f"odlyzko_{name}.npz"
    d=np.load(p,allow_pickle=False)
    x=np.asarray(d["x"],dtype=np.float64)
    if not np.all(np.diff(x)>0): raise RuntimeError(f"{name}: not strictly increasing")
    return p,d,x


def make_windows(x:np.ndarray,n:int):
    usable=(len(x)//n)*n
    for start in range(0,usable,n):
        yield start,x[start:start+n]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--workspace",required=True)
    ap.add_argument("--source",choices=SOURCES,required=True)
    ap.add_argument("--window",type=int,choices=WINDOWS,required=True)
    args=ap.parse_args()
    root=Path(args.workspace).resolve()
    src,d,x=load_src(root,args.source)
    outdir=root/"derived"/"riemann_windows"/args.source/f"n{args.window}"
    outdir.mkdir(parents=True,exist_ok=True)
    start_index=int(str(d["start_index"]))
    base=str(d["base"])
    rows=[]
    for j,(off,w) in enumerate(make_windows(x,args.window)):
        path=outdir/f"window_{j:06d}.npz"
        atomic_npz(path,levels=w,source_offset=np.asarray(off,dtype=np.int64),
                   global_start_index=np.asarray(start_index+off,dtype=np.int64))
        rows.append({"id":f"{args.source}-n{args.window}-w{j:06d}",
                     "file":str(path),"source":args.source,"n_levels":args.window,
                     "source_offset":off,"global_start_index":str(start_index+off),
                     "base":base})
    manifest=outdir/"manifest.json"
    if manifest.exists(): raise FileExistsError(manifest)
    manifest.write_text(json.dumps({"schema":"terj-riemann-windows-v1",
        "source_file":str(src),"source":args.source,"window_size":args.window,
        "n_source_levels":int(len(x)),"n_windows":len(rows),
        "nonoverlapping":True,"windows":rows},indent=2)+"\n",encoding="utf-8")
    print(f"{args.source} n={args.window}: {len(rows)} non-overlapping windows")


if __name__=="__main__":
    main()
