#!/usr/bin/env python3
"""Matched finite-size RMT controls for TERJ Cross-Spectral Stage A.

Primary nulls:
- Riemann: GUE beta=2
- Neural: real Wishart/Laguerre beta=1

Sensitivity/universality controls:
- CUE beta=2
- GOE beta=1
- Poisson non-repulsive control

Author: Paweł Majsterek
"""
from __future__ import annotations
import argparse,json,math,os,tempfile
from pathlib import Path
import numpy as np

def eig_gue(rng,n):
    z=(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))/math.sqrt(2*n)
    h=(z+z.conj().T)/2
    return np.linalg.eigvalsh(h).real

def eig_goe(rng,n):
    a=rng.normal(size=(n,n))/math.sqrt(n)
    h=(a+a.T)/2
    return np.linalg.eigvalsh(h)

def eig_wishart(rng,n):
    x=rng.normal(size=(n,n))/math.sqrt(n)
    return np.linalg.eigvalsh(x.T@x)

def eig_cue(rng,n):
    z=(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))/math.sqrt(2)
    q,r=np.linalg.qr(z)
    ph=np.diag(r); ph=np.where(np.abs(ph)>0,ph/np.abs(ph),1)
    u=q*ph.conj()
    ang=np.sort(np.mod(np.angle(np.linalg.eigvals(u)),2*np.pi))
    return ang

def eig_poisson(rng,n):
    return np.cumsum(rng.exponential(size=n))

GEN={"gue":eig_gue,"goe":eig_goe,"wishart":eig_wishart,"cue":eig_cue,"poisson":eig_poisson}

def atomic_npy(path,arr):
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+".",suffix=".partial",dir=path.parent); os.close(fd)
    p=Path(tmp)
    try:
        with p.open("wb") as f:
            np.save(f,np.asarray(arr,dtype=np.float64),allow_pickle=False); f.flush(); os.fsync(f.fileno())
        os.replace(p,path)
    finally:
        if p.exists(): p.unlink()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--workspace",required=True)
    ap.add_argument("--ensemble",choices=sorted(GEN),required=True)
    ap.add_argument("--n",type=int,required=True)
    ap.add_argument("--reps",type=int,default=5000)
    ap.add_argument("--seed",type=int,required=True)
    args=ap.parse_args()
    root=Path(args.workspace).resolve()
    out=root/"controls"/args.ensemble/f"n{args.n}"
    if out.exists() and any(out.iterdir()): raise RuntimeError(f"refusing existing populated dir: {out}")
    out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed)
    for i in range(args.reps):
        atomic_npy(out/f"sample_{i:05d}.npy",GEN[args.ensemble](rng,args.n))
        if (i+1)%100==0: print(f"{args.ensemble} n={args.n}: {i+1}/{args.reps}",flush=True)
    (out/"manifest.json").write_text(json.dumps({"schema":"terj-rmt-controls-v1",
        "ensemble":args.ensemble,"n":args.n,"reps":args.reps,"seed":args.seed},indent=2)+"\n")
    print("done",out)

if __name__=="__main__":
    main()
