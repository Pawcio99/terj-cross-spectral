#!/usr/bin/env python3
"""Production neural data generator for TERJ Cross-Spectral.

This is a crash-conscious, no-overwrite implementation for the frozen 3x10
production design. It stores full float64 model parameters at every epoch,
full Gram eigenvalue vectors at every epoch, per-epoch metrics, and Adam state
at the optimizer checkpoints declared in plan_v1.json.

Author: Paweł Majsterek
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, obj: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name+".", suffix=".partial", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_npz(path: Path, **arrays: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name+".", suffix=".partial", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with tmp.open("wb") as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_npy(path: Path, arr: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name+".", suffix=".partial", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with tmp.open("wb") as f:
            np.save(f, arr, allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def git_head(repo: Path) -> str | None:
    try:
        return subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"], text=True).strip()
    except Exception:
        return None


def numpy_config_text() -> str:
    buf=io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            np.show_config()
    except Exception as e:
        return f"unavailable: {e!r}"
    return buf.getvalue()


def load_plan(path: Path) -> dict[str, Any]:
    obj=json.loads(path.read_text(encoding="utf-8"))
    if obj.get("schema")!="terj-cross-spectral-production-plan-v1":
        raise ValueError("unexpected plan schema")
    return obj


def make_dataset(seed: int, n_train: int=2048, n_val: int=512, n_test: int=1024, dim: int=64) -> dict[str,np.ndarray]:
    rng=np.random.default_rng(seed)
    total=n_train+n_val+n_test
    x=rng.normal(size=(total,dim)).astype(np.float64)
    score=x[:,0]+0.8*x[:,1]*x[:,2]+0.5*np.sin(x[:,3])
    y=(score>0).astype(np.int64)
    a=n_train; b=n_train+n_val
    return {
        "x_train":x[:a],"y_train_true":y[:a],
        "x_val":x[a:b],"y_val":y[a:b],
        "x_test":x[b:],"y_test":y[b:],
    }


def prepare_datasets(work: Path, plan: dict[str,Any]) -> None:
    outdir=work/"raw"/"neural"/"datasets"
    outdir.mkdir(parents=True,exist_ok=True)
    for ds in plan["neural"]["datasets"]:
        seed=int(ds["dataset_seed"]); shuffle_seed=int(ds["shuffle_seed"])
        path=outdir/f"dataset_{seed}.npz"
        meta=outdir/f"dataset_{seed}.json"
        if path.exists() and meta.exists():
            print(f"[exists] dataset {seed}")
            continue
        if path.exists() or meta.exists():
            raise RuntimeError(f"partial dataset files for {seed}")
        d=make_dataset(seed)
        srng=np.random.default_rng(shuffle_seed)
        perm=srng.permutation(d["y_train_true"].size)
        y_shuf=d["y_train_true"][perm]
        atomic_npz(path,**d,y_train_shuffled=y_shuf,shuffle_permutation=perm)
        atomic_json(meta,{
            "schema":"terj-neural-dataset-v1",
            "created_utc":utcnow(),
            "dataset_seed":seed,
            "batch_seed":int(ds["batch_seed"]),
            "shuffle_seed":shuffle_seed,
            "n_train":int(d["x_train"].shape[0]),
            "n_val":int(d["x_val"].shape[0]),
            "n_test":int(d["x_test"].shape[0]),
            "input_dim":int(d["x_train"].shape[1]),
            "positive_fraction_train":float(d["y_train_true"].mean()),
            "shuffle_fixed_points":int(np.sum(perm==np.arange(perm.size))),
        })
        print(f"[created] {path}")


def make_jobs(work: Path, plan: dict[str,Any]) -> Path:
    path=work/"manifests"/"neural_jobs_v1.jsonl"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite jobs: {path}")
    rows=[]
    idx=0
    for ds in plan["neural"]["datasets"]:
        for width in plan["neural"]["widths"]:
            for init_seed in plan["neural"]["init_seeds"]:
                for condition in plan["neural"]["conditions"]:
                    rows.append({
                        "job_index":idx,
                        "dataset_seed":int(ds["dataset_seed"]),
                        "batch_seed":int(ds["batch_seed"]),
                        "shuffle_seed":int(ds["shuffle_seed"]),
                        "width":int(width),
                        "init_seed":int(init_seed),
                        "condition":str(condition),
                        "epochs":int(max(plan["neural"]["epochs"])),
                        "batch_size":int(plan["neural"]["batch_size"]),
                        "optimizer_checkpoints":[int(x) for x in plan["neural"]["optimizer_checkpoints"]],
                    })
                    idx+=1
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row,separators=(",",":"))+"\n")
    print(f"wrote {len(rows)} jobs -> {path}")
    return path


def read_job(path: Path,index: int) -> dict[str,Any]:
    with path.open(encoding="utf-8") as f:
        for i,line in enumerate(f):
            if i==index:
                obj=json.loads(line)
                if int(obj["job_index"])!=index:
                    raise RuntimeError("job index mismatch")
                return obj
    raise IndexError(index)


def init_params(width:int,seed:int,input_dim:int=64,output_dim:int=2)->dict[str,np.ndarray]:
    rng=np.random.default_rng(seed)
    def w(shape,nin):
        return rng.normal(0.0,1.0/math.sqrt(nin),size=shape).astype(np.float64)
    return {
        "W1":w((input_dim,width),input_dim),"b1":np.zeros(width,np.float64),
        "W2":w((width,width),width),"b2":np.zeros(width,np.float64),
        "W3":w((width,output_dim),width),"b3":np.zeros(output_dim,np.float64),
    }


def forward(p:dict[str,np.ndarray],x:np.ndarray):
    h1=np.tanh(x@p["W1"]+p["b1"])
    h2=np.tanh(h1@p["W2"]+p["b2"])
    logits=h2@p["W3"]+p["b3"]
    return logits,(x,h1,h2)


def loss_acc(logits:np.ndarray,y:np.ndarray)->tuple[float,float]:
    z=logits-logits.max(axis=1,keepdims=True)
    ez=np.exp(z); probs=ez/ez.sum(axis=1,keepdims=True)
    loss=-np.mean(np.log(np.maximum(probs[np.arange(y.size),y],1e-300)))
    return float(loss),float(np.mean(np.argmax(logits,axis=1)==y))


def loss_grads(p:dict[str,np.ndarray],x:np.ndarray,y:np.ndarray):
    logits,(x0,h1,h2)=forward(p,x)
    z=logits-logits.max(axis=1,keepdims=True)
    e=np.exp(z); probs=e/e.sum(axis=1,keepdims=True)
    loss=-np.mean(np.log(np.maximum(probs[np.arange(y.size),y],1e-300)))
    d=probs
    d[np.arange(y.size),y]-=1.0
    d/=y.size
    g={}
    g["W3"]=h2.T@d; g["b3"]=d.sum(axis=0)
    dh2=d@p["W3"].T; dz2=dh2*(1-h2*h2)
    g["W2"]=h1.T@dz2; g["b2"]=dz2.sum(axis=0)
    dh1=dz2@p["W2"].T; dz1=dh1*(1-h1*h1)
    g["W1"]=x0.T@dz1; g["b1"]=dz1.sum(axis=0)
    return float(loss),g


class Adam:
    def __init__(self,p,lr=1e-3,b1=.9,b2=.999,eps=1e-8):
        self.lr=lr; self.b1=b1; self.b2=b2; self.eps=eps; self.t=0
        self.m={k:np.zeros_like(v) for k,v in p.items()}
        self.v={k:np.zeros_like(v) for k,v in p.items()}
    def step(self,p,g):
        self.t+=1
        c1=1-self.b1**self.t; c2=1-self.b2**self.t
        for k in p:
            self.m[k]=self.b1*self.m[k]+(1-self.b1)*g[k]
            self.v[k]=self.b2*self.v[k]+(1-self.b2)*(g[k]*g[k])
            p[k]-=self.lr*(self.m[k]/c1)/(np.sqrt(self.v[k]/c2)+self.eps)


def eval_set(p,x,y)->dict[str,float]:
    l,_=forward(p,x)
    loss,acc=loss_acc(l,y)
    return {"loss":loss,"accuracy":acc}


def spectral_snapshot(w2:np.ndarray)->tuple[np.ndarray,dict[str,float|int|None]]:
    g=w2.T@w2
    eig=np.linalg.eigvalsh(g).astype(np.float64)
    eig=np.maximum(eig,0.0)
    trace=float(eig.sum()); maxeig=float(eig[-1])
    spec=math.sqrt(maxeig)
    stable=trace/maxeig
    cond=None if eig[0]<=0 else float(maxeig/eig[0])
    return eig,{
        "n":int(eig.size),
        "trace":trace,
        "trace_over_width":trace/eig.size,
        "max_eigenvalue":maxeig,
        "max_eig_over_trace":maxeig/trace,
        "spectral_norm":spec,
        "stable_rank":stable,
        "stable_rank_over_width":stable/eig.size,
        "condition_number":cond,
    }


def rel(job:dict[str,Any])->Path:
    return Path(f"d{job['dataset_seed']}")/f"w{job['width']}"/f"i{job['init_seed']}"/job["condition"]


def save_optimizer(path:Path,adam:Adam)->None:
    arrays={"t":np.asarray(adam.t,dtype=np.int64)}
    for k,v in adam.m.items(): arrays[f"m_{k}"]=v
    for k,v in adam.v.items(): arrays[f"v_{k}"]=v
    atomic_npz(path,**arrays)


def run_job(work:Path,repo:Path,job:dict[str,Any])->None:
    r=rel(job)
    done=work/"raw"/"neural"/"metrics"/r/"DONE.json"
    if done.exists():
        print(f"[done] {r}")
        return

    # Refuse ambiguous partial state. Resume support will be a separate,
    # explicitly tested code path before production.
    categories=["checkpoints","eigenvalues","metrics","optimizer"]
    existing=[]
    for c in categories:
        p=work/"raw"/"neural"/c/r
        if p.exists() and any(p.iterdir()):
            existing.append(str(p))
    if existing:
        raise RuntimeError("partial run exists; refusing unsafe restart: "+"; ".join(existing))

    ds_path=work/"raw"/"neural"/"datasets"/f"dataset_{job['dataset_seed']}.npz"
    d=np.load(ds_path,allow_pickle=False)
    xtr=d["x_train"]; ytrue=d["y_train_true"]
    yobj=d["y_train_true"] if job["condition"]=="true" else d["y_train_shuffled"]
    xval=d["x_val"]; yval=d["y_val"]; xtest=d["x_test"]; ytest=d["y_test"]

    p=init_params(int(job["width"]),int(job["init_seed"]))
    adam=Adam(p)
    batch_rng=np.random.default_rng(int(job["batch_seed"]))
    opt_epochs=set(int(x) for x in job["optimizer_checkpoints"])

    manifest_path=work/"raw"/"neural"/"metrics"/r/"RUN.json"
    meta={
        "schema":"terj-neural-production-run-v1",
        "started_utc":utcnow(),
        "job":job,
        "repo_head":git_head(repo),
        "python":sys.version,
        "platform":platform.platform(),
        "numpy_version":np.__version__,
        "numpy_config":numpy_config_text(),
    }
    atomic_json(manifest_path,meta)

    def snapshot(epoch:int)->None:
        cp=work/"raw"/"neural"/"checkpoints"/r/f"epoch_{epoch:03d}.npz"
        ev=work/"raw"/"neural"/"eigenvalues"/r/f"epoch_{epoch:03d}.npy"
        mt=work/"raw"/"neural"/"metrics"/r/f"epoch_{epoch:03d}.json"
        atomic_npz(cp,**p)
        eig,spec=spectral_snapshot(p["W2"])
        atomic_npy(ev,eig)
        obj_train=eval_set(p,xtr,yobj)
        true_train=eval_set(p,xtr,ytrue)
        vm=eval_set(p,xval,yval); tm=eval_set(p,xtest,ytest)
        atomic_json(mt,{
            "schema":"terj-neural-epoch-metrics-v1",
            "epoch":epoch,
            "train_objective":obj_train,
            "train_true_labels":true_train,
            "validation":vm,
            "test":tm,
            "spectral":spec,
            "eigenvalue_file":str(ev),
            "checkpoint_file":str(cp),
        })
        if epoch in opt_epochs:
            op=work/"raw"/"neural"/"optimizer"/r/f"epoch_{epoch:03d}.npz"
            save_optimizer(op,adam)
        print(f"[{r}] epoch {epoch:03d}",flush=True)

    snapshot(0)
    n=xtr.shape[0]; bs=int(job["batch_size"])
    for epoch in range(1,int(job["epochs"])+1):
        perm=batch_rng.permutation(n)
        for start in range(0,n,bs):
            idx=perm[start:start+bs]
            _,g=loss_grads(p,xtr[idx],yobj[idx])
            adam.step(p,g)
        snapshot(epoch)

    atomic_json(done,{
        "schema":"terj-neural-run-done-v1",
        "finished_utc":utcnow(),
        "job":job,
        "repo_head":git_head(repo),
    })
    print(f"[complete] {r}")


def verify_dataset_pairs(work:Path,plan:dict[str,Any])->None:
    for ds in plan["neural"]["datasets"]:
        seed=int(ds["dataset_seed"])
        p=work/"raw"/"neural"/"datasets"/f"dataset_{seed}.npz"
        d=np.load(p,allow_pickle=False)
        yt=d["y_train_true"]; ys=d["y_train_shuffled"]
        if yt.shape!=ys.shape: raise RuntimeError("label shape mismatch")
        if np.array_equal(yt,ys): raise RuntimeError("shuffled labels identical to true labels")
        if sorted(yt.tolist())!=sorted(ys.tolist()): raise RuntimeError("shuffle changed class counts")
    print("dataset integrity: OK")


def smoke(work:Path,repo:Path)->None:
    # Dedicated tiny data/job outside production directories.
    sw=work/"tmp"/"neural_production_smoke"
    if sw.exists():
        import shutil; shutil.rmtree(sw)
    (sw/"raw"/"neural"/"datasets").mkdir(parents=True,exist_ok=True)
    d=make_dataset(101,n_train=256,n_val=64,n_test=64)
    rng=np.random.default_rng(103); perm=rng.permutation(256)
    atomic_npz(sw/"raw"/"neural"/"datasets"/"dataset_101.npz",**d,
               y_train_shuffled=d["y_train_true"][perm],shuffle_permutation=perm)
    job={
        "job_index":0,"dataset_seed":101,"batch_seed":102,"shuffle_seed":103,
        "width":16,"init_seed":104,"condition":"true","epochs":2,"batch_size":32,
        "optimizer_checkpoints":[0,1,2],
    }
    run_job(sw,repo,job)
    r=rel(job)
    for e in (0,1,2):
        eig=np.load(sw/"raw"/"neural"/"eigenvalues"/r/f"epoch_{e:03d}.npy",allow_pickle=False)
        if eig.shape!=(16,) or not np.all(np.diff(eig)>=0):
            raise RuntimeError("smoke eigenvalue integrity failure")
    if not (sw/"raw"/"neural"/"metrics"/r/"DONE.json").exists():
        raise RuntimeError("smoke DONE marker missing")
    print(f"SMOKE OK: {sw}")


def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--workspace",required=True)
    ap.add_argument("--repo",required=True)
    sub=ap.add_subparsers(dest="cmd",required=True)

    sub.add_parser("prepare-datasets")
    sub.add_parser("make-jobs")
    v=sub.add_parser("verify-datasets")

    r=sub.add_parser("run-job")
    r.add_argument("--jobs",required=True)
    r.add_argument("--index",type=int,required=True)

    sub.add_parser("smoke")
    args=ap.parse_args()

    work=Path(args.workspace).expanduser().resolve()
    repo=Path(args.repo).expanduser().resolve()
    plan=load_plan(work/"manifests"/"plan_v1.json")

    if args.cmd=="prepare-datasets":
        prepare_datasets(work,plan)
    elif args.cmd=="make-jobs":
        make_jobs(work,plan)
    elif args.cmd=="verify-datasets":
        verify_dataset_pairs(work,plan)
    elif args.cmd=="run-job":
        job=read_job(Path(args.jobs),args.index)
        run_job(work,repo,job)
    elif args.cmd=="smoke":
        smoke(work,repo)


if __name__=="__main__":
    main()
