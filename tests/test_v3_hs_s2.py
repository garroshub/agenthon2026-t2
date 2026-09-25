from pathlib import Path
import json, os, sys, subprocess, hashlib, shutil, tomllib
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"submission_repo"
CAND=ROOT/"submission_v3_hs_probe_repo"
UNITS=ROOT/"reference"/"track2-forecasting-public"/"units"
OUT=ROOT/"outputs"/"v3_hs_s2"
OUT.mkdir(parents=True,exist_ok=True)
FIX=json.loads((ROOT/"experiments"/"v3_house_scenario_graph"/"s0_human_fixtures.json").read_text(encoding="utf-8"))

def run_cli(repo,unit,out):
    with (unit/"card.toml").open("rb") as fh: c=tomllib.load(fh)
    asof=str(c["provenance"]["data_cutoff"])
    env=os.environ.copy()
    for k in ["MODEL_ENDPOINT","MODEL_NAME","MODEL_TOKEN","http_proxy","HTTP_PROXY"]:
        env.pop(k,None)
    env["PYTHONPATH"]=str(repo/"src")
    cmd=[sys.executable,"-m","t2_forecaster.cli","--panels",str(unit),"--text",str(unit/"text"),"--asof",asof,"--out",str(out/"forecast.parquet"),"--draws","1000"]
    out.mkdir(parents=True,exist_ok=True)
    cp=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
    if cp.returncode:
        raise RuntimeError(f"{unit.name}:{cp.stderr[-1000:]}")
    return asof

def sha(p):
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def parity104():
    rows=[]
    for i,unit in enumerate(sorted(p for p in UNITS.iterdir() if (p/"card.toml").is_file()),1):
        b=OUT/"parity_base"/unit.name; c=OUT/"parity_candidate"/unit.name
        if b.exists(): shutil.rmtree(b)
        if c.exists(): shutil.rmtree(c)
        run_cli(BASE,unit,b); run_cli(CAND,unit,c)
        files=["forecast.parquet","forecast_meta.json","forecast_rationale.md"]
        exact=all(sha(b/f)==sha(c/f) for f in files)
        rows.append({"unit_id":unit.name,"exact_all_three":exact,**{f"exact_{f}":sha(b/f)==sha(c/f) for f in files}})
        if not exact: raise AssertionError("parity:"+unit.name)
        if i%20==0: print("PARITY",i,flush=True)
    d=pd.DataFrame(rows); d.to_csv(OUT/"S2_104_EXACT_PARITY.csv",index=False)
    return d

def import_candidate():
    sys.path.insert(0,str(CAND/"src"))
    import t2_forecaster.cli as cli
    import t2_forecaster.house_scenario as hs
    return cli,hs

def make_obj(hs,uid,assets,asof,snips,fx):
    if not snips: raise AssertionError(uid+":no_snips")
    s=snips[0]; q=s.quote[:min(160,len(s.quote))]
    fact={"fact_id":"F1","doc_id":s.doc_id,"published_at":s.published_at,"span_start":s.span_start,"span_end":s.span_start+len(q),"quote":q,"claim":"Cutoff text contains a market-relevant condition."}
    assumption={"assumption_id":"A1","mechanism":fx["mechanism"],"conditions":"The cited condition remains relevant over the forecast horizon.","counterexample":"A common global factor can dominate.","fact_refs":["F1"]}
    scenarios=[]
    for k,src in enumerate(fx["scenarios"],1):
        fids=[f"X{k}_{j+1}" for j in range(len(src["factors"]))]
        factors=[{"factor_id":fid,"description":desc,"rationale_refs":["A1"]} for fid,desc in zip(fids,src["factors"])]
        loads={a:{fid:int(v) for fid,v in zip(fids,src["loadings"][a])} for a in assets}
        scenarios.append({"scenario_id":f"S{k}","condition":src["name"],"factors":factors,"asset_loadings":loads})
    return {"schema_version":hs.SCHEMA_VERSION,"abstain":False,"reason":"mock fixture","facts":[fact],"assumptions":[assumption],"scenarios":scenarios}

def load_arr(path):
    d=pd.read_parquet(path)
    meta=json.loads((path.parent/"forecast_meta.json").read_text(encoding="utf-8"))
    assets=meta["asset_ids"]; hs=meta["horizons"]; ai={a:i for i,a in enumerate(assets)}; hi={int(h):i for i,h in enumerate(hs)}
    x=np.empty((int(meta["n_draws"]),len(assets),len(hs)))
    for r in d.itertuples(index=False): x[int(r.draw),ai[str(r.asset)],hi[int(r.horizon)]]=float(r.value)
    return x,assets,[int(h) for h in hs],meta

def marginal_exact(a,b):
    for i in range(a.shape[1]):
      for j in range(a.shape[2]):
        if not np.array_equal(np.sort(a[:,i,j]),np.sort(b[:,i,j])): return False
    return True

def fixture_integration():
    cli,hs=import_candidate()
    rows=[]
    for fx in FIX["units"]:
        uid=fx["unit_id"]; unit=UNITS/uid
        with (unit/"card.toml").open("rb") as fh: card=tomllib.load(fh)
        asof=str(card["provenance"]["data_cutoff"])
        bout=OUT/"mock_base"/uid
        if bout.exists(): shutil.rmtree(bout)
        run_cli(BASE,unit,bout)
        base,assets,horizons,_=load_arr(bout/"forecast.parquet")
        snips=hs.retrieve_text(unit/"text",asof)
        obj=make_obj(hs,uid,assets,asof,snips,fx)
        original=hs.one_house_request
        hs.one_house_request=lambda prompt,**kwargs:(obj,"ok")
        try:
            cout=OUT/"mock_candidate"/uid
            if cout.exists(): shutil.rmtree(cout)
            cout.mkdir(parents=True,exist_ok=True)
            cli.forecast_unit(unit,asof,cout/"forecast.parquet",1000,unit/"text")
        finally:
            hs.one_house_request=original
        cand,a2,h2,meta=load_arr(cout/"forecast.parquet")
        changed=not np.array_equal(base,cand)
        marg=marginal_exact(base,cand)
        method=meta["rationale"]["method"]
        if not (changed and marg and method=="primary_v1_hybrid_house_scenario_graph"):
            raise AssertionError(f"{uid}:changed={changed}:marg={marg}:method={method}")
        rows.append({"unit_id":uid,"changed":changed,"marginals_exact":marg,"method":method,"changed_entries":int(np.sum(base!=cand))})
    d=pd.DataFrame(rows); d.to_csv(OUT/"S2_MOCK_HOUSE_INTEGRATION.csv",index=False)
    return d

def numeric_parity():
    # Compare candidate pure-stdlib Gaussian-rank implementation to frozen S1 research implementation.
    import importlib.util
    cli,hs=import_candidate()
    spec=importlib.util.spec_from_file_location("s1frozen",ROOT/"experiments"/"v3_house_scenario_graph"/"s1_runtime.py")
    frozen=importlib.util.module_from_spec(spec)
    sys.modules["s1frozen"]=frozen
    spec.loader.exec_module(frozen)
    rows=[]
    for fx in FIX["units"]:
        uid=fx["unit_id"]; base,assets,horizons,_=load_arr(OUT/"mock_base"/uid/"forecast.parquet")
        rc=hs.gaussian_rank_corr(base); rf=frozen.gaussian_rank_corr(base)
        err=float(np.max(np.abs(rc-rf)))
        if err>1e-10: raise AssertionError(uid+":rank_parity:"+str(err))
        rows.append({"unit_id":uid,"rankcorr_max_abs_error":err})
    d=pd.DataFrame(rows); d.to_csv(OUT/"S2_NUMPY_RUNTIME_PARITY.csv",index=False)
    return d

if __name__=="__main__":
    p=parity104()
    m=fixture_integration()
    n=numeric_parity()
    summary={"status":"S2_HOST_INTEGRATION_PASS","parity_units":len(p),"parity_exact":int(p.exact_all_three.sum()),"mock_fixture_activations":len(m),"mock_marginals_exact":int(m.marginals_exact.sum()),"numpy_runtime_max_rankcorr_error":float(n.rankcorr_max_abs_error.max()),"house_called_real":False,"outcomes_read":False}
    (OUT/"S2_HOST_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))
