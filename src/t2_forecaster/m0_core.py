from __future__ import annotations
import math, zlib
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd

MACRO_STEPS = {
    't2-F1-cpi-glidepath-2023': {140: 8, 160: 9},
    't2-F1-sahm-watch-2024': {145: 8, 165: 9},
    't2-F4-covid-nfp-2020': {21: 2},
    't2-F4-cpi-vintage-2022': {21: 2},
}

def _asset_col(df: pd.DataFrame) -> str | None:
    if 'asset' in df.columns: return 'asset'
    if 'asset_id' in df.columns: return 'asset_id'
    return None

def _load_asset_series(unit: Path, asset: str, asof: pd.Timestamp) -> pd.Series:
    for p in sorted(unit.glob('*.parquet'), key=lambda x: x.name):
        try:
            d=pd.read_parquet(p)
        except Exception:
            continue
        ac=_asset_col(d)
        if ac is None or 'date' not in d.columns or 'value' not in d.columns:
            continue
        m=d[ac].astype(str).eq(asset)
        if not m.any():
            continue
        x=d.loc[m,['date','value']].copy()
        x['date']=pd.to_datetime(x['date'],errors='coerce')
        x['value']=pd.to_numeric(x['value'],errors='coerce')
        x=x[x['date'].notna() & x['value'].notna() & (x['date']<=asof)].sort_values('date')
        x=x.drop_duplicates('date',keep='last')
        if len(x)==0:
            raise ValueError(f'empty_series:{asset}')
        return pd.Series(x['value'].to_numpy(float), index=pd.DatetimeIndex(x['date']), name=asset)
    raise KeyError(f'asset_not_found:{asset}')

def _trailing_raw(unit: Path, assets: Iterable[str], asof: pd.Timestamp) -> dict[str,pd.Series]:
    return {a:_load_asset_series(unit,a,asof).iloc[-300:] for a in assets}

def _gap_keep_dates(s: pd.Series) -> tuple[np.ndarray,float]:
    idx=pd.DatetimeIndex(s.index)
    if len(idx)<2:
        return np.zeros(len(idx),dtype=bool), math.inf
    days=np.diff(idx.values).astype('timedelta64[D]').astype(float)
    med=float(np.median(days[np.isfinite(days)])) if len(days) else math.inf
    thresh=max(10.0*med,5.0)
    keep=np.zeros(len(idx),dtype=bool)
    keep[1:]=days<=thresh
    return keep,thresh

def _step_series(s: pd.Series, target_type: str) -> pd.Series:
    keep,_=_gap_keep_dates(s)
    vals=s.to_numpy(float)
    if target_type=='level':
        step=np.empty(len(vals)); step[:]=np.nan
        step[1:]=np.diff(vals)
    elif target_type=='log_return':
        if np.median(np.abs(vals[np.isfinite(vals)]))>=0.2:
            raise ValueError('log_return_tripwire')
        if np.any(vals<=-1.0):
            raise ValueError('invalid_simple_return')
        step=np.log1p(vals)
        step[0]=np.nan
    else:
        raise ValueError(target_type)
    step[~keep]=np.nan
    return pd.Series(step,index=s.index,name=s.name).dropna()

def estimate(unit: Path, assets: list[str], asof: pd.Timestamp, target_type: str):
    assets_sorted=sorted(assets)
    raw=_trailing_raw(unit,assets_sorted,asof)
    cols=[_step_series(raw[a],target_type).rename(a) for a in assets_sorted]
    aligned=pd.concat(cols,axis=1,join='inner').dropna()
    if len(aligned)<2:
        raise ValueError(f'insufficient_aligned_steps:{len(aligned)}')
    arr=aligned.to_numpy(float)
    mu=np.mean(arr,axis=0)
    if len(assets_sorted)==1:
        cov=np.asarray([[float(np.var(arr[:,0],ddof=1))]],float)
    else:
        cov=np.asarray(np.cov(arr,rowvar=False,ddof=1),float)
    if not np.isfinite(mu).all() or not np.isfinite(cov).all():
        raise FloatingPointError('nonfinite_params')
    anchors={a:(float(raw[a].iloc[-1]) if target_type=='level' else 0.0) for a in assets_sorted}
    return assets_sorted, raw, mu, cov, anchors, aligned

def build_cells(assets: list[str], horizons: list[int], step_matrix: np.ndarray, order: str='sorted'):
    # step_matrix is in original assets x original horizons order
    amap={a:i for i,a in enumerate(assets)}
    hmap={int(h):j for j,h in enumerate(horizons)}
    if order=='sorted':
        cells=[(a,int(h)) for a in sorted(assets) for h in sorted(map(int,horizons))]
    elif order=='card':
        cells=[(a,int(h)) for a in assets for h in horizons]
    else:
        raise ValueError(order)
    out=[]
    for a,h in cells:
        out.append((a,h,int(step_matrix[amap[a],hmap[h]])))
    return out

def official_steps(unit_id: str, assets: list[str], horizons: list[int], target_frequency: str) -> np.ndarray:
    steps=np.empty((len(assets),len(horizons)),int)
    for ai,_a in enumerate(assets):
        for hi,h in enumerate(horizons):
            steps[ai,hi]=int(MACRO_STEPS.get(unit_id,{}).get(int(h),int(h)))
    return steps

def generate(unit: Path, unit_id: str, asof: str|pd.Timestamp, assets: list[str], horizons: list[int],
             target_type: str, target_frequency: str, step_matrix: np.ndarray|None=None,
             order: str='sorted', base_draws: int=500, output_draws: int=1000):
    asof=pd.Timestamp(asof)
    if step_matrix is None:
        step_matrix=official_steps(unit_id,assets,horizons,target_frequency)
    assets_sorted,raw,mu,Sigma,anchors,aligned=estimate(unit,assets,asof,target_type)
    aidx={a:i for i,a in enumerate(assets_sorted)}
    cells=build_cells(assets,horizons,np.asarray(step_matrix,int),order=order)
    d=len(cells)
    mean=np.empty(d,float)
    cov=np.empty((d,d),float)
    for i,(a,h,s) in enumerate(cells):
        mean[i]=anchors[a]+s*mu[aidx[a]]
        for j,(b,h2,t) in enumerate(cells):
            cov[i,j]=min(s,t)*Sigma[aidx[a],aidx[b]]
    cov=(cov+cov.T)/2
    cov.flat[::d+1]+=1e-10
    work=cov+1e-9*np.eye(d)
    try:
        L=np.linalg.cholesky(work)
        chol_fallback=False
    except np.linalg.LinAlgError:
        L=np.linalg.cholesky(np.diag(np.maximum(np.diag(work),1e-18)))
        chol_fallback=True
    seed=zlib.crc32(unit_id.encode('utf-8')) & 0x7FFFFFFF
    rng=np.random.default_rng(seed)
    Z=rng.standard_normal((base_draws,d))
    base=mean+Z@L.T
    if output_draws==base_draws:
        samples=base
    elif output_draws % base_draws==0:
        samples=np.repeat(base,output_draws//base_draws,axis=0)
    else:
        pick=np.floor((np.arange(output_draws)+0.5)*base_draws/output_draws).astype(int)
        pick=np.clip(pick,0,base_draws-1)
        samples=base[pick]
    # map cell columns back to [draw, original asset, original horizon]
    col={(a,h):i for i,(a,h,_s) in enumerate(cells)}
    out=np.empty((output_draws,len(assets),len(horizons)),float)
    for ai,a in enumerate(assets):
        for hi,h in enumerate(horizons):
            out[:,ai,hi]=samples[:,col[(a,int(h))]]
    diag={'seed':seed,'n_aligned_steps':len(aligned),'cell_order':order,'chol_fallback':chol_fallback,
          'mu':{a:float(mu[aidx[a]]) for a in assets_sorted},
          'anchors':anchors,'cells':[{'asset':a,'horizon':h,'steps':s} for a,h,s in cells]}
    return out,diag
