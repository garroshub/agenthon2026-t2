from __future__ import annotations
import base64,http.client,json,math,os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import unquote,urlsplit
import numpy as np

TIMEOUT=45; MAX_TOKENS=1200; MAX_BYTES=524288; MAX_CHARS=24000; MAX_AGE=3
MIN_CONF=.90; MIN_SIGNAL=.35; SHIFT=.12
ASSETS={"UST_2Y","UST_10Y"}
DOC_TYPES={"fomc_statement","fomc_minutes","cb_speech","landmark"}
KINDS={"announced_decision","current_assessment","conditional_or_forward_guidance","released_observation","balance_sheet_action","dissent_or_preference","other"}
ACTORS={"committee","named_participant","issuer","data_release","other","unknown"}
TIMES={"past","current","future","mixed","unknown"}
DIRS={"tightening","easing","unchanged","stronger","weaker","higher","lower","neutral","ambiguous","not_applicable"}

@dataclass(frozen=True)
class Result:
    draws: np.ndarray
    applied: bool
    signal: float
    doc_id: str|None
    reason: str

def _loads(s):
    def pairs(xs):
        d={}
        for k,v in xs:
            if k in d: raise ValueError("duplicate_json_key")
            d[k]=v
        return d
    def bad(x): raise ValueError("nonfinite_json")
    return json.loads(s,object_pairs_hook=pairs,parse_constant=bad)

def _select_doc(td:Path,asof:str):
    p=td/"corpus_index.json"
    if not p.is_file(): return None
    try: idx=json.loads(p.read_text(encoding="utf-8")); cutoff=date.fromisoformat(asof[:10])
    except Exception: return None
    pr={"fomc_statement":4,"cb_speech":3,"landmark":3,"fomc_minutes":2}; rows=[]
    for x in idx.get("documents",[]):
        if not isinstance(x,dict): continue
        typ=str(x.get("doc_type","")); ts=str(x.get("timestamp",""))[:10]
        if typ not in DOC_TYPES: continue
        try: dt=date.fromisoformat(ts)
        except ValueError: continue
        if dt>cutoff or (cutoff-dt).days>MAX_AGE: continue
        fp=td/str(x.get("file",""))
        if not fp.is_file(): continue
        rows.append((ts,pr[typ],str(x.get("doc_id",fp.name)),typ,fp))
    if not rows: return None
    ts,_,did,typ,fp=max(rows)
    source=fp.read_text(encoding="utf-8",errors="replace")[:MAX_CHARS]
    return {"timestamp":ts,"doc_id":did,"doc_type":typ,"source":source} if source.strip() else None

def _prompt(d,asof):
    head=[
        "Extract explicit policy-path semantics from this one dated document.",
        f"As-of date: {asof}. Use only DOCUMENT_TEXT. Do not forecast yields or prices.",
        'Return exactly one JSON object: {"claims":[{"fact_kind":"conditional_or_forward_guidance","actor_scope":"committee","temporal_scope":"future","direction":"tightening","conditional":true,"negated":false,"confidence":0.95,"span_start":0,"span_end":10,"quote":"exact substring"}]}',
        "At most 4 claims. Each claim is one indivisible semantic unit.",
        "quote must exactly equal DOCUMENT_TEXT[span_start:span_end].",
        "Future guidance direction means future POLICY path: tightening=higher/more restrictive; easing=lower/less restrictive.",
        "A completed current decision is not future guidance. A dissent/preference is not a committee decision.",
        "Ignore boilerplate, navigation and unsupported inference. If no explicit policy-path claim exists return {\"claims\":[]}.",
        "Allowed fact_kind: "+",".join(sorted(KINDS)),
        "Allowed actor_scope: "+",".join(sorted(ACTORS)),
        "Allowed temporal_scope: "+",".join(sorted(TIMES)),
        "Allowed direction: "+",".join(sorted(DIRS)),
        f'DOCUMENT_METADATA doc_id={d["doc_id"]} doc_type={d["doc_type"]} timestamp={d["timestamp"]}',
        "DOCUMENT_TEXT",d["source"],
    ]
    return "\n".join(head)

def _house(prompt):
    ep=os.environ.get("MODEL_ENDPOINT","").strip(); model=os.environ.get("MODEL_NAME","").strip()
    tok=os.environ.get("MODEL_TOKEN","").strip(); prox=os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY") or ""
    if not ep or not model or not tok or not prox: return None
    t=urlsplit(ep); p=urlsplit(prox)
    if t.scheme!="http" or not t.hostname or t.username or t.password or t.path.rstrip("/") not in ("","/v1") or t.query or t.fragment: return None
    if p.scheme!="http" or not p.hostname or not p.port or not p.username or p.password is None: return None
    cred=unquote(p.username)+":"+unquote(p.password)
    if any(ord(c)<33 or ord(c)>126 for c in tok) or any(ord(c)<32 or ord(c)>126 for c in cred): return None
    body=json.dumps({"model":model,"messages":[{"role":"user","content":prompt}],"temperature":0,"max_tokens":MAX_TOKENS,"chat_template_kwargs":{"enable_thinking":False}},separators=(",",":"),ensure_ascii=False).encode()
    hdr={"Content-Type":"application/json","Authorization":"Bearer "+tok,"Proxy-Authorization":"Basic "+base64.b64encode(cred.encode()).decode()}
    con=http.client.HTTPConnection(p.hostname,p.port,timeout=TIMEOUT)
    try:
        con.request("POST",t.scheme+"://"+t.netloc+"/v1/chat/completions",body,hdr); r=con.getresponse()
        if r.status!=200: return None
        raw=r.read(MAX_BYTES+1)
        if len(raw)>MAX_BYTES: return None
        e=_loads(raw.decode()); ch=e["choices"][0]
        if ch.get("finish_reason")=="length": return None
        s=ch["message"]["content"]
        if not isinstance(s,str) or not s.strip(): return None
        z=_loads(s.strip()); return z if isinstance(z,dict) else None
    except Exception: return None
    finally:
        try: con.close()
        except Exception: pass

def _valid(obj,source):
    xs=obj.get("claims") if isinstance(obj,dict) else None
    if not isinstance(xs,list) or len(xs)>4: return []
    out=[]; req={"fact_kind","actor_scope","temporal_scope","direction","conditional","negated","confidence","span_start","span_end","quote"}
    for c in xs:
        if not isinstance(c,dict) or not req.issubset(c): continue
        if c["fact_kind"] not in KINDS or c["actor_scope"] not in ACTORS or c["temporal_scope"] not in TIMES or c["direction"] not in DIRS: continue
        if type(c["conditional"]) is not bool or type(c["negated"]) is not bool: continue
        q=c["confidence"]
        if type(q) not in (int,float) or isinstance(q,bool) or not math.isfinite(float(q)) or not 0<=float(q)<=1: continue
        a,b=c["span_start"],c["span_end"]
        if type(a) is not int or type(b) is not int or not(0<=a<b<=len(source)): continue
        if not isinstance(c["quote"],str) or c["quote"]!=source[a:b]: continue
        out.append(c)
    return out

def _piece(c,typ):
    q=float(c["confidence"])
    if q<MIN_CONF or c["negated"] or c["direction"] not in {"tightening","easing"}: return None
    sg=1. if c["direction"]=="tightening" else -1.; k=c["fact_kind"]; a=c["actor_scope"]
    if k=="conditional_or_forward_guidance":
        if c["temporal_scope"] not in {"future","mixed"}: return None
        if a=="committee": w=1.
        elif a=="named_participant" and typ in {"cb_speech","landmark"}: w=.70
        else: return None
    elif k=="dissent_or_preference" and a=="named_participant": w=.35
    elif k=="balance_sheet_action" and a=="committee": w=.50
    else: return None
    return sg*q*w,q*w

def _signal(obj,source,typ):
    z=[x for x in (_piece(c,typ) for c in _valid(obj,source)) if x]
    if not z: return None
    s=sum(x for x,_ in z); signs={1 if x>0 else -1 for x,_ in z if abs(x)>1e-12}
    if len(signs)>1 and abs(s)<.60: return None
    return float(np.clip(s,-1,1)) if abs(s)>=MIN_SIGNAL else None

def maybe_apply(draws,*,target_type,target_frequency,assets,asof,text_dir):
    base=np.asarray(draws,dtype=float)
    if target_type!="level" or target_frequency!="daily" or len(assets)!=1 or assets[0] not in ASSETS or text_dir is None:
        return Result(base,False,0.,None,"ineligible")
    td=Path(text_dir)
    if not td.is_dir(): return Result(base,False,0.,None,"missing_text")
    d=_select_doc(td,asof)
    if d is None: return Result(base,False,0.,None,"no_recent_policy_doc")
    obj=_house(_prompt(d,asof))
    if obj is None: return Result(base,False,0.,d["doc_id"],"house_noop")
    s=_signal(obj,d["source"],d["doc_type"])
    if s is None: return Result(base,False,0.,d["doc_id"],"no_signal")
    out=base.copy()
    for j in range(out.shape[2]):
        sd=float(np.std(out[:,0,j],ddof=1))
        if not math.isfinite(sd) or sd<=1e-12: return Result(base,False,0.,d["doc_id"],"bad_scale")
        out[:,0,j]+=SHIFT*sd*s
    return Result(out,True,s,d["doc_id"],"applied") if np.isfinite(out).all() else Result(base,False,0.,d["doc_id"],"nonfinite")

