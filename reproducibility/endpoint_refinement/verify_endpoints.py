"""Offline verification of saved endpoints and a strict-feasibility sensitivity check."""
from pathlib import Path
import json,platform,hashlib
import numpy as np
import scipy
from scipy.optimize import minimize
import baselines_public as bp

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'verification_results'

def solve(x):
    ev=bp.Evaluator();f,g,feas=ev.solve(x)
    return {'objective':f,'lmtd_margin_C':g.tolist(),'feasible_at_tolerance':{str(t):bool(np.all(g>=-t)) for t in [0,1e-6,1e-4]}}

def refine(x,strict=False,maxiter=200):
    bp.FEAS_TOL=0 if strict else 1e-4
    ev=bp.Evaluator();A,lb=bp.order_linear_u();margin=1e-6 if strict else 0
    con=[{'type':'ineq','fun':lambda u,k=k:float(A[k]@u-lb[k])} for k in range(2)]
    con +=[{'type':'ineq','fun':lambda u,k=k:float(ev.g_opt(u)[k]-margin)} for k in range(3)]
    res=minimize(ev.f_opt,np.clip(bp.to_u(x),0,1),method='SLSQP',bounds=[(0,1)]*4,constraints=con,options={'maxiter':maxiter,'ftol':1e-10})
    assert ev.best_x is not None
    return {'best':ev.best,'best_x':ev.best_x.tolist(),'solves':ev.n_solves,'solver_success':bool(res.success),'solver_message':str(res.message),'check':solve(ev.best_x),'cache_replay':[{'x':list(k),'objective':v[0],'constraints':None if v[1] is None else v[1].tolist(),'feasible':v[2]} for k,v in ev.cache.items()]}

def main():
    data_path=ROOT/'endpoint_data.json'
    data=json.loads(data_path.read_text(encoding='utf-8'))
    rows=[]
    for record in data:
        protocol=record['protocol'];i=record['run'];x=record['endpoint_x'];recorded=record['recorded']
        before=solve(x);assert before['feasible_at_tolerance']['0']
        assert abs(before['objective']-recorded['llm_best'])<1e-6
        rerun=refine(x);strict=refine(x,True)
        assert strict['check']['feasible_at_tolerance']['0']
        rows.append({**record,'before':before,'verification_rerun':rerun,'strict_sensitivity':strict,'endpoint_data_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest()})
        print(protocol,i,'old/replay solves',recorded['extra_distinct_solves'],rerun['solves'],flush=True)
    direct=refine([2,.44,.09,7],True,300)
    report={'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},'strict_test':'Reporting tolerance zero, optimisation constraints tightened by 1e-6 degC to avoid roundoff at the boundary; same bounds and properties. New offline sensitivity check, not historical LLM run.','rows':rows,'direct_strict':direct,'direct_verification_rerun':refine([2,.44,.09,7],False,300)}
    OUT.mkdir(exist_ok=True,parents=True);(OUT/'endpoint_revision_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('max historical objective replay discrepancy',max(abs(r['verification_rerun']['best']-r['recorded']['polished']) for r in rows))
    print('strict improvement range',min((r['before']['objective']-r['strict_sensitivity']['best'])/r['before']['objective']*100 for r in rows),max((r['before']['objective']-r['strict_sensitivity']['best'])/r['before']['objective']*100 for r in rows))
    print('strict direct',direct['best'],direct['solves'])

if __name__=='__main__':main()
