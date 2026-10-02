"""Counts-based exact target decisions and symbolic identity; not a model rerun."""
from pathlib import Path
import argparse,json,time
import pandas as pd
import sympy as s
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path(__file__).resolve().parent/'aggregate_data/research_extensions');p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'exact_and_algebra_results.json');args=p.parse_args()
a=pd.read_csv(args.data/'S_research_all_504_policy_risks.csv')
rows=[]
for _,r in a[(a.case.isin(['A','C']))&(a.uncertainty=='TAB')&(a.policy=='expanded_delta')].iterrows():
 n=int(r.test_states);k=int(r.failures_including_unknown);b=20 if r['case']=='A' else 50
 # p=1/b, Binomial(n,p) lower tail = numerator / b**n.
 term=(b-1)**n;total=term
 for j in range(k):
  term,rem=divmod(term*(n-j),(j+1)*(b-1));assert rem==0
  total+=term
 passed=200000*total<=b**n
 rows.append({'case':r['case'],'dependence':r.dependence,'root':int(r.root_index),'n':n,'k':k,'target':1/b,'one_sided_alpha':'1/200000','exact_tail_le_alpha':passed,'matches_reported_target':bool(passed==(r.adjusted_one_sided_upper<=1/b))})
A,N,S,F,B,H,W,gN,g2,gs,g1,K,a0,b0,c0,as0,cs,R,t,kappa=s.symbols('A N S F B H W gN g2 gs g1 K a b c a_s c_s R t kappa')
T=B+a0*N-b0*N*S/A-c0*N*A;U=H+as0*S-cs*S*A
E=kappa*((gN-g2/A)*T+gs*U+W+g1*F/A+K)
P=kappa*((gN*A**2-g2*A)*(B+a0*N-c0*N*A)-b0*(gN*A-g2)*N*S+gs*A**2*(H+as0*S-cs*S*A)+A**2*W+g1*F*A+K*A**2)-(R-t)*A**2
identity=s.simplify(s.expand(P-(E-(R-t))*A**2))==0
x=s.symbols('x');subs={v:s.Symbol(str(v)+'0')+s.Symbol(str(v)+'1')*x for v in [A,N,S,F,B,H,W]}
degree=s.Poly(s.expand(P.subs(subs)),x).degree()
result={'scope':'Independent verification of the supplied 18 target counts and displayed S24 symbolic identity only; no feed inputs, simulated states or physical validation reconstructed.','target_checks':rows,'all_18_exact_decisions_pass':bool(all(z['exact_tail_le_alpha'] for z in rows)),'quartic_identity_exact':identity,'polynomial_degree_with_affine_inputs':degree,'required_domain':'A>0 and all original reference-chain and nutrient-input domains defined. Fixed microbial-protein approximation is retained; this is not full NASEM validation.'}
args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2));print('Exact checks:',len(rows),'all pass:',result['all_18_exact_decisions_pass'],'identity:',identity,'degree:',degree)
