#!/usr/bin/env python3
"""Deterministic checks of illustrative mathematics; no research simulations."""
from pathlib import Path
import math,json
from fractions import Fraction
from itertools import product
out=Path(__file__).absolute().parent/'mathematical_identity_checks_v2.json'
assert not out.exists()
# Rows are two equally likely hidden states; columns are actions.
losses=((0,1),(1,0));rf=sum(Fraction(min(r),2) for r in losses)
constant_risks=[sum(Fraction(r[a],2) for r in losses) for a in range(2)];ry=min(constant_risks)
assert rf==0 and ry==Fraction(1,2)
a1=a2=set(range(4));b1=set(range(3));b2=set(range(3,6))
ms_a=Fraction(len(a1)+len(a2),100);un_a=Fraction(len(a1|a2),100);ms_b=Fraction(len(b1)+len(b2),100);un_b=Fraction(len(b1|b2),100)
assert ms_b<ms_a and un_a<un_b
Phi=lambda x:.5*(1+math.erf(x/math.sqrt(2)));phi=lambda x:math.exp(-x*x/2)/math.sqrt(2*math.pi)
checks=[]
for m,v in product([-2.,0.,2.],[.5,1.,2.]):
 exact=m*phi(m/math.sqrt(v))/(2*v**1.5);h=1e-5*v;finite=(Phi(-m/math.sqrt(v+h))-Phi(-m/math.sqrt(v-h)))/(2*h)
 assert abs(exact-finite)<1e-10
 checks.append(dict(m=m,v=v,analytic=exact,finite_difference=finite,absolute_error=abs(exact-finite)))
r=dict(status='PASS',margin_definition='m is the effective signed margin m_raw + original tolerance t; violation is m_raw < -t',hidden_state_example=dict(full_information_risk=float(rf),implementable_risk=float(ry),algorithmic_excess=0),marginal_example=dict(a_sum=float(ms_a),a_union=float(un_a),b_sum=float(ms_b),b_union=float(un_b)),variance_derivative_checks=checks,scope='deterministic illustrative algebra, no research data or stochastic draws')
out.write_text(json.dumps(r,indent=2)+'\n');print(r['status'])
