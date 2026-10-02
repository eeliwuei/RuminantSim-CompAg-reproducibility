from pathlib import Path
import csv, json, hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
B=Path(__file__).absolute().parent
D=Path('work/revision_r7_20261002/model_tail').absolute()
O=B/'package/03_figures'; O.mkdir(exist_ok=True)
a=list(csv.DictReader((D/'descriptive_same_protocol_differences.csv').open()))
r=list(csv.DictReader((D/'tail_identity_v1/NEL_all96_rule_records_safe.csv').open()))
key=lambda z:(z['family'],z['case'],z['scenario'],z['training_rep_id'],z['policy'])
risk={key(z):float(z['joint_failure_rate_total']) for z in r}
a=[z for z in a if z['constraint_id']=='PN-NEL-FIXEDDMI']
configs=[('a  Full pool: coverage','coverage_formal',[('Q2_safe_delta','Q3_delta','o','Q2 − Q3')]),
('b  Full pool: measurement','measurement_formal',[('Q2_ideal_delta','Q3_ideal_delta','o','Ideal'),('Q2_postUA_0p1','Q3_postUA_0p1','s','Noise 0.1'),('Q2_postUA_0p5','Q3_postUA_0p5','^','Noise 0.5')]),
('c  Variance score: noise 0.1','measurement_formal',[('Q3_postUA_0p1','Q3_raw_noise_0p1','o','UA − raw'),('Q3_postUA_0p1','Q3_postmean_0p1','s','UA − mean')]),
('d  Variance score: noise 0.5','measurement_formal',[('Q3_postUA_0p5','Q3_raw_noise_0p5','o','UA − raw'),('Q3_postUA_0p5','Q3_postmean_0p5','s','UA − mean')])]
plt.rcParams.update({'font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':1,'xtick.major.width':1,'ytick.major.width':1,'xtick.minor.width':1,'ytick.minor.width':1,'lines.linewidth':1.1,'pdf.fonttype':42})
fig,axs=plt.subplots(2,2,figsize=(5.6,5.65)); colors={'A':'#2166AC','C':'#B2182B'}
points=[]
for ax,(title,fam,groups) in zip(axs.flat,configs):
    for pol,comp,marker,lab in groups:
        rr=[z for z in a if z['family']==fam and z['policy']==pol and z['comparator']==comp]
        for z in rr:
            kp=key(z);kc=kp[:-1]+(comp,)
            x=100*(risk[kp]-risk[kc]);y=100*float(z['normalized_top_ceil5pct_mean_delta'])
            ax.scatter(x,y,s=27,c=colors[z['case']],marker=marker,linewidths=1.0,edgecolors='white',alpha=.9,zorder=3)
            points.append({**{k:z[k] for k in ['family','case','scenario','training_rep_id','policy','comparator']},'risk_difference_percentage_points':x,'NEL_tail_difference_percent_of_target':y,'panel':title[0]})
    ax.axhline(0,color='#555555',lw=1); ax.axvline(0,color='#777777',lw=1)
    ax.set_title(title,loc='left',pad=8)
    ax.set_xlabel('Joint failure difference (pp)',labelpad=5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_ylabel('NEL top-5% mean difference\n(% target)',labelpad=2)
    ax.grid(color='#DDDDDD',linewidth=1,alpha=.6);ax.set_axisbelow(True)
    handles=[Line2D([],[],color='#555555',marker=m,ls='',markersize=4.5,label=l) for _,_,m,l in groups]
    
    if title[0]!='a': ax.legend(handles=handles,fontsize=8.0,frameon=False,loc='best',handletextpad=.3,borderpad=.2)
fig.legend(handles=[Line2D([],[],color=c,marker='o',ls='',markersize=5,label='Case '+k) for k,c in colors.items()],loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.007),fontsize=8.5)
fig.subplots_adjust(left=.18,right=.99,top=.95,bottom=.155,hspace=.48,wspace=.68)
for ext in ['pdf','png']:fig.savefig(O/f'FigR7_2_latest_tail.{ext}',dpi=300)
import shutil
for ext in ['pdf','png']:shutil.copy2(O/f'FigR7_2_latest_tail.{ext}',B/f'package/05_latex_source/figures/FigR7_2_latest_tail.{ext}')
with (B/'latest_tail_figure_points_r7_2.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(points[0]));w.writeheader();w.writerows(points)
receipt={'status':'PASS_SOURCE_BOUND_DESCRIPTIVE_ONLY','n_points':len(points),'panels':{p:sum(z['panel']==p for z in points) for p in 'abcd'},'inputs':{str(p.relative_to(D)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [D/'descriptive_same_protocol_differences.csv',D/'tail_identity_v1/NEL_all96_rule_records_safe.csv']},'all_prespecified_scope_groups_displayed':True,'new_tail_inference':False,'new_draws':False}
(B/'R7_2_latest_tail_figure_binding.json').write_text(json.dumps(receipt,indent=2))
print(receipt)
