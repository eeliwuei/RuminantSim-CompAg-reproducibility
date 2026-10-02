"""Figure 5: bind completed coverage and measurement repairs, without new draws."""
from pathlib import Path
import csv, hashlib, json, shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

B=Path(__file__).absolute().parent
R=B.parent.parent/'revision_round3_20261002'
P=B/'package'
coverage=R/'coverage_repair/analysis/coverage_risks.csv'
measurement=R/'measurement_analysis/formal_readback_v2/all60_rates_private.csv'
cov=list(csv.DictReader(coverage.open()))
mea=list(csv.DictReader(measurement.open()))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'lines.markeredgewidth':1.3,
 'axes.linewidth':1.25,'xtick.major.width':1.25,'ytick.major.width':1.25,
 'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
colors=['#0072B2','#D55E00','#009E73']
fig,axs=plt.subplots(2,2,figsize=(5.2,5.55))
records=[]
for col,case in enumerate(['A','C']):
 ax=axs[0,col]
 ys=[];labels=[]
 for source_i,source in enumerate(['C0','C1_Table6_matched','C2_Corn4D_author']):
  for pool_i,policy in enumerate(['Q3_delta','Q2_safe_delta']):
   y=2*source_i+pool_i;ys.append(y)
   labels.append(f'{source[:2]}: '+('compressed Q3' if pool_i==0 else 'full Q2'))
   data=[r for r in cov if r['case']==case and r['target']==source and r['policy']==policy]
   assert len(data)==3
   for r in data:
    root=int(r['root']);rate=100*float(r['risk']);upper=100*float(r['adjusted_upper'])
    yy=y+(root-1)*.16
    ax.plot([rate,upper],[yy,yy],color=colors[root],linewidth=1.35)
    ax.plot(upper,yy,marker='|',color=colors[root],markersize=7,markeredgewidth=1.35)
    ax.plot(rate,yy,marker='o',color=colors[root],markersize=4.5)
    records.append({'experiment':'coverage','case':case,'target':source,'root':root,'policy':policy,'n':int(r['n']),'risk_percent':rate,'upper_percent':upper})
 ax.set_yticks(ys,labels=labels if col==0 else ['']*len(labels));ax.set_ylim(5.5,-.5)
 ax.set_xlim(0,4 if case=='A' else 2.2)
 ax.axvline(1 if case=='C' else 5,color='#666666',linestyle='--',linewidth=1.25)
 ax.set_title(f'({chr(97+col)}) Coverage, {case}',loc='left',fontweight='bold',fontsize=9.5,pad=8)
 ax.set_xlabel('Failure probability (%)')
 for y in [1.5,3.5]:ax.axhline(y,color='#DCDCDC',linewidth=1.1,zorder=0)
 ax.grid(axis='x',color='#E5E5E5',linewidth=1.1,zorder=0)
 ax.spines[['top','right']].set_visible(False)

 policies=['Q3_ideal_delta','Q3_raw_noise_0p5','Q3_postmean_0p5','Q3_postUA_0p5','Q2_postUA_0p5']
 labels=['Ideal nutrients: Q3','Raw observation: Q3','Posterior mean: Q3','Mean + variance: Q3','Mean + variance: Q2']
 ax=axs[1,col]
 for y,policy in enumerate(policies):
  data=[r for r in mea if r['case']==('dev_case_v3a' if case=='A' else 'dev_case_v3c') and r['policy']==policy]
  assert len(data)==3
  for r in data:
   root=int(r['training_root_index']);rate=100*float(r['failure_rate']);upper=100*float(r['adjusted_upper_recomputed']);yy=y+(root-1)*.16
   ax.plot([rate,upper],[yy,yy],color=colors[root],linewidth=1.35)
   ax.plot(upper,yy,marker='|',color=colors[root],markersize=7,markeredgewidth=1.35)
   ax.plot(rate,yy,marker='o',color=colors[root],markersize=4.5)
   records.append({'experiment':'measurement','case':case,'target':'C0','root':root,'policy':policy,'n':int(r['n_states']),'risk_percent':rate,'upper_percent':upper})
 ax.set_yticks(range(len(labels)),labels=labels if col==0 else ['']*len(labels));ax.set_ylim(4.5,-.5)
 ax.set_xlim(0,12 if case=='A' else 10)
 ax.axvline(5 if case=='A' else 2,color='#666666',linestyle='--',linewidth=1.25)
 ax.set_xlabel('Failure probability (%)')
 ax.set_title(f'({chr(99+col)}) Noise 0.5 SD, {case}',loc='left',fontweight='bold',fontsize=9.5,pad=8)
 ax.grid(axis='x',color='#E5E5E5',linewidth=1.1,zorder=0)
 ax.spines[['top','right']].set_visible(False)

handles=[Line2D([],[],color=c,marker='o',linewidth=1.35,label=f'Replicate {i}',markersize=4.5) for i,c in enumerate(colors)]
fig.legend(handles=handles,ncol=3,loc='lower center',bbox_to_anchor=(.55,.014),frameon=False,
 handlelength=1.0,handletextpad=.25,columnspacing=.7,fontsize=8.5)
fig.subplots_adjust(left=.34,right=.97,top=.94,bottom=.15,wspace=.35,hspace=.42)
out=P/'03_figures';out.mkdir(exist_ok=True)
for suffix in ['pdf','png']:
 target=out/f'FigR7_coverage_measurement.{suffix}'
 fig.savefig(target,metadata={'Title':'Candidate coverage and measurement uncertainty'} if suffix=='pdf' else None)
plt.close(fig)
shutil.copyfile(out/'FigR7_coverage_measurement.pdf',P/'05_latex_source/figures/FigR7_coverage_measurement.pdf')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt={'status':'PASS','new_draws':0,'plotted_rate_records':len(records),
 'complete_source_records':{'coverage':len(cov),'measurement':len(mea)},
 'source_sha256':{'coverage_risks.csv':sha(coverage),'all60_rates_private.csv':sha(measurement)},
 'output_sha256':{f.name:sha(f) for f in [out/'FigR7_coverage_measurement.pdf',out/'FigR7_coverage_measurement.png']},
 'line_width_pt_min':1.1,'records':records,
 'scope':'Separate coverage (36 plotted) and measurement (30 plotted) test populations; root-specific rate and one-sided bound, never pooled. Measurement panel shows hypothetical 0.5 train-SD nutrient noise except ideal control. Dashes: upper panels C 1%; lower A 5% and C 2%; A coverage 5% lies outside plotted range.'}
(B/'FigR7_coverage_measurement_binding.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('PASS: 66 plotted risk records bound to two frozen experiment tables.')
