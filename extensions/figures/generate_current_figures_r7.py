"""Readable SR-width main figures, bound to unchanged aggregate evidence."""
from pathlib import Path
import csv,json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
B=Path(__file__).absolute().parent;P=B/'package';D=P/'03_figures';F=P/'05_latex_source/figures';R=B.parent.parent/'revision_round3_20261002'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'lines.markeredgewidth':1.3,'pdf.fonttype':42,'ps.fonttype':42,
 'axes.linewidth':1.2,'xtick.major.width':1.2,'ytick.major.width':1.2,'savefig.dpi':300})
colors=['#0072B2','#D55E00','#009E73'];bindings=[]
def save(fig,name,records,sources):
 for ext in ['pdf','png']:fig.savefig(D/f'{name}.{ext}')
 shutil.copyfile(D/f'{name}.pdf',F/f'{name}.pdf');plt.close(fig)
 bindings.append({'name':name,'records':records,'source_sha256':{str(s.relative_to(R)):sha(s) for s in sources},'pdf_sha256':sha(D/f'{name}.pdf'),'png_sha256':sha(D/f'{name}.png')})

# Figure1: retained ideal-information timing; no numerical model inputs.
fig,ax=plt.subplots(figsize=(5.2,5.55));fig.subplots_adjust(left=.012,right=.988,top=.995,bottom=.005)
ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
blue='#21576B';green='#376849';ink='#1F3039';grey='#707C82'
def box(x,y,w,h,title,body,edge=blue,fill='#F1F5F6'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.007,rounding_size=0.012',linewidth=1.4,edgecolor=edge,facecolor=fill))
 ax.text(x+w/2,y+h-.018,title,ha='center',va='top',fontsize=9.2,fontweight='bold',color=edge)
 ax.text(x+w/2,y+h-.052,body,ha='center',va='top',fontsize=8.5,color=ink,linespacing=1.23)
def arrow(x1,y1,x2,y2,color=blue,style='-'):
 ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=10,linewidth=1.4,color=color,linestyle=style))
box(.045,.875,.91,.103,'Shared case and evaluation definition','Reference animal and inventory; fixed nutrient targets\nCase supply; declared composition distribution',edge=green,fill='#EDF4EE')
box(.045,.72,.91,.113,'Training: freeze one admissible menu Q','Construct Q from legal recipes and physical mixtures\nFreeze fixed recipe, adaptive rules and DM moments',edge=green,fill='#EDF4EE')
arrow(.5,.867,.5,.840,green)
ax.plot([.025,.975],[.685,.685],color=grey,linewidth=1.4,linestyle=(0,(4,3)))
ax.text(.5,.696,'Policies fixed before independent test sampling',ha='center',va='center',fontsize=8.5,color=grey)
box(.045,.56,.91,.088,'Test state from the declared model','Nutrient concentrations theta and actual ingredient DM d',edge=grey)
arrow(.5,.674,.5,.655,green)
box(.045,.39,.41,.12,'Fixed choice from Q','No nutrient observation\nUse frozen fixed recipe')
box(.545,.39,.41,.12,'Adaptive choice from Q','Ideal non-DM nutrients\nFrozen score; DM hidden')
arrow(.25,.552,.25,.518);arrow(.75,.552,.75,.518)
box(.045,.257,.41,.082,'Fixed as-fed action','Same q in each test state')
box(.545,.257,.41,.082,'Selected as-fed action','One recipe q(theta) in Q')
arrow(.25,.382,.25,.347);arrow(.75,.382,.75,.347)
box(.045,.115,.91,.10,'Common selected-chain evaluation','Chosen q + realised nutrients and DM; original domain\nFailure = known violation OR unknown',edge=grey)
arrow(.25,.249,.25,.223);arrow(.75,.249,.75,.223)
arrow(.5,.552,.5,.223,grey,(0,(3,3)))
ax.text(.5,.388,'Actual DM: evaluator only',ha='center',va='center',rotation=90,fontsize=8.5,color=grey,bbox={'facecolor':'white','edgecolor':'none','pad':.7})
box(.045,.012,.91,.079,'Report per case, source and training replicate','Failure rates; paired comparisons; descriptive costs',edge=grey,fill='#FAFBFB')
arrow(.5,.107,.5,.098,grey)
save(fig,'FigR7_timing_SR',[],[])


# Figure3: matched menu risks above, complete six compression contrasts below.
rs=R/'analysis/attribution_risks.csv';ps=R/'analysis/attribution_pairs.csv'
rates=list(csv.DictReader(rs.open()));pairs=list(csv.DictReader(ps.open()))
fig=plt.figure(figsize=(5.2,5.55));gs=fig.add_gridspec(2,2,height_ratios=[1.12,1],hspace=.65,wspace=.30)
axs=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1])];records=[]
for col,(ax,case,name,target) in enumerate(zip(axs,['dev_case_v3a','dev_case_v3c'],['A','C'],[5,2])):
 for mode,marker,offset in [('static','s',-.13),('delta','o',.13)]:
  for root in range(3):
   for j,q in enumerate(['Q0','Q1','Q2','Q3','Q4']):
    r=next(r for r in rates if r['case']==case and r['world']=='SD-H0' and r['root']==f'r{root}' and r['policy']==q+'_'+mode)
    x=j+offset+(root-1)*.07;v=100*float(r['failure_rate']);u=100*float(r['adjusted_upper'])
    ax.plot([x,x],[v,u],color=colors[root],linewidth=1.4);ax.plot(x,u,'_',color=colors[root],markersize=4,markeredgewidth=1.3)
    ax.plot(x,v,marker=marker,color=colors[root],markerfacecolor='white' if mode=='static' else colors[root],markersize=3.8,markeredgewidth=1.2)
    records.append({'case':name,'root':root,'policy':q+'_'+mode,'risk_percent':v,'upper_percent':u})
 ax.axhline(target,color='#666666',linestyle='--',linewidth=1.3);ax.set_ylim(0,16);ax.set_xlim(-.45,4.45)
 ax.set_xticks(range(5),['Q0','Q1','Q2','Q3','Q4']);ax.set_xlabel('Frozen menu')
 ax.set_yticks([0,5,10,15]);ax.set_ylabel('Failure probability (%)' if col==0 else '')
 ax.set_title(f'({chr(97+col)}) Case {name}',loc='left',fontweight='bold',fontsize=9.5)
 ax.grid(axis='y',color='#E5E5E5',linewidth=1.1);ax.spines[['top','right']].set_visible(False)
ax=fig.add_subplot(gs[1,:]);labels=[]
for j,(case,name,root) in enumerate([(c,n,r) for c,n in [('dev_case_v3a','A'),('dev_case_v3c','C')] for r in range(3)]):
 p=next(r for r in pairs if r['case']==case and r['world']=='SD-H0' and r['root']==f'r{root}' and r['policy']=='Q3_delta' and r['comparator']=='Q2_delta')
 v,l,u=[100*float(p[x]) for x in ['difference','ci_low','ci_high']]
 ax.plot([l,u],[j,j],color=colors[root],linewidth=1.4);ax.plot([l,u],[j,j],'|',color=colors[root],markersize=7,markeredgewidth=1.4);ax.plot(v,j,'o',color=colors[root],markersize=4)
 labels.append(f'{name}, replicate {root}');records.append({'case':name,'root':root,'paired_policy':'Q3_delta','paired_comparator':'Q2_delta','difference_points':v,'lower_points':l,'upper_points':u})
ax.axvline(0,color='#666666',linewidth=1.3);ax.set_ylim(5.5,-.5);ax.set_yticks(range(6),labels)
ax.set_xlabel('Q3 minus Q2 failure (percentage points)');ax.set_title('(c) Compression: paired differences',loc='left',fontweight='bold',fontsize=9.5)
ax.grid(axis='x',color='#E5E5E5',linewidth=1.1);ax.spines[['top','right']].set_visible(False)
handles=[Line2D([],[],color=c,marker='o',linestyle='',label=f'Replicate {i}',markersize=4) for i,c in enumerate(colors)]
handles += [Line2D([],[],color='#555555',marker='s',markerfacecolor='white',linestyle='',label='Fixed',markersize=4),Line2D([],[],color='#555555',marker='o',linestyle='',label='Delta',markersize=4)]
fig.legend(handles=handles,ncol=3,frameon=False,loc='lower center',bbox_to_anchor=(.55,.003),fontsize=8.5,columnspacing=.7,handletextpad=.2)
fig.subplots_adjust(left=.18,right=.97,top=.935,bottom=.17)
save(fig,'FigR7_attribution_SR',records,[rs,ps])


(B/'R7_decision_and_attribution_figure_bindings.json').write_text(json.dumps({'status':'PASS','new_draws':0,'records':bindings,'scope':'Presentation-only redraw; point values retain original replicate-specific rates and intervals.'},indent=2)+'\n')
