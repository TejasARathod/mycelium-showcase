"""Rebuild summary figures from thesis-derived aggregate data; no inference."""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'#f7f9fc','axes.facecolor':'#f7f9fc','text.color':'#152238','axes.labelcolor':'#152238','svg.fonttype':'none'})
base,ft='#7990aa','#087f8c'
intervals=json.loads((ROOT/'results/india_rural_intervals.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(12,4.5),layout='constrained')
for ax,key,label in zip(axes,['georeasoner','plonk','locale_specificity'],['Country prediction accuracy','PLONK GeoScore','Locale-specificity rating']):
 d=intervals[key]; vals=[d['base'],d['finetuned']]; cis=[d['base_ci'],d['finetuned_ci']]
 ax.bar([0,1],vals,color=[base,ft],width=.58)
 ax.errorbar([0,1],vals,yerr=[[v-ci[0] for v,ci in zip(vals,cis)],[ci[1]-v for v,ci in zip(vals,cis)]],fmt='none',ecolor='#152238',capsize=5)
 ax.set(xticks=[0,1],xticklabels=['Base','Fine-tuned'],ylim=(0,1.13),title=label)
 for x,v,ci in zip([0,1],vals,cis):ax.text(x,ci[1]+.035,f'{v:.3f}',ha='center',weight='bold')
 ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Rural India: measurable gains across three evaluation measures',fontsize=16)
fig.supxlabel('250 images per condition · 95% bootstrap intervals · 5,000 resamples\nSource: submitted thesis, Table 14. Locale-specificity is model-judged.',fontsize=10)
fig.savefig(ROOT/'assets/india-rural-results.png',dpi=160);plt.close(fig)
rows=list(csv.DictReader((ROOT/'results/geographic_scores.csv').open()))
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
x=np.arange(4)
for ax,kind,title in zip(axes,['accuracy','plonk'],['GeoReasoner country accuracy','PLONK GeoScore']):
 for offset,condition,c in [(-.18,'base',base),(.18,'finetuned',ft)]:
  vals=[float(r[f'{condition}_correct'])/int(r[f'{condition}_valid_n']) if kind=='accuracy' else float(r[f'{condition}_plonk_geoscore']) for r in rows]
  ax.bar(x+offset,vals,.34,label='Base' if condition=='base' else 'Fine-tuned',color=c)
 ax.set(xticks=x,xticklabels=['India\nrural','India\nurban','UK\nrural','UK\nurban'],ylim=(0,1.30),title=title);ax.legend(frameon=False,loc='upper center',ncol=2);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Four scenes, a controlled comparison',fontsize=16)
fig.supxlabel('Source: submitted thesis, Table 13. Valid denominators are retained in results/geographic_scores.csv.',fontsize=10)
fig.savefig(ROOT/'assets/four-cell-results.png',dpi=160);plt.close(fig)
print('Rebuilt two result figures from included source tables.')
