#!/usr/bin/env python3
"""Render tables and a figure from existing results; no new model evaluation."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
(OUT / 'tables').mkdir(exist_ok=True)
(OUT / 'figures').mkdir(exist_ok=True)
paths = [
 'fold1/w2v2_large_960h_oracle_shift_local_replica_core/core_metrics.json',
 'fold2/w2v2_large_960h_oracle_shift_local_replica_core/core_metrics.json',
 'fold0/data2vec_large_960h_shift_core/core_metrics.json',
]
data = [json.loads((ROOT/'artifacts/results/l2_arctic_official_ut8'/p).read_text())['splits']['test']['conditions'] for p in paths]
lines = [r'\begin{tabular}{lrrr}', r'\toprule', r'Condition & W1 & W2 & D0\\', r'\midrule']
for label, key in [('NoShift','no_shift'), ('Full','full'), (r'DADS, 75\%','utility75'), (r'DADS, 50\%','utility50'), (r'Magnitude, 50\%','magnitude50')]:
    lines.append(label + ' & ' + ' & '.join(f'{100*d[key]["wer"]:.2f}' for d in data) + r'\\')
lines += [r'\bottomrule',r'\end{tabular}']
(OUT/'tables/retention.tex').write_text('\n'.join(lines)+'\n')
p = ROOT/'artifacts/results/l2_arctic_official_ut8/fold0/data2vec_large_960h_shift_empirical_package/metrics.json'
methods = json.loads(p.read_text())['splits']['test']['retention']['methods']
lines = [r'\begin{tabular}{@{}lrrr@{}}',r'\toprule',r'Method & 25\% & 50\% & 75\%\\',r'\midrule']
for key,label in [('Utility','DADS'),('Magnitude','Magnitude'),('Gradient','Gradient'),('Random','Random'),('Random+Rescale','Random scaled')]:
    cells=[]
    for k in ['25','50','75']:
        z=methods[key][k]
        s=f'{z["wer_percent"]:.2f}'
        if 'std_wer_percentage_points' in z:
            s=f'${s}\\pm{z["std_wer_percentage_points"]:.2f}$'
        cells.append(s)
    lines.append(label+' & '+' & '.join(cells)+r'\\')
lines += [r'\bottomrule',r'\end{tabular}']
(OUT/'tables/data2vec_controls.tex').write_text('\n'.join(lines)+'\n')
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':10,
                     'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig, ax=plt.subplots(figsize=(3.42,2.25))
y=np.arange(3)
for offset,key,label,color,hatch in [(-.18,'drop_best25','Revert high utility','#365973',''),(.18,'drop_worst25','Revert low utility','#cdcdcd','///')]:
    vals=[100*(d[key]['wer']-d['full']['wer']) for d in data]
    ax.barh(y+offset,vals,height=.29,label=label,color=color,edgecolor='#222222',linewidth=.5,hatch=hatch)
    for yy,v in zip(y+offset,vals):
        ax.text(max(0,v)+.16,yy,f'{v:+.2f}',va='center',fontsize=10)
ax.set_yticks(y,['W1','W2','D0'])
ax.invert_yaxis()
ax.set_xlim(-.8,12)
ax.set_xticks([0,3,6,9,12])
ax.axvline(0,color='#555555',linewidth=.6)
ax.set_xlabel('WER change (percentage points)',fontsize=10)
ax.legend(loc='lower center',bbox_to_anchor=(.5,1.0),frameon=False,ncol=1,fontsize=10,handlelength=1.2,labelspacing=.2)
fig.subplots_adjust(left=.115,right=.97,bottom=.24,top=.73)
fig.savefig(OUT/'figures/deletion_effect.pdf')
fig.savefig(OUT/'figures/deletion_effect.png',dpi=180)
plt.close(fig)
print('Rendered two tables and the deletion figure from saved metrics.')
