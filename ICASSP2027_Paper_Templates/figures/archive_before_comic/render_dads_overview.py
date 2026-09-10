"""Single DADS overview in the author's supplied academic-diagram style.

Original vector geometry; no imported assets. The feature marks are schematic,
not measurements. Run with MPLCONFIGDIR=/tmp/dads-mpl python3 <this file>.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Arc, Polygon
from matplotlib.path import Path as MplPath
from matplotlib.colors import to_rgb

OUT = Path(__file__).resolve().parent
S = .7
INK, GRAY, EDGE = '#24272B', '#80858A', '#BBC0C4'
TEAL, PURPLE = '#4B928A', '#796396'
COLORS = ['#629DB5', '#D9BA60', '#86A064', '#CD927A', '#6B9D98', '#9278AB']
plt.rcParams.update({'font.family': 'DejaVu Serif', 'mathtext.fontset': 'dejavuserif',
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                     'hatch.linewidth': .5})
fig = plt.figure(figsize=(720*S/72, 200/72), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, 720), ylim=(342, 0)); ax.axis('off')


def text(x, y, value, size=12, color=INK, weight='normal', ha='left', **kw):
    return ax.text(x, y, value, fontsize=size*S, color=color, weight=weight,
                   ha=ha, va='center', zorder=8, **kw)


def box(x, y, w, h, fill='white', edge=EDGE, radius=5, lw=.85, zorder=1):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={radius}',
                      facecolor=fill, edgecolor=edge, linewidth=lw*S, zorder=zorder)
    ax.add_patch(p); return p


def line(points, color=INK, lw=1.1, dashed=False, zorder=2):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=lw*S, ls=(0, (4,3)) if dashed else '-', zorder=zorder)


def arrow(points, color=INK, dashed=False, width=1.3, outline=False):
    path = MplPath(points, [MplPath.MOVETO]+[MplPath.CURVE3 if outline else MplPath.LINETO]*(len(points)-1))
    p = FancyArrowPatch(path=path, arrowstyle='simple,head_length=6,head_width=9,tail_width=3' if outline else '-|>',
                        mutation_scale=S if outline else 8*S, lw=width*S,
                        facecolor='white' if outline else color, edgecolor=color,
                        linestyle=(0,(4,3)) if dashed else 'solid', zorder=4)
    ax.add_patch(p)


def lock(x, y, scale=1):
    ax.add_patch(Arc((x+4*scale, y+4*scale), 6*scale, 8*scale,
                    theta1=180, theta2=360, color=GRAY, lw=1.4*S, zorder=7))
    box(x, y+4*scale, 8*scale, 7*scale, fill=GRAY, edge=GRAY, radius=1, zorder=7)
    line([(x+4*scale,y+6*scale),(x+4*scale,y+9*scale)], 'white', .8, zorder=8)


def network(x, y, w, h, labels, fill='#ECEBED', font=12):
    ax.add_patch(Polygon([(x,y),(x+w,y+8),(x+w,y+h-8),(x,y+h)],
                        closed=True, facecolor=fill, edgecolor=INK, lw=1*S, zorder=5))
    lock(x+w/2-4, y+7)
    for j, label in enumerate(labels):
        text(x+w/2, y+28+j*13, label, size=font, ha='center', weight='bold')


def blend(c, a):
    return tuple((1-a)+a*v for v in to_rgb(c))


def matrix(x, y, w=31, h=27, color=TEAL, hatch=False):
    for col in range(6):
        for row in range(3):
            a = .26+.13*((col*2+row)%5)
            ax.add_patch(Rectangle((x+col*w/6,y+row*h/3), w/6, h/3,
                                  facecolor=blend(color,a), edgecolor='white', linewidth=.35*S,
                                  hatch='///' if hatch else None, zorder=5))
    ax.add_patch(Rectangle((x,y),w,h, facecolor='none',edgecolor=GRAY,lw=.7*S,zorder=6))


def ribbon(x, y, values, step=18, height=23, chosen=None):
    for i,v in enumerate(values):
        c=COLORS[i]
        is_off=chosen is not None and i not in chosen
        ax.add_patch(FancyBboxPatch((x+i*step,y),step-3,height,
            boxstyle='round,pad=0,rounding_size=2', facecolor='#F5F4F0' if is_off else blend(c,.18+.8*v),
            edgecolor=EDGE if is_off else c, linewidth=.8*S, zorder=5))
        if not is_off:
            # Filled length inside each tile encodes schematic intensity.
            ax.add_patch(Rectangle((x+i*step+3,y+height-4-v*(height-8)),step-9,v*(height-8),
                                  facecolor=c,edgecolor='none',zorder=6))


# Soft regions and serif headings echo the supplied reference's visual grammar.
box(4,34,184,269,fill='#F1F6F7',edge='none',radius=22)
box(200,34,259,269,fill='#FAF8E7',edge='none',radius=27)
box(472,34,244,269,fill='#F1F2E9',edge='none',radius=30)
text(96,15,'Given adapted recognizer',size=13.1,weight='bold',ha='center')
text(330,15,'(a) Decision-aligned selection',size=14,weight='bold',ha='center')
text(593,15,'(b) Counterfactual recognition',size=13.7,weight='bold',ha='center')

# Given feature sources. Adaptation has already happened; all weights are fixed.
box(26,49,111,38,fill='white',edge=EDGE,radius=9)
text(81,59,r'Speech $x$',size=12,weight='bold',ha='center')
for i,h in enumerate([3,5,10,17,9,4,8,15,7,12,5,3]):
    line([(51+5*i,75-h/2),(51+5*i,75+h/2)],'#5E91A0',1.2)
arrow([(26,69),(17,69),(17,136),(31,136)],GRAY)
arrow([(17,136),(17,216),(31,216)],GRAY)
network(32,109,78,53,['Reference','encoder'])
network(32,189,78,53,['Adapted','encoder'])
matrix(140,122,color='#9A9FA2',hatch=True);text(155,111,r'$E_r$',ha='center')
matrix(140,202);text(155,191,r'$E_a$',ha='center')
arrow([(110,136),(139,136)])
arrow([(110,216),(139,216)])

# Reference/adapted features jointly define the shift (the only subtraction).
arrow([(171,136),(191,136),(191,89),(225,89)],GRAY)
arrow([(171,216),(195,216),(195,103),(225,103)],GRAY)
text(265,57,'Representation shift',size=11.5,weight='bold',ha='center')
text(265,74,r'$\Delta=E_a-E_r$',size=12,ha='center')
delta=np.array([.90,.28,.65,.20,.75,.32])
grad=np.array([.08,.95,.65,.12,.15,.80])
product=delta*grad
selected=set(np.argsort(product)[-3:])
ribbon(224,86,delta,step=14,height=23)
text(395,57,'CTC sensitivity',size=11.5,weight='bold',ha='center')
text(395,74,r'$G=\partial\ell/\partial E_a$',size=11.5,ha='center')
ribbon(354,86,grad,step=14,height=23)

# A visible adapted head precedes the calibration loss.
arrow([(171,222),(182,222),(182,249),(70,249),(70,259)],GRAY)
box(28,260,84,35,fill='#ECEBED',edge=INK,radius=3)
lock(37,271)
text(80,270,'CTC head',size=11.5,weight='bold',ha='center')
text(80,286,r'$g_a$ (frozen)',size=11,ha='center')
box(136,266,44,29,fill='#FFF1D0',edge='none',radius=2)
text(158,273,'CTC',size=11.5,weight='bold',ha='center')
text(158,286,'loss',size=11.5,weight='bold',ha='center')
arrow([(112,280),(135,280)])
text(156,249,r'$y$',size=12,ha='center');arrow([(156,255),(156,265)],GRAY)
# The dashed feature gradient is available during calibration only.
arrow([(181,281),(199,281),(199,300),(451,300),(451,118),(395,118),(395,110)],PURPLE,True)

# Core idea: the same feature dimensions meet at an elementwise product hub.
arrow([(265,110),(265,124),(318,144)],COLORS[0])
arrow([(395,110),(395,124),(342,144)],PURPLE)
ax.add_patch(Circle((330,146),14,facecolor='#FFFDF7',edgecolor=PURPLE,lw=1*S,zorder=5))
text(330,146,'×',size=22,color=PURPLE,ha='center')
text(330,174,r'$|\Delta\odot G|$',size=14,color=PURPLE,ha='center')
ribbon(257,190,product/product.max(),step=25,height=25)
text(330,231,'Mean over calibration frames',size=11.5,ha='center')
arrow([(330,239),(330,250)],PURPLE)
for i in range(6):
    x=265+22*i
    ax.add_patch(Rectangle((x,254),17,17,facecolor=TEAL if i in selected else 'white',
                          edgecolor=TEAL if i in selected else GRAY,lw=.85*S,zorder=5))
    if i in selected:
        line([(x+4,262),(x+7,265),(x+13,258)],'white',1.2,zorder=7)
text(330,286,'Global top-K mask',size=13,weight='bold',ha='center')

# Counterfactual feature assembly: two sources, unchanged coordinate positions.
text(536,58,r'Adapted $E_a$',size=12,weight='bold',ha='center')
text(654,58,r'Reference $E_r$',size=12,weight='bold',ha='center')
matrix(494,77,w=83,h=32,color=TEAL)
matrix(613,77,w=83,h=32,color='#9A9FA2',hatch=True)
arrow([(535,110),(535,132),(554,161)],TEAL,outline=True)
arrow([(654,110),(654,132),(635,161)],GRAY,outline=True)
text(510,134,'Keep',size=12,color=TEAL,weight='bold',ha='center')
text(682,134,'Revert',size=12,color=GRAY,weight='bold',ha='center')
box(505,158,189,46,fill='#FAFBF6',edge='#CCD2C0',radius=9)
for i in range(6):
    x=517+28*i
    keep=i in selected
    ax.add_patch(Rectangle((x,169),21,24,facecolor=blend(TEAL,.7) if keep else '#E1E3E3',
        edgecolor=TEAL if keep else GRAY,linewidth=.8*S,hatch=None if keep else '///',zorder=5))
    text(x+10.5,180,str(i+1),size=10,color='white' if keep else '#656B70',ha='center')
arrow([(396,262),(464,262),(464,181),(504,181)],TEAL,width=1.6)
text(463,169,r'$m$',size=13,color=TEAL,ha='center')
text(600,217,'Counterfactual features',size=12.5,weight='bold',ha='center')
text(600,237,r'$\widetilde E_m=E_r+m\odot\Delta$',size=13,ha='center')
arrow([(601,243),(601,256),(571,256),(571,264)])
# Same frozen adapted head, used once the global mask has been calibrated.
box(507,265,106,31,fill='#ECEBED',edge=INK,radius=3)
lock(515,276)
text(560,273,'Same CTC head',size=10.5,weight='bold',ha='center')
text(560,288,r'$g_a$ (frozen)',size=11,ha='center')
arrow([(614,281),(642,281)])
box(647,264,48,31,fill='white',edge=GRAY,radius=5)
for yy,xx in [(272,687),(280,686),(288,678)]:
    line([(654,yy),(xx,yy)],GRAY,1)
text(672,255,r'Decode $\hat y$',size=11.5,ha='center')

# Compact visual legend and the experimental constraint are part of the figure.
box(5,314,710,25,fill='white',edge=EDGE,radius=9)
lock(17,320,.9);text(34,327,'Frozen network',size=11.2)
ax.add_patch(Rectangle((154,322),9,9,facecolor=TEAL,edgecolor=TEAL,lw=.6*S))
text(169,327,'Keep adapted',size=11.2)
ax.add_patch(Rectangle((279,322),9,9,facecolor='#E1E3E3',edgecolor=GRAY,hatch='///',lw=.6*S))
text(294,327,'Restore reference',size=11.2)
arrow([(427,327),(450,327)],PURPLE,True);text(456,327,'Calibration only',size=11.2)
text(643,327,'No retraining',size=11.7,weight='bold',ha='center')

for ext in ['pdf','svg','png']:
    kwargs={'dpi':300} if ext=='png' else {}
    if ext=='pdf': kwargs['metadata']={'Title':'DADS: decision-aligned selection and counterfactual recognition'}
    fig.savefig(OUT/f'dads_overview.{ext}',**kwargs)
plt.close(fig)
print('Saved single overview: dads_overview.pdf / .svg / .png')
