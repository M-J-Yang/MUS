"""Render the separate introduction and method figures as editable vectors.

Run: MPLCONFIGDIR=/tmp/dads-mpl python3 <this file>
All geometry is schematic, not measured data. Drawing units are physical points.
Icons are original vector primitives; no raster or third-party assets are used.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Arc
from matplotlib.path import Path as MplPath

OUT = Path(__file__).resolve().parent
PURPLE, PALE = '#6B45B5', '#EEE9F7'
INK, GRAY, LIGHT = '#22232A', '#747780', '#D7D9DF'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans',
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})


def canvas(w, h):
    fig = plt.figure(figsize=(w/72, h/72), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, w), ylim=(h, 0)); ax.axis('off')
    return fig, ax


def text(ax, x, y, value, size=9, color=INK, weight='normal', ha='left', **kw):
    return ax.text(x, y, value, fontsize=size, color=color, weight=weight,
                   ha=ha, va='center', **kw)


def box(ax, x, y, w, h, fill='white', edge=LIGHT, radius=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f'round,pad=0,rounding_size={radius}', facecolor=fill,
        edgecolor=edge, linewidth=.75, zorder=1))


def arrow(ax, points, color=INK, dashed=False, width=1):
    path = MplPath(points, [MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=7,
        lw=width, color=color, linestyle=(0, (3, 2)) if dashed else 'solid', zorder=3))


def dot(ax, x, y, color=PURPLE, hollow=False):
    ax.add_patch(Circle((x, y), 2.5, facecolor='white' if hollow else color,
                        edgecolor=color, lw=.9, zorder=4))


def lock(ax, x, y, color=GRAY):
    ax.add_patch(Arc((x+3, y+3), 4.5, 6, theta1=180, theta2=360, color=color, lw=.8))
    box(ax, x, y+3, 6, 5, fill='white', edge=color, radius=.8)


def wave(ax, x, y):
    for i, h in enumerate([3, 7, 12, 8, 4, 10, 6]):
        ax.plot([x+2.5*i]*2, [y-h/2, y+h/2], color=GRAY, lw=.9, solid_capstyle='round')


def save(fig, name, title):
    fig.savefig(OUT/f'{name}.pdf', metadata={'Title': title})
    fig.savefig(OUT/f'{name}.svg')
    fig.savefig(OUT/f'{name}.png', dpi=300)
    plt.close(fig)


# Fig. 1: a frame-token boundary links movement to recognizer behavior.
fig, ax = canvas(242, 125)
text(ax, 121, 8, 'Shift magnitude ≠ recognition impact', size=9.5, weight='bold', ha='center')
for x, title in [(3, 'Large shift'), (128, 'Small shift')]:
    text(ax, x+54, 27, title, weight='bold', ha='center')
    ax.add_patch(Rectangle((x+2, 39), 108, 26, facecolor='#F5F5F7', edgecolor='none'))
    ax.add_patch(Rectangle((x+2, 65), 108, 27, facecolor=PALE, edgecolor='none'))
    ax.plot([x+2, x+110], [65, 65], color=GRAY, lw=.8, ls=(0,(3,2)))
    text(ax, x+7, 47, '“a”', size=8, color=GRAY)
    text(ax, x+7, 83, '“e”', size=8, color=PURPLE)
arrow(ax, [(29, 55), (98, 55)], GRAY, width=1.5)
dot(ax, 29, 55, GRAY, True); dot(ax, 98, 55)
arrow(ax, [(183, 57), (183, 76)], PURPLE, width=1.6)
dot(ax, 183, 57, GRAY, True); dot(ax, 183, 76)
text(ax, 57, 101, 'Same CTC token', size=8.7, color=GRAY, ha='center')
text(ax, 182, 101, 'Changed CTC token', size=8.7, color=PURPLE, ha='center')
dot(ax, 13, 118, GRAY, True); text(ax, 19, 118, 'Reference', size=8)
dot(ax, 83, 118); text(ax, 89, 118, 'Adapted', size=8)
ax.plot([146, 158], [118, 118], color=GRAY, lw=.8, ls=(0,(3,2)))
text(ax, 163, 118, 'Token boundary', size=8)
save(fig, 'dads_motivation', 'Shift magnitude and recognition impact: conceptual illustration')


# Fig. 2: two visual ideas, alignment and counterfactual intervention.
fig, ax = canvas(504, 200)
text(ax, 3, 9, 'Given adapted recognizer', weight='bold', size=10)
text(ax, 248, 9, 'Decision alignment', weight='bold', size=11, color=PURPLE)
text(ax, 248, 26, 'Weight each shift by recognizer sensitivity', size=8.6, color=GRAY)
# Same input, two feature sources. Frozen status applies after adaptation.
wave(ax, 3, 60); text(ax, 11, 77, '$x$', ha='center')
box(ax, 33, 28, 94, 27)
text(ax, 80, 37, 'Pretrained reference', size=8.3, ha='center')
text(ax, 80, 48, r'$f_r\;\rightarrow\;E_r$', size=9, ha='center', color=GRAY)
box(ax, 33, 69, 94, 27)
text(ax, 80, 78, 'Adapted encoder', size=8.5, ha='center')
text(ax, 80, 89, r'$f_a\;\rightarrow\;E_a$', size=9, ha='center', color=GRAY)
arrow(ax, [(21, 58), (26, 58), (26, 41), (32, 41)], GRAY)
arrow(ax, [(21, 62), (26, 62), (26, 82), (32, 82)], GRAY)
# Both representations feed their difference; no omitted head on the loss path.
arrow(ax, [(127, 41), (136, 41), (136, 37), (141, 37)], GRAY)
arrow(ax, [(127, 76), (141, 76), (141, 37)], GRAY)
arrow(ax, [(141, 37), (239, 37), (239, 67), (246, 67)], GRAY)
text(ax, 195, 24, r'$\Delta=E_a-E_r$', size=9, ha='center', color=GRAY)
box(ax, 153, 69, 74, 27)
lock(ax, 159, 78)
text(ax, 193, 78, 'CTC head', ha='center', size=8.6)
text(ax, 193, 89, r'$g_a$  frozen', ha='center', color=GRAY, size=8)
arrow(ax, [(127, 83), (152, 83)], GRAY)
box(ax, 153, 106, 74, 18)
text(ax, 190, 115, 'CTC loss', size=8.6, ha='center')
arrow(ax, [(190, 96), (190, 105)], GRAY)
text(ax, 34, 111, 'Training transcript', size=8, color=GRAY)
text(ax, 131, 115, '$y$', size=9)
arrow(ax, [(137, 115), (152, 115)], GRAY)
text(ax, 34, 123, 'Labels for calibration only', size=8, color=GRAY)
# A common set of dimensions, with exactly computed illustrative products.
delta = np.array([.9, .28, .65, .2, .75, .32])
grad = np.array([.08, .95, .65, .12, .15, .8])
product = delta*grad
selected = np.argsort(product)[-3:]

def bars(x, values, product_row=False):
    ax.plot([x-1, x+49], [80, 80], color=LIGHT, lw=.7)
    for i, val in enumerate(values):
        color = PURPLE if product_row and i in selected else ('#B8A6D8' if not product_row else LIGHT)
        ax.add_patch(Rectangle((x+8*i, 80-val*27), 5.5, val*27,
                              facecolor=color, edgecolor='none', zorder=3))
        if product_row and i in selected:
            ax.plot([x+8*i+1, x+8*i+2.5, x+8*i+5], [86, 87.5, 84.5], color=PURPLE, lw=.9)

for x, label in [(248, 'Shift'), (329, 'Sensitivity'), (420, 'Relevance')]:
    text(ax, x+23, 45, label, size=8.7, ha='center', weight='bold' if x==420 else 'normal')
bars(248, delta); bars(329, grad); bars(420, product/product.max(), True)
text(ax, 312, 67, '×', size=18, color=PURPLE, ha='center')
text(ax, 402, 67, '=', size=16, color=PURPLE, ha='center')
text(ax, 274, 94, r'$|\Delta|$', color=GRAY, ha='center')
text(ax, 355, 94, r'$|G|$', color=GRAY, ha='center')
text(ax, 447, 98, 'Average; top K', size=8.4, color=PURPLE, ha='center')
arrow(ax, [(227, 115), (355, 115), (355, 103)], PURPLE, dashed=True)
text(ax, 285, 109, 'CTC sensitivity', size=8, color=PURPLE, ha='center')
text(ax, 447, 115, 'One global mask', size=8.5, color=PURPLE, ha='center')
# Counterfactual intervention is a full-width visual stage, not a mask footnote.
ax.plot([3, 501], [136, 136], color=LIGHT, lw=.7)
text(ax, 3, 148, 'Counterfactual intervention', weight='bold', size=10)
text(ax, 501, 148, 'Frozen models · no retraining', size=8.5, ha='right', color=GRAY)
mask = [1 if i in selected else 0 for i in range(6)]
# The feature strip encodes source identity, not zeroing or dimension deletion.
for i, keep in enumerate(mask):
    x = 169+14*i
    ax.add_patch(Rectangle((x, 162), 10, 19, facecolor=PURPLE if keep else 'white',
                          edgecolor=PURPLE if keep else GRAY, linewidth=.9))
    if not keep:
        ax.plot([x+2, x+8], [177, 165], color=LIGHT, lw=.7)
ax.add_patch(Rectangle((4, 165), 7, 7, facecolor=PURPLE, edgecolor=PURPLE))
text(ax, 16, 169, 'Keep adapted values', size=8.5, color=PURPLE)
ax.add_patch(Rectangle((4, 180), 7, 7, facecolor='white', edgecolor=GRAY))
text(ax, 16, 184, 'Revert to reference', size=8.5, color=GRAY)
text(ax, 210, 193, r'$\widetilde E_m=E_r+m\odot\Delta$', size=9, ha='center')
arrow(ax, [(485, 115), (498, 115), (498, 130), (210, 130), (210, 161)], PURPLE)
# White underlay keeps the crossing connector out of the stage heading.
box(ax, 321, 161, 91, 25)
lock(ax, 328, 169)
text(ax, 373, 169, 'Same CTC head', size=8.4, ha='center')
text(ax, 370, 180, r'$g_a$  frozen', size=8, color=GRAY, ha='center')
arrow(ax, [(251, 172), (320, 172)])
arrow(ax, [(412, 172), (442, 172)])
# Speech-bubble glyph for recognizer output.
box(ax, 449, 160, 48, 25, edge=GRAY)
for yy, xx in [(167, 487), (173, 489), (179, 480)]:
    ax.plot([456, xx], [yy, yy], color=GRAY, lw=.8)
text(ax, 474, 194, 'Decode', size=8.5, ha='center')
save(fig, 'dads_overview', 'DADS: decision alignment and counterfactual intervention')
print(f'Saved motivation and method PDF/SVG/PNG figures in {OUT}')
