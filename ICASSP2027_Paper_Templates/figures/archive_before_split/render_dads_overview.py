"""Publication-sized vector overview; run with Python and Matplotlib.

The left panel is conceptual, not experimental data. Coordinates are in design
points; the 720-point canvas is exported at 7 inches for the paper. Minimum
label size is 13 design points (9.1 points at the exported size).
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.path import Path as MplPath

OUT = Path(__file__).resolve().parent
SCALE = 0.7
W, H = 720, 324
EXPORT_HEIGHT = 296
PURPLE, PALE = "#6B45B5", "#EEE9F7"
INK, GRAY, LIGHT = "#171717", "#777777", "#D8D8DE"
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
fig = plt.figure(figsize=(W*SCALE/72, EXPORT_HEIGHT*SCALE/72), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W), ylim=(H, 0))
ax.axis("off")


def text(x, y, value, size=13, color=INK, weight="normal", ha="left", **kw):
    return ax.text(x, y, value, fontsize=size*SCALE, color=color,
                   weight=weight, ha=ha, va="center", **kw)


def box(x, y, w, h, fill="white", edge=LIGHT, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=3",
                 facecolor=fill, edgecolor=edge, linewidth=0.9*SCALE,
                 linestyle=(0, (4, 3)) if dashed else "solid", zorder=1))


def arrow(points, color=INK, dashed=False, width=1.35):
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>",
                 mutation_scale=8*SCALE, lw=width*SCALE, color=color,
                 linestyle=(0, (4, 3)) if dashed else "solid", zorder=2))


def dot(x, y, color, open_circle=False):
    ax.add_patch(Circle((x, y), 3.5, facecolor="white" if open_circle else color,
                        edgecolor=color, lw=SCALE, zorder=4))


# Two clearly unequal visual regions: motivation, then the contribution path.
text(8, 15, "(a) Why magnitude fails", size=14.5, weight="bold")
text(240, 15, "(b) Decision-aligned delta selection", size=14.5, weight="bold")
ax.plot([227, 227], [6, 316], color=LIGHT, lw=0.8*SCALE)

text(12, 49, "Dimension A", weight="bold")
arrow([(20, 77), (179, 77)], GRAY)
dot(20, 77, GRAY, True)
dot(174, 77, GRAY)
text(12, 99, "Large shift · low sensitivity", color=GRAY)
box(12, 112, 99, 23, fill="#F2F2F4", edge="#F2F2F4")
text(61.5, 123.5, "Low utility", ha="center", color=GRAY, weight="bold")

text(12, 164, "Dimension B", weight="bold")
arrow([(20, 192), (77, 192)], PURPLE, width=2.2)
dot(20, 192, GRAY, True)
dot(72, 192, PURPLE)
text(12, 214, "Small shift · high sensitivity", color=PURPLE)
box(12, 227, 104, 23, fill=PALE, edge=PALE)
text(64, 238.5, "High utility", ha="center", color=PURPLE, weight="bold")
text(12, 278, "Movement × sensitivity", weight="bold")
text(12, 299, "→ decision importance", color=PURPLE, weight="bold")

# Encoders are fixed components; the same x supplies both.
text(257, 43, "Frozen encoders", color=GRAY)
box(260, 60, 90, 40)
text(305, 73, "Adapted", ha="center")
text(305, 90, r"$f_a(x)=E_a$", ha="center")
box(260, 112, 90, 40)
text(305, 125, "Reference", ha="center")
text(305, 142, r"$f_r(x)=E_r$", ha="center")
text(241, 105, r"$x$", ha="center")
arrow([(248, 100), (251, 100), (251, 80), (259, 80)], GRAY)
arrow([(248, 110), (251, 110), (251, 132), (259, 132)], GRAY)
box(260, 186, 90, 41, PALE, PURPLE)
text(305, 198, "Shift", color=PURPLE, ha="center", weight="bold")
text(305, 216, r"$\Delta=E_a-E_r$", color=PURPLE, ha="center")
arrow([(350, 80), (363, 80), (363, 175), (332, 175), (332, 185)], GRAY)
arrow([(305, 152), (305, 185)], GRAY)

# The only dashed scope: labels and gradients are needed for calibration only.
box(380, 42, 332, 193, fill="#FCFBFE", edge=PURPLE, dashed=True)
text(396, 59, "DADS scoring", color=PURPLE, weight="bold")
text(396, 78, "Labeled training utterances only", color=GRAY)
text(476, 98, r"$y$", ha="center")
arrow([(476, 105), (476, 113)], GRAY)
box(397, 114, 162, 40)
text(478, 127, "Adapted CTC loss", ha="center")
text(478, 144, r"$\ell\,(g_a(E_a),y)$", ha="center")
arrow([(363, 132), (396, 132)], GRAY)
ax.add_patch(Circle((363, 132), 1.8, color=GRAY, zorder=3))
text(374, 120, r"$E_a$", ha="center", color=GRAY)
arrow([(559, 134), (589, 134)], PURPLE, dashed=True)
text(647, 132, r"$G=\frac{\partial\ell}{\partial E_a}$", color=PURPLE, ha="center", size=15)
text(647, 158, "Sensitivity", ha="center", color=PURPLE)
box(397, 186, 168, 41, PALE, PURPLE)
text(481, 198, "Utility per dimension", color=PURPLE, ha="center")
text(481, 216, r"$U_i=\mathrm{mean}_{x,t}|\Delta_{t,i}G_{t,i}|$", color=PURPLE, ha="center")
arrow([(350, 206), (396, 206)], PURPLE, width=1.8)
arrow([(647, 168), (647, 176), (541, 176), (541, 185)], PURPLE, dashed=True)
box(607, 186, 88, 41, PURPLE, PURPLE)
text(651, 200, "Top-K", ha="center", color="white", weight="bold")
text(651, 217, r"$m$", ha="center", color="white", weight="bold")
arrow([(565, 206), (606, 206)], PURPLE, width=1.8)

# A single intervention strip replaces the old three-matrix panel.
arrow([(305, 227), (305, 266)], PURPLE)
arrow([(651, 227), (651, 249), (430, 249), (430, 266)], PURPLE, width=1.8)
text(485, 243, "Global mask", color=PURPLE)
box(254, 267, 235, 48)
text(371.5, 280, r"$\widetilde E_m=E_r+m\odot\Delta$", ha="center", size=14)
mask = [1, 1, 0, 1, 0, 0, 1, 0]
for i, keep in enumerate(mask):
    ax.add_patch(Rectangle((265+11*i, 295), 9, 10, facecolor=PURPLE if keep else "white",
                          edgecolor=PURPLE if keep else GRAY, linewidth=0.7*SCALE))
text(365, 300, "Keep / revert", color=GRAY)
box(535, 268, 101, 44)
text(585.5, 281, "CTC head", ha="center")
text(585.5, 299, r"$g_a$  (frozen)", ha="center", color=GRAY)
arrow([(489, 290), (534, 290)])
arrow([(636, 290), (679, 290)])
text(695, 288, r"$\hat y$", size=16, ha="center")
text(673, 310, "Greedy", color=GRAY, ha="center")

fig.savefig(OUT / "dads_overview.pdf", metadata={"Title": "Motivation and overview of DADS"})
fig.savefig(OUT / "dads_overview.svg")
fig.savefig(OUT / "dads_overview.png", dpi=300)
plt.close(fig)
print(f"Saved vector PDF, editable SVG, and PNG in {OUT}")
