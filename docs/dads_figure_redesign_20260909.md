# 2026-09-09 — DADS Fig. 1 redesign

## 2026-09-10 — Editable PowerPoint delivery

Exported the current Comic Sans / gray-arrow overview to
`figures/dads_overview.pptx`, one slide at 14 × 5.556 inches with the same aspect
ratio as the paper figure. The reproducible converter is
`figures/export_dads_overview_pptx.py`. It reads the existing Matplotlib artists
and creates native DrawingML freeform shapes, Bezier curves, lines, editable text
boxes, and formula subscripts. No overview bitmap is embedded. Mathematical
symbols unavailable in Comic Sans use DejaVu Sans. There are 275 editable
objects, including 44 text boxes; diagonal reference markings are editable lines.

The workspace dependency-loader tool was unavailable. Installed python-pptx and
its dependencies into the isolated `/tmp/dads-pptx-runtime` directory, then ran:

```bash
PYTHONPATH=/tmp/dads-pptx-runtime MPLCONFIGDIR=/tmp/dads-mpl python3 ICASSP2027_Paper_Templates/figures/export_dads_overview_pptx.py
```

Opened the PPTX through LibreOffice's headless PDF converter and inspected its
rendering. Explicit geometry fixes preserve unfilled arrow shafts, filled
arrowheads, and reference hatching, and prevent theme shadows. All content and
paper files retain their existing scientific meaning. The slide notes identify
the editable components and source scripts.

## 2026-09-10 — Comic lettering and gray arrows

Applied the author's requested style-only revision to the single overview:
Comic Sans MS regular/bold for lettering and available mathematical glyphs,
uniform gray arrows (including the calibration legend and outlined keep/revert
arrows), and a 30% increase in stroke widths. Existing feature colors and layout
are preserved. The font is loaded explicitly from the installed Microsoft core
fonts; unsupported mathematical symbols use STIX fallback. Regenerated PDF,
SVG, PNG, and the manuscript with the reproduction commands below. Inspected
the image for text collisions and arrow colors; Comic Sans fonts are embedded
in the PDF. The manuscript remains five pages with no undefined-reference or
overfull warnings. Previous renderer and figure PDF are retained in
`figures/archive_before_comic/`. No method, result, or manuscript text changed.

## Purpose and design

Follow the author's supplied editorial notes: one double-column figure with a
roughly 32% conceptual motivation panel and 68% method overview. The two
coordinate examples are schematic, not measurements. Large movement with low
sensitivity and small movement with high sensitivity motivate decision-aware
selection. Purple (#6B45B5) identifies Delta, the gradient, utility, the mask,
and retained dimensions; existing model components are neutral gray and black.

The former three equally weighted panels are replaced by a dominant calibration
module and a single small keep/revert strip. Both encoders consume the same
utterance; adapted features supply the frozen CTC objective and Delta. The
gradient is computed at full adapted features. The mean in the figure is over
all valid frames of the calibration utterances, consistent with Eq. (5), taking
absolute products before averaging. Top-K produces one mask, reused at decoding.
The caption explains that white columns restore reference values and labels are
needed only for calibration. No model speedup or empirical statistic is implied
by the diagram.

## Files and reproduction

The active manuscript is ICASSP2027_Paper_Templates/Template.tex. The former
paper_draft_icassp directory was absent at the start of this task. All new figure
assets are therefore maintained directly in the template directory.

- figures/render_dads_overview.py: editable Matplotlib source.
- figures/dads_overview.pdf: embedded-font vector figure, 7 × 2.878 inches.
- figures/dads_overview.svg: vector editor version with text preserved as text.
- figures/dads_overview.png: 300 dpi preview.
- figures/archive_before_motivation/: original figure, manuscript, and PDF.

Run from the repository root:

```bash
MPLCONFIGDIR=/tmp/dads-mpl python3 ICASSP2027_Paper_Templates/figures/render_dads_overview.py
bash ICASSP2027_Paper_Templates/build.sh
```

Only the figure caption and the corresponding panel reference changed in the
manuscript. The original introduction, experiments, conclusion, other figures,
and bibliography were preserved.

## Verification and outcome

- Viewed both the standalone preview and the final page-2 rendering.
- Normal labels are 9.1 pt at 7-inch export width; headings are 10.15 pt.
- Fonts are embedded in the vector PDF. qpdf reports no syntax/stream errors
  for either the figure or normalized manuscript.
- Final LaTeX pass has no undefined-reference or overfull warnings.
- Manuscript remains five pages; references begin on page five.
- No training or evaluation was run. This is a presentation update, not new
  evidence about the method. Subsequent visual revisions can use the SVG or
  the plotting script.

## 2026-09-09 follow-up — separate motivation and method

The author's new request supersedes the combined 32%/68% layout above. The
attached critique is design context; the requested work is to split the figure,
place motivation in the introduction, and explain the method with graphical
marks and concise notation.

### Design and changes

- Fig. 1 (`dads_motivation`) is a single-column conceptual illustration on page 1.
  Two schematic CTC frame-token regions show a large shift parallel to a boundary
  and a smaller shift crossing it. These are illustrative geometries, not measured
  feature projections or claims about utterance-level decoding. The legend uses
  an open reference point and a purple adapted point consistently.
- Fig. 2 (`dads_overview`) is a full-width method illustration at the top of page 2.
  The visual center is **Shift × Sensitivity = Relevance**, with corresponding
  dimensions represented by bars. Example products are computed from the two
  illustrative arrays; heights are normalized within each strip for visibility.
  Averaging occurs after the absolute framewise product, as in the manuscript.
  These bars are not experimental statistics.
- A waveform marks the common input, lock glyphs mark the adapted CTC head, and
  a text-output glyph marks decoding. The explicit path is adapted features →
  frozen adapted head → CTC loss → feature sensitivity. Labels are calibration
  only. No gradient formula or large scoring container occupies the diagram.
- Counterfactual intervention has its own full-width row. Purple dimensions
  retain adapted values; outlined dimensions restore reference values. Feature
  dimensionality remains unchanged. The same adapted head decodes without
  retraining. The figure does not assert computational pruning or guaranteed
  recognition preservation for any budget.
- The long utility equation remains in the method text. The overview retains
  short delta and intervention identities; the caption defines the gradient.
  The frozen status applies to DADS after adaptation, not to fine-tuning.
- The manuscript now includes vector PDFs instead of `figures/image.png`.
  Figure references and captions were updated. Three introductory paragraphs
  were shortened to remove explanation duplicated by the figures and preserve
  four technical pages. The related-work paragraphs, contribution list, method
  equations, experiments, tables, conclusions, and bibliography are preserved.
- LaTeX must encounter a two-column top float on the preceding page. Its source
  declaration is therefore queued within the introduction, with an explanatory
  comment; its rendered position is directly above Section 2 on page 2.

### Graphical sources and reproduction

Reviewed [Lucide](https://lucide.dev/) and its
[license](https://github.com/lucide-icons/lucide/blob/main/LICENSE) for simple
line-icon conventions. The delivered waveform, locks, strips, and output glyph
are original Matplotlib primitives; no external SVG paths or raster assets were
copied or embedded. Both figures remain editable through the Python source and
text-preserving SVG exports.

```bash
MPLCONFIGDIR=/tmp/dads-mpl python3 ICASSP2027_Paper_Templates/figures/render_dads_overview.py
bash ICASSP2027_Paper_Templates/build.sh
```

The immediately preceding manuscript, PDF, and renderer are saved in
`ICASSP2027_Paper_Templates/figures/archive_before_split/`.

### Validation and outcome

The initial layout spilled to six pages and delayed Fig. 2 to page 3. After
shortening the repeated introductory explanation and queuing the wide float,
the compiled manuscript is **five pages: four technical pages plus references
on page 5**. Viewed the final page 1, page 2, and page 4 renderings. Normal figure
labels are 8–9 pt at the designed publication dimensions; no font shrinking or
template margin changes were used. Both vector PDFs have embedded fonts and
pass `qpdf --check`; the final LaTeX log has no undefined references or overfull
boxes. Text from “Both encoders receive…” through the end of the manuscript is
identical to the pre-edit snapshot. No experiment or evaluation was run, and
this presentation update adds no empirical evidence.

## 2026-09-09 follow-up — one overview in the supplied reference style

The author superseded the split-figure request: deliver one overview, no
motivation figure, using the supplied prototype-calibration diagram as a
structural and aesthetic reference. At the start of this revision, the current
manuscript already omitted the motivation figure and contained further author
edits to the introduction. Those edits were preserved; only the overview caption
changed in `Template.tex` during this revision.

The new single `dads_overview` uses three soft rounded regions (pale blue,
cream, and sage), serif headings, locked trapezoid encoders, small feature grids,
colored feature ribbons, and an outlined keep/revert merge. These are original
Matplotlib vector primitives inspired by the supplied figure; no scientific
content, labels, or assets from that reference were imported.

- Left: same speech through the given reference and adapted encoders, with the
  explicit adapted-feature → frozen adapted CTC head → labeled CTC loss path.
- Center (a): representation shifts and feature gradients meet at the absolute
  elementwise product. All six schematic products remain visible before frame
  averaging. The three selected dimensions are shown only in the global mask,
  preventing any suggestion that selection happens before aggregation. The
  short gradient definition specifies differentiation with respect to adapted
  features, and the mean explicitly names calibration frames.
- Right (b): retained dimensions use adapted values, and reverted dimensions use
  reference values. The six coordinate positions stay fixed. Teal means keep;
  gray hatching means restore. The same adapted head decodes without retraining.
- The glyphs are schematic; the caption defines valid-frame aggregation, the
  shared feature dimensions, and calibration-only labels/gradients. There is no
  motivation panel, result chart, or claim of encoder acceleration.

`render_dads_overview.py` now renders only the single overview, producing PDF,
SVG with editable text, and PNG. Existing motivation assets are unused. The
immediately preceding manuscript, PDF, renderer, and vector overview are retained
in `figures/archive_before_reference_style/`.

The full-width vector figure is 7 × 2.778 inches. Vertical spacing was tightened
without shrinking text to preserve the paper's four technical pages. The final
manuscript has five pages; Fig. 1 is at the top of page 2 with the method, and
references start on page 5. The caption and feature-gradient/aggregation labels
were reviewed for scientific clarity. Viewed the standalone figure and final
page-2 rendering; PDF fonts are embedded and the LaTeX build has no undefined
references or overfull boxes. Figure and manuscript PDF checks passed. No
experiments were run and no empirical result was changed.

Reproduction remains:

```bash
MPLCONFIGDIR=/tmp/dads-mpl python3 ICASSP2027_Paper_Templates/figures/render_dads_overview.py
bash ICASSP2027_Paper_Templates/build.sh
```
