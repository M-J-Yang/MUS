# 2026-09-09 — DADS Fig. 1 redesign

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
