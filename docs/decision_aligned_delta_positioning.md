# Decision-Aligned Delta Selection: research positioning and frozen claims

## Primary-source links used for the matrix

- Mind the Shift: <https://arxiv.org/abs/2601.20142>
- Chiu et al., Learnable Layer Selection and Model Fusion: <https://www.isca-archive.org/interspeech_2024/chiu24_interspeech.html>
- Selective Attention / HSA Merge project: <https://github.com/balaji1312/hsa_merge>
- AudioSAE: <https://aclanthology.org/2026.eacl-long.149/>
- SADI: <https://proceedings.iclr.cc/paper_files/paper/2025/hash/c4d26a95fd83f8e590f81c54ae670b5d-Abstract-Conference.html>
- AUSteer: <https://proceedings.iclr.cc/paper_files/paper/2026/hash/423d0909791493b7c10916fd328c2913-Abstract-Conference.html>
- DAREx: <https://proceedings.iclr.cc/paper_files/paper/2025/hash/d0074bea472f8b9b839fa2d50ce67595-Abstract-Conference.html>
- TIES-Merging (background on selective Delta parameters): <https://proceedings.neurips.cc/paper_files/paper/2023/hash/1644c9af28ab7916874f6fd6228a9bcf-Abstract-Conference.html>

**Status:** claim and protocol record, 2026-08-29. This document records a literature-to-method analysis; it reports no new ASR experiment.

## Decision

The working method name is **Decision-Aligned Delta Selection (DADS)**. Its central object is an input-dependent representation delta, not a parameter delta:

$$\Delta E^{(l)}(x)=E^{(l)}_{ft}(x)-E^{(l)}_{pt}(x).$$

The paper must not present layer selection, generic dimension selection, or delta sparsification as the main novelty. The primary claim is narrower:

> Within a fixed representation-delta layer, DADS ranks Delta coordinates by their consistent signed contribution to actual CTC target-versus-competitor decisions, then evaluates a freshly trained ASR fusion model using the selected coordinates.

Layer scanning is retained as a diagnostic that motivates the fixed layer; it is not part of the method name or principal novelty.

## Related-work comparison matrix

| Work | Problem / observation | Method and utility criterion | Granularity / data | Difference from DADS |
|---|---|---|---|---|
| Mind the Shift (2026) | Fine-tuning shifts can complement a second SSL representation for child ASR. | Representation Delta $E_{ft}-E_{pt}$ is concatenated as a full embedding; no coordinate utility ranking. | Representation layer; MyST ASR. | Direct predecessor, but treats all coordinates in the chosen Delta uniformly. DADS selects coordinates by CTC decision contribution. |
| Chiu et al. (Interspeech 2024) | A final SSL layer is not always best. | Gumbel-Softmax layer selection and dimension-wise choice of a source layer. | Layer/source-coordinate; speech SSL. | Chooses where each representation coordinate comes from. DADS evaluates whether an already formed fine-tuning Delta coordinate helps a CTC decision. |
| Selective Attention Merging (ICASSP 2025) | Domain-adaptation changes are not uniformly distributed in parameters. | Selectively merges attention-module task-vector parameters. | Parameter-space attention modules; MyST ASR. | Parameter Delta selection and parameter merging, rather than input-dependent representation Delta selection or CTC-margin attribution. |
| HSA Merge (2026) | Adaptation can concentrate in salient attention heads. | Restricts task-vector merging to selected Q/K/V attention heads. | Parameter-space heads; ASR domain adaptation. | Selects parameter heads; DADS selects representation coordinates conditioned on aligned CTC decisions. |
| AudioSAE (EACL 2026) | Dense audio activations contain sparse, interpretable factors. | Learns sparse autoencoders, studies feature stability and concept ablation/steering. | Learned SAE features across audio-model layers. | Uses learned latent factors for interpretability/steering; DADS retains native representation-Delta coordinates using recognition-margin evidence. |
| SADI (ICLR 2025) | Whole activation steering is too coarse. | Contrastive examples localize critical elements for dynamic steering. | LLM activations: heads, states, neurons. | Its criterion is behavioral contrast between examples; DADS uses supervised CTC forced alignment and the strongest frame-level competitor. |
| AUSteer (ICLR 2026) | Activation dimensions can be beneficial, irrelevant, or harmful. | Contrastive activation-direction consistency identifies atomic units for steering. | LLM activation dimensions. | Closest conceptual neighbor. DADS evaluates representation Delta, not raw activations, and its consistency is CTC target-vs-competitor margin consistency, including blank competition. |
| DAREx (ICLR 2025) | Random Delta-parameter pruning can fail for large/aggressive pruning. | Importance-based pruning of fine-tuning parameter deltas. | Parameter-space model updates. | Same selective-adaptation intuition, but no input-dependent representation Delta, CTC alignment, or coordinate-level output-margin utility. |

## Claims to make and claims to avoid

### Defensible claims

- DADS selects **representation-Delta coordinates** using a CTC decision-aligned criterion.
- A coordinate is scored by whether it consistently increases the aligned target token logit relative to the strongest actual competitor.
- The criterion is different from shift magnitude, variance, source-layer selection, parameter-delta pruning, and SAE feature discovery.
- The blank/non-blank competitor split creates a CTC-specific analysis of token emission versus token discrimination.

### Claims to avoid

- Do not claim the first speech layer-selection method or the first dimension-wise speech selection method.
- Do not claim that different hidden dimensions have unequal importance as a standalone discovery.
- Do not call a coordinate a semantic feature or assume that a coordinate is an independent computational feature.
- Do not claim that this is the first selective fine-tuning Delta method; parameter-space methods already do this.

Use **Delta coordinate** or **Delta dimension** throughout the paper.

## Attribution-head requirement

The source-compatible Mind-the-Shift fusion baseline is `LayerNorm(2048) -> Dropout(0.1) -> Linear CTC`. In that architecture the term

$$q_{t,i}=\delta_{t,i}(W^\Delta_{a_t,i}-W^\Delta_{c_t,i})$$

is not an exact coordinate decomposition of a logit margin: LayerNorm makes each output depend on all input coordinates. DADS therefore uses two explicitly separated protocols:

1. **Reproduction baseline:** preserve the source-compatible LayerNorm, dropout, and linear head to reproduce Mind the Shift.
2. **DADS diagnostic, attribution, and selection protocol:** use pure-linear frozen-encoder CTC heads for the layer scan, the full-Delta teacher, and every selected-mask ablation. For the teacher, $z_t = W [E_ref,t; Delta_t] + b$, so the displayed $q_{t,i}$ is exactly additive in the target-versus-competitor margin.

All Random-K, Magnitude-K, Utility-K, Full-Delta, Drop-Best, and Drop-Worst comparisons used to support DADS must be trained and compared within the pure-linear protocol. The source-compatible result remains a separate baseline-validation result, not an attribution result.

## Frozen DADS protocol

1. Freeze WavLM-ft and W2V2-ft/W2V2-pt encoders after independent ASR fine-tuning.
2. Scan W2V2 Delta layers with pure-linear heads; select $l^*$ by development WER. Report normalized shift magnitude as the first magnitude-versus-utility diagnostic.
3. At $l^*$, train the pure-linear full-Delta teacher on the training split.
4. On the utility-estimation split, compute Viterbi CTC forced alignment. Keep only frames with non-blank aligned target $a_t$.
5. Let $c_t=argmax_{k != a_t} z_{t,k}$ over the complete vocabulary, including blank. Score every coordinate by $q_{t,i}$ and $U_i = mean(sign(q_{t,i}))$.
6. Freeze a Top-K mask. Train a newly initialized pure-linear CTC head on the masked representation. Select K only on development data; use test once after the protocol is frozen.

To prevent selection leakage in a final paper, derive the utility ranking from a held-out subset of training data (or cross-fitting), rather than from the same examples used to fit the attribution teacher. The exact split must be pre-registered in the experiment manifest.

## CTC-specific blank versus token analysis

Partition aligned non-blank frames:

- `A_blank = {t : a_t != blank and c_t = blank}`: coordinates that principally help emit the target token instead of blank.
- `A_token = {t : a_t != blank and c_t != blank}`: coordinates that principally discriminate the target token from another token.

Compute $U_i^blank$ and $U_i^token$ by applying the same signed-contribution mean to each subset. Report:

- the fraction of aligned decisions in each subset;
- Spearman correlation between the two utility vectors;
- Top-K Jaccard overlap;
- WER of masks selected from each utility vector; and
- a combined-utility mask as the primary method.

This experiment is optional for the first baseline run but is a high-value DADS ablation because blank is specific to CTC decoding.

## Required evidence chain

| Question | Required measurement | Preferred result |
|---|---|---|
| Is shift magnitude a utility score? | per-layer $S_l$ versus dev WER; per-coordinate magnitude versus $U_i$ correlation | weak correspondence |
| Does DADS capture task information? | Utility-K vs mean/std over multiple Random-K masks | lower WER for Utility-K |
| Is magnitude enough? | Utility-K vs Magnitude-K | lower WER for Utility-K |
| Is selection merely compression? | Utility-K vs Full-Delta | Utility-K can match or improve Full-Delta |
| Are high-/low-utility coordinates causally asymmetric? | Drop-Best and Drop-Worst from full Delta | dropping best hurts more; dropping worst can help |
| Does CTC blank competition matter? | blank/token utility overlap and separate masks | distinct or complementary masks |

Use the WER direction consistently: **smaller is better**. Thus the desired statement is `WER(Utility-K) < WER(Random-K)` rather than `Utility-K > Random-K`.

## Documentation outcome

**Method used:** comparison of the eight closest works by object (parameter Delta, representation Delta, or activation), selection criterion, granularity, and downstream task. The existing source-compatible baseline implementation was also checked against the algebra required by the proposed attribution.

**Effect:** the paper's novelty is now constrained to decision-aligned, cross-decision-consistent ranking of representation-Delta coordinates for CTC ASR. The attribution teacher is specified as pure linear so the reported contribution score has the claimed mathematical meaning.

## Reconciliation of the 2026-08-29 literature scan

A second structured review of the direct predecessor, speech neighbors, LLM activation-selection methods, and parameter-Delta methods reaches the same boundary as this document. It adds one useful emphasis: unlike a static parameter task vector $\Delta\theta$, a representation Delta is input-conditioned, $\Delta E(x)=E_{ft}(x)-E_{pt}(x)$. DADS therefore studies **input-conditioned representation-adaptation utility**, not model-update compression.

The name is frozen as **Decision-Aligned Delta Selection (DADS)** because it names the operative selection procedure. **CTC decision-aligned Delta utility** is the name of its scoring principle, not a competing method acronym. This retains the CTC-specific meaning without changing the manuscript name again.

The scan confirms the following final novelty sentence:

> Existing Delta-SSL fusion treats the coordinates of a representation shift uniformly; DADS measures whether individual input-conditioned Delta coordinates consistently enlarge the correct CTC token margin over the strongest actual competitor.

The CTC-specific blank/token split is retained as a required high-value analysis:

- `U_blank`: target-token versus blank competition, interpreted as emission support.
- `U_token`: target-token versus non-blank-token competition, interpreted as token discrimination support.

Do not describe this as a literal linear decomposition into semantic subrepresentations. It is an empirical partition of aligned decision contexts. Report Top-K Jaccard overlap, rank correlation, subset sizes, and retrained-mask WERs.

## Mathematical audit outcome

For a pure linear readout $z_t=W[E_t^{ref};\Delta_t]+b$, the target-versus-competitor margin is

$$z_{t,a_t}-z_{t,c_t}=\text{constant w.r.t. Delta coordinate }i + \sum_i \delta_{t,i}(W^\Delta_{a_t,i}-W^\Delta_{c_t,i}).$$

Thus $q_{t,i}=\delta_{t,i}(W^\Delta_{a_t,i}-W^\Delta_{c_t,i})$ is exactly the additive contribution of coordinate $i$ to that margin. It is not exact after input LayerNorm, because LayerNorm couples all coordinates through its frame mean and variance. The two-head protocol in this document is therefore mandatory, not optional:

For rigor, $c_t$ is selected once from the full teacher logits and then held fixed while decomposing that observed target-versus-competitor margin. Therefore $q_{t,i}$ is an exact **local attribution** for that fixed pair, not a claim that masking coordinate $i$ leaves the strongest competitor unchanged. The freshly retrained masked-model WER and Drop-Best/Drop-Worst experiments supply the required empirical test beyond this local attribution.

1. source-compatible LayerNorm -> Dropout -> Linear CTC for the Mind-the-Shift reproduction;
2. pure-linear full-Delta teacher and pure-linear retrained selection heads for DADS attribution and ablations.

**Effect.** The new review does not broaden the method. It confirms a sharper contribution: representation Delta + CTC decision geometry + cross-decision directional consistency, with blank/token competition as the ASR-specific analysis.

## Insight-first narrative — frozen

The paper is not framed as a collection of layer selection, feature selection, and pruning modules. Its single scientific question is:

> **When fine-tuning changes an SSL representation, which changes actually matter for ASR?**

The central hypothesis is **shift magnitude is not shift utility**. DADS is the natural measurement and selection mechanism that follows from this question, rather than the headline insight by itself.

### Three empirical insights to establish

1. **Layer level:** a large normalized representation shift $S_l$ need not yield low downstream fusion WER. Report the selected layer, $\arg\max_l S_l$, and a rank correlation between $S_l$ and utility (for example $-\mathrm{WER}_l$); do not describe this as a new generic layer-selection method.
2. **Coordinate level:** mean coordinate magnitude $m_i=\mathbb{E}|\delta_i|$ need not predict decision utility $U_i$. Show helpful large/small coordinates and harmful large coordinates; report Spearman correlation between $m_i$ and $U_i$.
3. **Intervention:** masks selected by decision utility should outperform magnitude-matched and random masks, and can outperform Full Delta. These are WER comparisons, so the desired direction is lower WER.

### Narrative flow

`Mind the Shift` shows that a representation shift can be useful. We ask whether every component of that shift is useful. The two diagnostic levels show that the amount of change is an unreliable proxy for its ASR value. DADS therefore measures usefulness in the geometry of actual CTC decisions and selects the coordinates with stable positive margin contribution.

Layer-level analysis remains in the paper because it supports the central magnitude-versus-utility insight. It is no longer advertised as an independent novelty or as an attempt to beat prior generic layer-selection methods.

### Title and terminology

The working manuscript title is **Beyond Shift Magnitude: What Makes a Fine-Tuning Shift Useful for ASR?** The method remains **Decision-Aligned Delta Selection (DADS)**. Use `Delta coordinate` and `decision utility`; reserve `feature` for citations that use that terminology.

### Required figure and tables

- **Figure 1:** layer index versus normalized shift magnitude and development fusion WER, with the selected layer marked.
- **Figure 2:** coordinate magnitude versus $U_i$, with exemplar large-harmful and small-helpful coordinates if observed.
- **Table 1:** source-compatible Mind-the-Shift baseline reproduction across Full/10h/5h/1h.
- **Table 2:** pure-linear DADS Full, Random-K, Magnitude-K, and Utility-K WERs.
- **Table 3:** Drop-Best / Drop-Worst and blank-versus-token utility analysis.

**Effect.** This framing preserves the rich two-level evidence chain without turning it into a collection of defensive novelty claims. It makes the experimental question, the CTC-margin criterion, and the selected-Delta intervention all answer one coherent insight.

## Story-consistency audit — 2026-08-29

**Verdict: consistent after one protocol correction.** The manuscript and frozen protocol now implement the story in the following order:

1. Mind the Shift establishes that a representation Delta can contain useful adaptation information.
2. The paper asks whether the *amount* of fine-tuning change is a valid proxy for its ASR value.
3. A layer-level diagnostic and a coordinate-level analysis supply two instances of the same magnitude-versus-utility question.
4. DADS defines utility by the stable signed contribution to an aligned CTC target-versus-competitor margin.
5. Retrained Utility-K, Random-K, Magnitude-K, Full-Delta, Drop-Best, Drop-Worst, and blank/token analyses test the insight rather than a pruning artifact.

The original draft had one inconsistency: source-compatible LayerNorm heads were used for the layer scan while DADS attribution used a pure-linear teacher. This was corrected. The final-layer Mind-the-Shift reproduction alone uses LayerNorm -> Dropout -> Linear; all DADS diagnostics, attribution, and selection comparisons use pure-linear CTC heads.

Two rigor additions remain intentional and are compatible with the story:

- derive utility on a held-out training subset, avoiding selection leakage;
- fix the full-teacher competitor during each local margin decomposition, then use retrained-mask WER and drop ablations to test the resulting intervention.

**Effect.** The paper has one scientific story, not two unrelated stages: `change is not useful change -> measure utility in CTC decision geometry -> retain consistently useful shifts -> test whether less, selected adaptation is better`.
