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

**Status:** updated 2026-09-02 after the formal UT8 Fold 0 run. The frozen-head Taylor-utility intervention is the primary protocol; the earlier retrained-head result remains secondary.

## Decision

The working method name is **Decision-Aligned Delta Selection (DADS)**. Its central object is an input-dependent representation delta, not a parameter delta:

$$\Delta E^{(l)}(x)=E^{(l)}_{ft}(x)-E^{(l)}_{pt}(x).$$

The paper must not present layer selection, generic dimension selection, or delta sparsification as the main novelty. The primary claim is now narrower and directly tested by the formal Fold 0 intervention:

> Within a fixed representation-delta layer, Taylor utility ranks input-conditioned Delta coordinates by their contribution to the true fine-tuned CTC objective; masking half of the coordinates and reusing the original frozen fine-tuned CTC head retains 83.5% of the adaptation gain on test.

The earlier selected-representation-plus-retrained-head experiment is a secondary protocol. It answers whether the selected representation remains informative after readout adaptation, not whether the original fine-tuned behavior survives direct shift pruning.

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

- The primary formal protocol selects **representation-Delta coordinates** using Taylor utility from the frozen fine-tuned CTC loss.
- The primary intervention reuses the original fine-tuned CTC head; the selected mask is not followed by head retraining.
- The earlier target-vs-competitor margin score is retained only for the explicitly labeled secondary DADS attribution protocol.
- Both criteria are distinct from shift magnitude, variance, source-layer selection, parameter-delta pruning, and SAE feature discovery; the blank/non-blank split remains an auxiliary CTC analysis.

### Claims to avoid

- Do not claim the first speech layer-selection method or the first dimension-wise speech selection method.
- Do not claim that different hidden dimensions have unequal importance as a standalone discovery.
- Do not call a coordinate a semantic feature or assume that a coordinate is an independent computational feature.
- Do not claim that this is the first selective fine-tuning Delta method; parameter-space methods already do this.

Use **Delta coordinate** or **Delta dimension** throughout the paper.

## Primary and secondary protocols

The formal Fold 0 result fixes the fine-tuned Wav2Vec2-Large-960h model and its original CTC head. For the primary intervention, define

$$E_m(x)=E_0(x)+m\odot\Delta(x),\qquad \Delta(x)=E_{ft}(x)-E_{pt}(x).$$

The no-shift, full-shift, and Top-512 conditions all use the same frozen fine-tuned CTC head. Taylor utility is computed from the real CTC loss on the held-out `train_utility` split:

$$U_i=\mathbb{E}\left|\Delta_i\,\frac{\partial L_{CTC}}{\partial\Delta_i}\right|.$$

This is the primary pruning claim because the only intervention after fine-tuning is the shift mask.

The earlier exploratory DADS protocol remains useful but is explicitly secondary:

1. **Secondary retrained-head protocol:** select a representation using a newly trained readout and retrain a downstream CTC head after coordinate selection. Its earlier 98.8% gain-retention result measures retained representational information under readout adaptation.
2. **Primary frozen-head protocol:** use the same fine-tuned model, original CTC head, and direct $E_0+M_K\odot\Delta$ intervention. Its formal test result is 83.5% gain retention at $K=512/1024$.

The two percentages must not be pooled or presented as measurements of the same intervention. The identity check $E_0+\Delta=E_{ft}$ and prediction match 1.0 are sanity checks that validate the primary starting point, not a novelty claim.

## Secondary retrained-head protocol and auxiliary analyses

The historical DADS protocol is retained for secondary analysis. It uses pure-linear frozen-encoder readouts when an additive CTC-margin attribution is required, and a newly initialized readout after selection. It must be labeled as retrained-head selection, not direct pruning of the fine-tuned model.

The formal primary protocol instead uses the following fixed sequence:

1. Initialize the same Wav2Vec2-Large-960h encoder from its inherited processor, tokenizer, and CTC head; fine-tune on the fixed Fold 0 split.
2. Freeze the resulting model and extract $E_0$, $E_{ft}$, and $\Delta$ at the final layer for `train_utility`, dev, and test.
3. Verify numerical and behavioral identity of $E_0+\Delta$ and $E_{ft}$.
4. Compute $U_i=\mathbb{E}|\Delta_i\,\partial L_{CTC}/\partial\Delta_i|$ only on `train_utility`.
5. Freeze the Top-K mask and evaluate $E_0+M_K\odot\Delta$ with the original fine-tuned CTC head; do not train a new head.

Layer scans, magnitude/random comparisons, Drop-Best/Drop-Worst, and the blank/token split remain useful auxiliary or secondary analyses. They should not be described as necessary components of the formal primary claim unless their corresponding protocol is explicitly reported.

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


The formal Fold 0 mainline reports the frozen-head intervention directly: on test, no shift is 22.75%, full shift is 16.48%, and Top-512 Taylor utility is 17.51%, retaining 83.5% of the full-shift gain. These values are not a substitute for the retrained-head auxiliary comparisons; they answer the stricter pruning question.

## Documentation outcome

**Method used:** the formal UT8 Fold 0 run fixes one fine-tuned Wav2Vec2-Large-960h model and its original CTC head, computes Taylor utility from the held-out `train_utility` split, and evaluates direct shift masks without retraining.

**Effect:** the paper's primary claim is now functional concentration of the fine-tuning representation shift: Top-512 of 1024 coordinates retain 83.5% of the test adaptation gain. The earlier retrained-head 98.8% result and the pure-linear CTC-margin attribution remain secondary evidence for retained representational information, not the main pruning claim.

## Reconciliation of the 2026-08-29 literature scan

The literature scan still supports the distinction between parameter deltas, representation deltas, and learned activation factors. The formal result sharpens the paper boundary: unlike a static parameter task vector, the representation Delta is input-conditioned, $\Delta E(x)=E_{ft}(x)-E_{pt}(x)$, and the primary utility score is derived from the frozen fine-tuned CTC objective.

The name **Decision-Aligned Delta Selection (DADS)** can remain as the umbrella method name, but the operational primary criterion is **Taylor shift utility**. The main novelty sentence should therefore be:

> Existing Delta-SSL fusion treats a representation shift as a full embedding; this work measures the functional utility of input-conditioned Delta coordinates under the original fine-tuned CTC head and directly prunes the lower-utility coordinates.

The earlier CTC blank/token split and pure-linear margin attribution remain optional secondary analyses. They must not be used to imply that the formal frozen-head result depends on a newly trained attribution teacher.

## Mathematical audit outcome

For the primary protocol, define

$$L(x,y;\Delta)=L_{CTC}(g_{ft}(E_0(x)+\Delta(x)),y).$$

Removing coordinate $i$ changes $\Delta_i$ by $-\Delta_i$. The first-order loss change is therefore

$$\Delta L_i\approx-\Delta_i\,\frac{\partial L}{\partial\Delta_i},$$

which motivates the nonnegative score

$$U_i=\mathbb{E}\left|\Delta_i\,\frac{\partial L_{CTC}}{\partial\Delta_i}\right|.$$

This is a local first-order ranking criterion, not a claim that the score is a causal effect or that discarded coordinates are information-free. The direct frozen-head WER intervention is the empirical test beyond the Taylor approximation.

The historical pure-linear margin decomposition remains valid only for the explicitly labeled secondary retrained-head attribution protocol. It is not required for, and must not be conflated with, the primary Taylor-loss protocol.

## Insight-first narrative — frozen

The paper is not framed as a collection of layer selection, feature selection, and pruning modules. Its single scientific question is:

> **When fine-tuning changes an SSL representation, which changes actually matter for ASR?**

The central hypothesis is **shift magnitude is not shift utility**. DADS is the natural measurement and selection mechanism that follows from this question, rather than the headline insight by itself.

### Three empirical insights to establish

1. **Functional effect:** the complete fine-tuning shift improves the same frozen-head model, establishing that the shift itself carries task-relevant adaptation.
2. **Coordinate utility:** the CTC-loss Taylor score identifies which input-conditioned shift coordinates have the largest local effect; coordinate magnitude is retained only as an auxiliary baseline.
3. **Direct intervention:** masking the lower-utility half while freezing the original model and head preserves most of the full-shift gain.

Layer scans, magnitude-versus-utility correlations, and blank/token analyses can remain as auxiliary evidence, but they are not needed to define the primary method.

### Narrative flow

Mind the Shift establishes that a representation shift can carry useful adaptation information. We ask how much of that shift is necessary for the learned ASR behavior. The formal experiment answers this with three controlled steps:

1. compare no shift with the complete shift using the original fine-tuned CTC head;
2. rank shift coordinates using Taylor utility from the actual CTC loss; and
3. remove the lower-ranked half while keeping the model and head frozen.

The result is a direct intervention statement: the functional effect of fine-tuning is concentrated in a subset of input-conditioned representation-shift coordinates. The earlier retrained-head result is reported separately because it permits readout compensation.

### Title and terminology

The working manuscript title is **Beyond Shift Magnitude: What Makes a Fine-Tuning Shift Useful for ASR?** The method remains **Decision-Aligned Delta Selection (DADS)**. Use `Delta coordinate` and `decision utility`; reserve `feature` for citations that use that terminology.

### Required figure and tables

- **Figure 1:** the primary frozen-head test intervention: No Shift, 50% Taylor Utility, and Full Shift, with the 83.5% gain-retention annotation and the $E_0\rightarrow\Delta\rightarrow E_0+M\Delta$ method flow.
- **Figure 2 / supplementary:** layer magnitude versus fusion WER and coordinate magnitude versus utility, if the auxiliary scans are retained.
- **Table 1:** primary frozen-head Fold 0 result and identity gate.
- **Table 2:** primary versus secondary protocol distinction, including 83.5% versus the earlier 98.8%.
- **Supplementary tables:** source-compatible Mind-the-Shift reproduction, retrained-head Utility/Random/Magnitude comparisons, Drop-Best/Drop-Worst, and blank/token analyses.

**Effect.** This framing makes the experimental question, Taylor criterion, and frozen-head intervention answer one coherent insight while preserving earlier retrained-head experiments as secondary evidence.

## Post-formal story-consistency audit — 2026-09-02

**Verdict: the formal result now matches the paper's central claim.** The main evidence chain is:

1. Fine-tuning changes the representation: $\Delta=E_{ft}-E_{pt}$.
2. The complete shift improves the same frozen-head model from 22.75% to 16.48% test WER.
3. Taylor utility measures which shift coordinates affect the actual CTC objective.
4. Directly retaining only the top 512 of 1024 coordinates gives 17.51% test WER, preserving 83.5% of the adaptation gain.
5. The original CTC head is never retrained after selection.

The identity gate is a short sanity check: the maximum reconstruction error is $4.77\times10^{-7}$, reconstruction logit error is $7.63\times10^{-6}$, and direct-vs-cache prediction match is 1.0. These checks validate the intervention but are not presented as a contribution.

The earlier 98.8% retrained-head result remains preserved as a secondary experiment. It demonstrates that a selected representation can remain highly informative when the readout is allowed to adapt; it is not evidence for the stricter frozen-head pruning claim.

**Final central sentence:** Fine-tuning improves speech recognition through representation shifts, but these shifts are functionally concentrated: a simple Taylor utility identifies the coordinates that matter, allowing half of the shift dimensions to be removed while retaining 83.5% of the learned adaptation gain without retraining the downstream model.
