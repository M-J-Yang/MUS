# DADS paper reading library

**Created:** 2026-08-29  
**Purpose:** a reproducible, local reading library for refining Decision-Aligned Delta Selection (DADS). This library is reference material only; it does not alter the MUS Python environment or training artifacts.

## Source policy

Every downloaded PDF comes from an official archival, venue, or author-hosted page. The source URL and SHA-256 below identify the exact local artifact. A publisher-restricted HSA PDF was not bypassed: the official code and the publisher's public abstract/method description are retained instead.

## Core comparison set

| Work | Local artifact | Official source | Status / first-pass relevance |
|---|---|---|---|
| Mind the Shift (2026) | `papers/mind_the_shift_2026.pdf` | https://arxiv.org/abs/2601.20142 | Downloaded. Direct predecessor: representation Delta fusion, but full Delta is concatenated without coordinate utility selection. |
| Chiu et al., Layer Selection (2024) | `papers/chiu_layer_selection_2024.pdf` | https://www.isca-archive.org/interspeech_2024/chiu24_interspeech.html | Downloaded. Establishes that layer/source-coordinate selection is not a DADS novelty. |
| Selective Attention Merging (2025) | `papers/selective_attention_merging_2025.pdf` | https://www.seas.ucla.edu/spapl/paper/Balaji_ICASSP_2025.pdf | Downloaded from the authors' public UCLA page. Selects parameter-space attention updates, not representation Delta coordinates. |
| HSA Merge (2026/2027 issue) | no local publisher PDF | https://www.sciencedirect.com/science/article/pii/S0885230826000756 | Publisher PDF request returned HTTP 403. Official code is local at `../repos/hsa_merge`; its public abstract/method page and README establish magnitude-based Q/K/V-head selection. |
| AudioSAE (2026) | `papers/audiosae_2026.pdf` | https://aclanthology.org/2026.eacl-long.149/ | Downloaded. SAE learned features are an interpretability/steering object, not native fine-tuning Delta coordinates. |
| SADI (2025) | `papers/sadi_2025.pdf` | https://proceedings.iclr.cc/paper_files/paper/2025/file/c4d26a95fd83f8e590f81c54ae670b5d-Paper-Conference.pdf | Downloaded. Contrastive activation intervention; useful conceptual comparator, but not CTC supervision. |
| AUSteer (2026) | `papers/austeer_2026.pdf` | https://proceedings.iclr.cc/paper_files/paper/2026/file/423d0909791493b7c10916fd328c2913-Paper-Conference.pdf | Downloaded. Closest conceptual neighbor: dimension heterogeneity and consistent direction, but in LLM steering rather than Delta selection / CTC margins. |
| DAREx (2025) | `papers/darex_2025.pdf` | https://proceedings.iclr.cc/paper_files/paper/2025/file/d0074bea472f8b9b839fa2d50ce67595-Paper-Conference.pdf | Downloaded. Importance pruning of parameter deltas; motivates Random/Magnitude/Utility comparisons but is a different object. |

## Supporting background

| Work | Local artifact | Official source | Why retained |
|---|---|---|---|
| TIES-Merging (2023) | `papers/ties_merging_2023.pdf` | https://proceedings.neurips.cc/paper_files/paper/2023/file/1644c9af28ab7916874f6fd6228a9bcf-Paper-Conference.pdf | Parameter-task-vector background: trim small changes and resolve sign conflicts. It limits generic 'selective Delta' claims. |

## Exact downloaded PDF checksums

| File | SHA-256 |
|---|---|
| `audiosae_2026.pdf` | `ef40aceacf7de7de5d3d6db14c5afd6e08fdc17a373b96eda2d331a26afc90de` |
| `austeer_2026.pdf` | `fdab7828eb2e25a7a74168a6b05e5cf9216610aec5ec562911800ba2a8c5a4dc` |
| `chiu_layer_selection_2024.pdf` | `e2b83a2c44a48fedea315fd93287472dacf751c4966d9b279381b870dab8b8fc` |
| `darex_2025.pdf` | `9c3c833436dec2f98311451253301075dc4ee9f02cf43797583de2ac4cf80b27` |
| `mind_the_shift_2026.pdf` | `83558655cf912338996d48b513d2190877e52e21bde085cbf5f244f9b61f37a6` |
| `sadi_2025.pdf` | `48be2b4e8d0edc7831a8f3e50431dbeb3b4f48589eb28aff88a4a62dea4eff41` |
| `selective_attention_merging_2025.pdf` | `5d12333bc04cb6dc041af1b2dafdebcc676acc41be60a508be29d7a65d46e3fc` |
| `ties_merging_2023.pdf` | `c7ee1cfce4625f8a3008a7d1c62f08ac4d8d0816f3e699aab7899eb9b78469aa` |

## Official implementation sources

| Source | Local path | Retrieval form | What it confirms |
|---|---|---|---|
| Mind the Shift | `../repos/Delta-Embedding-Fusion` | git clone, commit `ac87fd08d073f585bfb3eabe967c32515fd8e55a` | Source-compatible fusion is LayerNorm -> Dropout -> Linear, so it cannot be used as an exact coordinate-additive attribution head. |
| SA Merge | `../repos/sa_merging` | official GitHub source archive, main branch | SA Merge implementation and MyST split materials. |
| HSA Merge | `../repos/hsa_merge` | official GitHub source archive, main branch | HSA score selects Q/K/V parameter heads by task-vector magnitude; it does not score downstream CTC target-vs-competitor margins. |

## First-pass technical reading conclusions

1. **Mind the Shift:** use it for the exact WavLM + final W2V2 Delta reproduction target. Its contribution is representation-space Delta fusion, not internal Delta-coordinate selection.
2. **Chiu:** cite it directly where layer scanning appears; it already performs dimension-wise *source-layer* selection. DADS must say it ranks coordinates within an already fixed Delta representation.
3. **SA/HSA and DAREx/TIES:** their selection objects are parameter deltas. HSA's released code confirms a magnitude saliency proxy over attention heads, which is deliberately different from DADS's output-decision criterion.
4. **AudioSAE:** avoid describing a native hidden coordinate as an interpretable semantic feature. It is safer to use 'Delta coordinate' or 'Delta dimension'.
5. **SADI/AUSteer:** they justify comparing stable helpful/harmful directions, but DADS's distinctive evidence is supervised CTC Viterbi alignment and the strongest actual competitor, including blank.

## Frozen consequence for DADS

The attribution score remains

$$q_{t,i}=\delta_{t,i}(W^\Delta_{a_t,i}-W^\Delta_{c_t,i}), \qquad U_i=\operatorname{mean}_t\operatorname{sign}(q_{t,i}).$$

For this equality to be an exact coordinate contribution, the DADS teacher and all selection ablations use a pure-linear CTC readout. The Mind-the-Shift reproduction remains a separately reported source-compatible LayerNorm -> Dropout -> Linear baseline.

## Reading sequence now in progress

1. **Complete:** Read Mind the Shift methods/results against the local reproduction code and record every reproducibility choice.
2. Extract the exact selection objectives and validation protocols from Chiu, SA/HSA, DAREx, SADI, and AUSteer.
3. Convert the comparison matrix into a manuscript Related Work section with verified citations only.
4. Add a CTC-specific blank-versus-token utility ablation after the baseline has been reproduced.

## Execution record

**Method used.** Located official venue, author, and publisher pages; downloaded the publicly accessible PDFs; validated each download with the system `file` utility; extracted opening sections with `pdftotext`; and inspected official SA/HSA implementation READMEs and merge code.

**Observed result.** Eight PDFs are locally available (the eight-work core is covered by seven PDFs plus the HSA source/method record; SA and HSA are separate works). TIES is retained as an additional background PDF. The HSA publisher PDF was unavailable through normal public access (HTTP 403), so no access control was bypassed.

**Effect.** DADS claim boundaries are now backed by locally available primary sources. The next technical work can cite checked formulas and experimental protocols rather than rely on abstracts or memory.

## Step P2 — Mind the Shift close-reading record — COMPLETE

**Method.** Read the local paper's feature-extraction, fusion, dataset, and result sections against the already archived author code. The pass used `pdftotext -layout` so table values and formulas could be checked without relying on a secondary summary.

**Observed result.** The paper filters MyST by Whisper-large-v2 WER above 50%, fewer than three words, and duration above 30 seconds; it reports 133 h train, 21 h dev, and 25 h test after filtering. It evaluates Full/10 h/5 h/1 h, uses 16-kHz audio, 20-ms effective stride, 24 layers, 1024 dimensions, and removes the original upper CTC layers before freezing SSL features and fitting a new head.

The exact Concat reference numbers to reproduce for WavLM plus Delta-W2V2 are:

| MyST condition | Paper WER (%) |
|---|---:|
| Full | 9.64 |
| 10 h | 11.61 |
| 5 h | 12.88 |
| 1 h | 21.81 |

The paper explicitly chooses final-layer Delta based on prior upper-layer-shift evidence and final-layer CTC optimization. This confirms that our layer scan is a post-baseline diagnostic, not a claim of reproducing a hidden layer choice from the paper.

**Effect.** These four WERs are now the fixed reproduction targets for Step 1. The implementation must first match the paper's filtering/split protocol and source-compatible readout before any DADS pure-linear attribution experiment is reported.
