# Source alignment: Mind the Shift baseline

This note records the public implementation facts that define the Step 1
baseline. It prevents a later USDE experiment from accidentally comparing to a
different fusion system.

| Component | Author public source | Step 1 implementation |
| --- | --- | --- |
| Data inputs | Kaldi `wav.scp` and `text` | validated JSONL adapter; can be generated from the same split files |
| Final state | `hidden_states[-1]` | semantic transformer layer 24, `hidden_states[24]` |
| Delta | fine-tuned minus pretrained | identical |
| Extraction waveform | no padding/truncation | identical; 16-kHz mono asserted |
| Fusion head | LayerNorm, dropout 0.1, linear | identical default |
| Fusion optimizer | AdamW, 10 epochs, 1e-4 LR, 0.01 WD, batch 16, seed 42 | identical defaults |
| Unequal stream length at eval | truncate to min length | reject as a data/configuration error |
| Test split | code accepts eval path | project blocks test from selection; test remains final-only |

The only functional hardening is mismatched-frame handling. It never changes a
valid source-compatible run because WavLM-Large and W2V2-Large have the same
20-ms effective stride. If a mismatch appears, its utterance ID is a reproducible
diagnostic rather than silently changing the data.

For the full-data paper comparison, the published WavLM + Delta W2V2 fusion
result to target is **9.64 WER**, subject to matching the licensed MyST split
