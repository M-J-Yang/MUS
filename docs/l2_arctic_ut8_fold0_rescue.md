# Fold 0 W2V2-Large rescue run

This run addresses the previous 16.48% direct test WER. It keeps the inherited
`facebook/wav2vec2-large-960h` processor and CTC head, but aligns the training
loss and precision with the public robust-atc-asr recipe: mean-reduced CTC,
`zero_infinity`, one-epoch FP32 head warm-up with a full linear warm-up, then
joint bf16 fine-tuning at `1e-5`, cosine decay, clip 1, max 40 epochs, and
patience 5.

The old run accidentally retained the pretrained config's `ctc_loss_reduction=sum`
and `ctc_zero_infinity=false`, used fp16 for both phases, and selected checkpoints
with padded-batch WER. This launcher now evaluates with batch 1 for checkpoint
selection, matching the direct greedy evaluator.

On the currently available RTX 3090s, a real longest-utterance
forward/backward/AdamW probe passed at batch 11 with bf16 and no gradient
checkpointing (22.16 GiB peak reserved), but the phase-transition smoke showed
that batch 11 can OOM by about 100 MiB. Batch 12 also OOMed, so the rescue run
uses the safe per-device batch 10, accumulation 1, and keeps the official head warm-up
effective batch at 16 via per-device batch 4, accumulation 2, and two DDP ranks.

Run:

```bash
CUDA_VISIBLE_DEVICES=0,3 \
NPROC_PER_NODE=2 \
PRETRAINED_PATH=checkpoints/wav2vec2_large_960h_pretrained \
OUTPUT_DIR=artifacts/runs/l2_arctic_ut8/fold0/w2v2_large_960h_rescue_ddp_g0g3_b10 \
JOINT_BATCH_SIZE=10 \
JOINT_GRAD_ACCUM=1 \
HEAD_BATCH_SIZE=4 \
HEAD_GRAD_ACCUM=2 \
DATALOADER_NUM_WORKERS=8 \
GRADIENT_CHECKPOINTING=false \
MASTER_PORT=29541 \
bash scripts/run_l2_arctic_fold0_w2v2_960h.sh
```

The completed DDP run used 16,386 train examples, 1,827 dev examples, and 667
test examples. It stopped after 14 joint epochs and selected
`joint/checkpoint-7380` by dev WER. The Trainer summary reports 12.85% dev WER
and 16.59% test WER; an independent single-GPU batch-1 greedy evaluator gives
12.79% dev and 16.71% test. The same independent evaluator on the old baseline
gives 13.25% dev and 16.48% test, so the rescue clearly improves dev WER but does
not improve this held-out test split. The independent numbers are recorded in
`artifacts/runs/l2_arctic_ut8/fold0/w2v2_large_960h_rescue_ddp_g0g3_b10/independent_eval.json`.

The existing `w2v2_large_960h` artifact is retained unchanged.
