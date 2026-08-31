# Arctic 数据下载与转换记录（Step 1 数据准备）

更新日期：2026-08-30（Asia/Shanghai）

## 目标与范围

本步骤准备论文配对实验需要的两个语料：

- **L2-ARCTIC**：用户提供的 Google Drive 文件 ID `1ciCw_ttbw7a9r7d5DZzTJwoZq5rQB3TA`，目标是完整原始语料。
- **CMU ARCTIC**：使用 FestVox 官方 archive，先下载 native/reference speakers `BDL/SLT/CLB/RMS`。

原始 L2 页面和许可信息见 [Texas A&M L2-ARCTIC 页面](https://psi.engr.tamu.edu/l2-arctic-corpus/)；CMU archive 地址与 checksum 采用 `torchaudio.datasets.CMUARCTIC` 内置的官方 FestVox 配置。MUS 环境单独使用，未修改 `py311`。

## Step 1-A：环境和网络路径确认 — COMPLETE

**方法。** 在 `/home/zbzb/.conda/envs/MUS` 中检查 `torchaudio` 和 `gdown`，并分别用代理和不经代理的路径探测官方端点。

**结果。** 两个 Python 模块均已存在，不需要额外安装。FestVox 的官方 HTTP endpoint 在系统代理下返回 502；使用 `curl --noproxy '*'` 直连后返回 200，并支持 HTTP Range。因此下载脚本固定使用官方 HTTP + no-proxy + 分段下载。Google Drive 的 `drive.google.com` 和 `drive.usercontent.google.com` 在 IPv4 直连（25 秒超时）和现有代理路径均未建立连接；没有绕过 Google Drive 权限或访问控制。

## Step 1-B：CMU ARCTIC 下载与校验 — COMPLETE

**命令。**

```bash
bash scripts/download_cmu_arctic.sh data/raw/cmu_arctic 4
```

脚本对每个 archive 分成 4 个 Range 段，检查段长度，拼接后检查完整 SHA256，再解压并检查 `etc/txt.done.data`。官方 archive 及校验结果：

| speaker | archive | bytes | SHA256 | transcript 行数 |
|---|---:|---:|---|---:|
| BDL | `cmu_us_bdl_arctic.tar.bz2` | 73,590,286 | `26b91aaf48b2799b2956792b4632c2f926cd0542f402b5452d5adecb60942904` | 1,131 |
| SLT | `cmu_us_slt_arctic.tar.bz2` | 81,326,064 | `7c173297916acf3cc7fcab2713be4c60b27312316765a90934651d367226b4ea` | 1,132 |
| CLB | `cmu_us_clb_arctic.tar.bz2` | 90,892,292 | `3f16dc3f3b97955ea22623efb33b444341013fc660677b2e170efdcc959fa7c6` | 1,132 |
| RMS | `cmu_us_rms_arctic.tar.bz2` | 92,541,266 | `c6dc11235629c58441c071a7ba8a2d067903dfefbaabc4056d87da35b72ecda4` | 1,132 |

解压目录为 `data/raw/cmu_arctic/ARCTIC/cmu_us_<speaker>_arctic/`。音频审计显示 4,527 个 WAV 全部为 16 kHz；BDL 目录有 1,132 个 WAV，但 `arctic_a0507.wav` 在官方 `txt.done.data` 中没有对应文本，故后续 manifest 只保留 1,131 个有文本的 BDL 条目并记录该异常。

## Step 1-C：CMU/L2 清单与配对转换器 — COMPLETE

脚本 [prepare_arctic_pairs.py](../scripts/prepare_arctic_pairs.py) 做以下工作：

1. 解析每个 CMU speaker 的 `etc/txt.done.data`；
2. 由 WAV 文件名提取 `arctic_a/b####` prompt ID；
3. 写出 `data/processed/arctic/cmu_manifest.jsonl`；
4. 以 prompt ID 将每个 L2 WAV 与每个指定 CMU native speaker 配对，并写出 `l2_manifest.jsonl`、`paired_manifest.jsonl`；同时为 suitcase 子集写出 `suitcase_manifest.jsonl`；
5. 写出 `summary.json`，包含计数、采样率和缺失 transcript 记录。

完整 Arctic 审计与转换命令：

```bash
/home/zbzb/.conda/envs/MUS/bin/python scripts/prepare_arctic_pairs.py \
  --cmu-root data/raw/cmu_arctic \
  --l2-root data/raw/l2_arctic \
  --output-dir data/processed/arctic
```

当前结果保存在 `data/processed/arctic/summary.json`，状态为 `paired`：26,867 条 scripted L2 音频、107,444 条 L2×CMU 配对（4 个 native speaker），以及 22 条 suitcase 样本。L2 原始转写保留在 `l2_manifest.jsonl`，配对 manifest 的 `transcript` 使用 CMU canonical prompt，另保留 `l2_transcript` 字段。脚本不会重采样或改写原始音频。已用一个临时 symlink WAV 做端到端 smoke test：1 条 L2 输入正确生成 4 条配对记录；测试文件位于 `/tmp`，未写入真实语料目录。

## Step 1-D：L2-ARCTIC 下载与校验 — COMPLETE

用户提供的 Google Drive 在本环境不可达。为完成数据准备，使用公开、非 gated 的 [chikingsley/l2-arctic-release-v5.0](https://huggingface.co/datasets/chikingsley/l2-arctic-release-v5.0) 镜像；其 README 声明这是 L2-ARCTIC v5.0 原始 speaker-level ZIP 发布版，包含 24 个 speaker archive、suitcase archive、PROMPTS、原始 README 和 CC BY-NC 4.0 LICENSE。下载的 25 个 ZIP 总计 7,523,333,755 bytes，逐项与 Hub 的文件大小和 SHA256 元数据匹配；明细见 `data/raw/l2_arctic_release_v5.0/archive_checksums.json`。

ZIP 已解压到 `data/raw/l2_arctic/`，保留原始 44.1 kHz WAV、transcript、TextGrid 和 annotation。24 个 scripted speaker 合计 26,867 条 WAV/transcript，suitcase 子集 22 条 WAV/transcript；目录结构与 L2-ARCTIC v5.0 发布布局一致。原始 Drive ID 仍记录在本文开头，镜像只作为当前网络不可达时的可读数据源，未绕过任何访问控制。

完整配对结果：`data/processed/arctic/paired_manifest.jsonl`（107,444 行），`l2_manifest.jsonl`（26,867 行），`suitcase_manifest.jsonl`（22 行）。CMU BDL 的官方 archive 有一个 WAV（`arctic_a0507.wav`）没有 transcript，因此 CMU manifest 记录并跳过该条；这导致配对数比 `26,867 × 4` 少 24 条，属于可解释的 prompt 缺失。

## Step 1-E：SSL 输入格式转换 — COMPLETE

L2-ARCTIC 原始 WAV 按发布版保留为 44.1 kHz；由于 WavLM/Wav2Vec2 输入契约为 16 kHz，使用 [resample_arctic.py](../scripts/resample_arctic.py) 另行生成处理副本，不覆盖原始文件：

```bash
/home/zbzb/.conda/envs/MUS/bin/python scripts/resample_arctic.py \
  --input-dir data/processed/arctic \
  --raw-root data/raw/l2_arctic \
  --output-root data/processed/arctic/audio16k \
  --workers 8
```

方法使用 `scipy.signal.resample_poly` 做 44.1 kHz→16 kHz 重采样，输出 PCM-16、单声道 WAV。共生成 26,889 个 16 kHz WAV（3,170,209,650 bytes），逐文件头审计无异常。对应产物为：

- `data/processed/arctic/l2_manifest_16k.jsonl`：26,867 条；
- `data/processed/arctic/paired_manifest_16k.jsonl`：107,444 条；
- `data/processed/arctic/suitcase_manifest_16k.jsonl`：22 条。

16 kHz manifest 的所有音频路径均存在且唯一 ID 数与行数一致，可直接提供给后续 SSL 特征提取；原始 44.1 kHz 路径在字段 `audio_path_44k_path`/`l2_audio_path_44k` 中保留。
