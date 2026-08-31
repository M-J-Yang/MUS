#!/usr/bin/env python3
"""Resample raw Arctic WAV manifests to the 16 kHz SSL input contract."""
from __future__ import annotations
import argparse, json, math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

def output_path(audio_path: str, raw_root: Path, output_root: Path, project_root: Path) -> Path:
    p=Path(audio_path)
    if not p.is_absolute(): p=project_root / p
    try: rel=p.resolve().relative_to(raw_root.resolve())
    except ValueError: raise ValueError(f"audio path is outside raw root: {audio_path}")
    return output_root / rel

def convert_one(src: Path, dst: Path, target_sr: int) -> tuple[str,int,int]:
    if dst.is_file():
        try:
            with sf.SoundFile(dst) as f:
                if f.samplerate == target_sr and f.channels == 1: return (str(src), f.frames, 0)
        except RuntimeError: pass
    audio, sr = sf.read(str(src), dtype="float32", always_2d=True)
    if audio.shape[1] > 1: audio=audio.mean(axis=1)
    else: audio=audio[:,0]
    if sr != target_sr:
        g=math.gcd(int(sr), int(target_sr)); audio=resample_poly(audio, target_sr//g, sr//g)
    audio=np.asarray(audio, dtype=np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), audio, target_sr, subtype="PCM_16")
    return (str(src), int(audio.shape[0]), 1)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir',type=Path,default=Path('data/processed/arctic'))
    ap.add_argument('--raw-root',type=Path,default=Path('data/raw/l2_arctic'))
    ap.add_argument('--output-root',type=Path,default=Path('data/processed/arctic/audio16k'))
    ap.add_argument('--project-root',type=Path,default=Path('.'))
    ap.add_argument('--workers',type=int,default=8)
    args=ap.parse_args(); project=args.project_root.resolve(); raw=args.raw_root.resolve(); out=args.output_root.resolve(); inp=args.input_dir.resolve()
    manifests=[('l2_manifest.jsonl','l2_manifest_16k.jsonl'),('paired_manifest.jsonl','paired_manifest_16k.jsonl'),('suitcase_manifest.jsonl','suitcase_manifest_16k.jsonl')]
    source_rows={}; jobs={}
    for name,_ in manifests:
        p=inp/name
        if not p.is_file(): continue
        rows=read_jsonl(p); source_rows[name]=rows
        paths=[]
        for row in rows:
            key=row.get('audio_path') or row.get('l2_audio_path')
            if key: paths.append(key)
            if row.get('cmu_audio_path'): continue
        for key in paths:
            src=project/ key if not Path(key).is_absolute() else Path(key); dst=output_path(key,raw,out,project); jobs[(str(src),str(dst))]=(src,dst)
    print('unique_audio_jobs',len(jobs),flush=True)
    converted=0
    with ThreadPoolExecutor(max_workers=max(1,args.workers)) as ex:
        fs=[ex.submit(convert_one,src,dst,16000) for src,dst in jobs.values()]
        for fut in as_completed(fs):
            fut.result(); converted+=1
            if converted % 1000 == 0: print('processed',converted,flush=True)
    def mapped(key): return str(output_path(key,raw,out,project).resolve().relative_to(project)) if not Path(key).is_absolute() else str(output_path(key,raw,out,project))
    for source_name,target_name in manifests:
        rows=source_rows.get(source_name)
        if rows is None: continue
        out_rows=[]
        for row in rows:
            row=dict(row)
            if row.get('audio_path'): row['audio_path_44k_path']=row['audio_path']; row['audio_path']=mapped(row['audio_path']); row['sample_rate']=16000
            if row.get('l2_audio_path'): row['l2_audio_path_44k']=row['l2_audio_path']; row['l2_audio_path']=mapped(row['l2_audio_path']); row['l2_sample_rate']=16000
            out_rows.append(row)
        write_jsonl(inp/target_name,out_rows)
    print('converted',converted,'files to',out)
if __name__=='__main__': main()
