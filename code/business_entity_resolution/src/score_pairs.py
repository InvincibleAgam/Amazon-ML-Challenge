"""Scores more pairs with the already trained text models (text_model.py), in chunks, so that
memory stays small.  Used to widen the open zone of stage 4: the fold-f model scores the extra
training pairs of the other fold (out-of-fold, as in training) and all extra test pairs.

    python score_pairs.py --tag A --fold 0 --gpu 0 --data <dir> --out <dir of the tag>
        reads  <data>/extra_train.parquet, <data>/extra_test.parquet, <out>/text_model_fold<f>/
        writes <out>/extra_held_<f>.parquet, <out>/extra_test_<f>.parquet
    python score_pairs.py --merge --tag A --out <dir of the tag> --into <dir>
        -> <into>/text_scores_train_A.parquet, <into>/text_scores_test_A.parquet (open zone + extra pairs)
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

MAX_LEN = 128
CHUNK = 100_000
BATCH = 512


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def score(model, tok, a: list[str], b: list[str], device) -> np.ndarray:
    import torch
    out = np.zeros(len(a), dtype=np.float32)
    for i in range(0, len(a), CHUNK):
        enc = tok(a[i:i + CHUNK], b[i:i + CHUNK], truncation=True, max_length=MAX_LEN, return_token_type_ids=True)
        lens = np.array([len(x) for x in enc["input_ids"]])
        order = np.argsort(lens, kind="stable")                # similar lengths together: little padding
        with torch.no_grad():
            for j in range(0, len(order), BATCH):
                idx = order[j:j + BATCH]
                n = lens[idx].max()
                ids = np.full((len(idx), n), tok.pad_token_id, dtype=np.int64)
                seg = np.zeros((len(idx), n), dtype=np.int64)
                mask = np.zeros((len(idx), n), dtype=np.int64)
                for r, k in enumerate(idx):
                    m = lens[k]
                    ids[r, :m], seg[r, :m], mask[r, :m] = enc["input_ids"][k], enc["token_type_ids"][k], 1
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    z = model(input_ids=torch.from_numpy(ids).to(device), token_type_ids=torch.from_numpy(seg).to(device),
                              attention_mask=torch.from_numpy(mask).to(device)).logits.squeeze(-1)
                out[i + idx] = torch.sigmoid(z.float()).cpu().numpy()
        del enc
        log(f"  {min(i + CHUNK, len(a))} of {len(a)} pairs scored")
    return out


def run(args):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    path = os.path.join(args.out, f"text_model_fold{args.fold}")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    if device.type == "cuda":
        model = model.half()
    tr = pd.read_parquet(os.path.join(args.data, "extra_train.parquet"))
    tr = tr[tr.fold != args.fold]                             # out-of-fold, as in training
    te = pd.read_parquet(os.path.join(args.data, "extra_test.parquet"))
    if args.limit:
        tr, te = tr.iloc[:args.limit], te.iloc[:args.limit]
    log(f"{args.tag} fold {args.fold} on {device}: {len(tr)} training pairs (other fold), {len(te)} test pairs")
    p = score(model, tok, tr.text_a.tolist(), tr.text_b.tolist(), device)
    tr[["q_gi", "s1_gi"]].assign(pt=p).to_parquet(os.path.join(args.out, f"extra_held_{args.fold}.parquet"))
    p = score(model, tok, te.text_a.tolist(), te.text_b.tolist(), device)
    te[["q_gi", "s1_gi"]].assign(pt=p).to_parquet(os.path.join(args.out, f"extra_test_{args.fold}.parquet"))
    log(f"{args.tag} fold {args.fold} done")


def merge(args):
    j = lambda n: os.path.join(args.out, n)                   # noqa: E731
    held = pd.concat([pd.read_parquet(j(f"{k}_{f}.parquet")) for k in ("part_held", "extra_held") for f in (0, 1)])
    held = held.drop_duplicates(["q_gi", "s1_gi"])
    held.to_parquet(os.path.join(args.into, f"text_scores_train_{args.tag}.parquet"))
    per = [pd.concat([pd.read_parquet(j(f"part_test_{f}.parquet")), pd.read_parquet(j(f"extra_test_{f}.parquet"))])
             .drop_duplicates(["q_gi", "s1_gi"]) for f in (0, 1)]
    t = per[0].merge(per[1], on=["q_gi", "s1_gi"], suffixes=("_0", "_1"))
    t["pt"] = (t.pt_0 + t.pt_1) / 2
    t[["q_gi", "s1_gi", "pt"]].to_parquet(os.path.join(args.into, f"text_scores_test_{args.tag}.parquet"))
    log(f"merged {args.tag}: {len(held)} training pairs (out-of-fold), {len(t)} test pairs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default=".")
    ap.add_argument("--into", default=".")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge(a) if a.merge else run(a)
