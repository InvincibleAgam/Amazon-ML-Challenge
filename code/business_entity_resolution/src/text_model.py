"""Text model for the open pairs: a small pretrained transformer (MiniLM, 33 M parameters,
MIT licence) fine-tuned as a cross-encoder on the raw text of the two records.  It reads what
the hand-made features only summarise: spelling, punctuation, word order, how a number is
written.  Its out-of-fold probability becomes one more feature of stage 4.

Trained on a GPU (about one hour on two T4).  One process per fold:

    python text_model.py --fold 0 --gpu 0 &      # fits on fold 0, scores fold 1 and test
    python text_model.py --fold 1 --gpu 1        # fits on fold 1, scores fold 0 and test
    python text_model.py --merge --tag _A        # -> text_scores_train_A.parquet, text_scores_test_A.parquet

Input : pairs_train.parquet, pairs_test.parquet (export_pairs.py) in --data
Output: in --out.  The folds are those of every other stage (parity of the true Source-1 id),
so the training scores are out-of-fold.
"""
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import pandas as pd

MODEL = "microsoft/MiniLM-L12-H384-uncased"     # MIT licence, 33 M parameters; lower-cases, strips accents
MAX_LEN = 128
EPOCHS = 3
BATCH = 128
LR = 5e-5
WARMUP = 0.05
SEED = 13


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def encode(tok, a: list[str], b: list[str]) -> list[list[int]]:
    """Token ids of "[CLS] record [SEP] Source-1 entity [SEP]", truncated to MAX_LEN."""
    out = []
    for i in range(0, len(a), 50_000):
        enc = tok(a[i:i + 50_000], b[i:i + 50_000], truncation=True, max_length=MAX_LEN,
                  return_token_type_ids=True, return_attention_mask=False)
        # small integer arrays instead of Python lists: a tenth of the memory (the machine is shared)
        out += [(np.asarray(i, dtype=np.int32), np.asarray(t, dtype=np.int8))
                for i, t in zip(enc["input_ids"], enc["token_type_ids"])]
        del enc
    return out


def batches(enc, order: np.ndarray, size: int, pad: int):
    """Padded batches (ids, segment ids, mask, positions in `enc`) in the given order."""
    import torch
    for i in range(0, len(order), size):
        idx = order[i:i + size]
        n = max(len(enc[j][0]) for j in idx)
        ids = np.full((len(idx), n), pad, dtype=np.int64)
        seg = np.zeros((len(idx), n), dtype=np.int64)
        mask = np.zeros((len(idx), n), dtype=np.int64)
        for r, j in enumerate(idx):
            k = len(enc[j][0])
            ids[r, :k], seg[r, :k], mask[r, :k] = enc[j][0], enc[j][1], 1
        yield torch.from_numpy(ids), torch.from_numpy(seg), torch.from_numpy(mask), idx


def by_length(enc, chunk: int, rng: np.random.Generator | None) -> np.ndarray:
    """Order with similar lengths next to each other (little padding); shuffled if rng is given."""
    n = len(enc)
    base = rng.permutation(n) if rng is not None else np.arange(n)
    parts = []
    for i in range(0, n, chunk):
        part = base[i:i + chunk]
        parts.append(part[np.argsort([len(enc[j][0]) for j in part], kind="stable")])
    if rng is not None:                       # shuffle whole batches, keep lengths together
        groups = [np.concatenate(parts)[i:i + BATCH] for i in range(0, n, BATCH)]
        groups = [groups[i] for i in rng.permutation(len(groups))]
        return np.concatenate(groups)
    return np.concatenate(parts)


def predict(model, enc, device, pad: int, amp: bool) -> np.ndarray:
    import torch
    model.eval()
    out = np.zeros(len(enc), dtype=np.float32)
    with torch.no_grad():
        for ids, seg, mask, idx in batches(enc, by_length(enc, 4 * BATCH * 64, None), 4 * BATCH, pad):
            with torch.autocast(device_type=device.type, enabled=amp):
                z = model(input_ids=ids.to(device), token_type_ids=seg.to(device),
                          attention_mask=mask.to(device)).logits.squeeze(-1)
            out[idx] = torch.sigmoid(z.float()).cpu().numpy()
    model.train()
    return out


def run(args):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(args.seed + args.fold)
    rng = np.random.default_rng(args.seed + args.fold)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else
                          "mps" if args.mps and torch.backends.mps.is_available() else "cpu")
    amp = device.type == "cuda"
    train = pd.read_parquet(os.path.join(args.data, "pairs_train.parquet"))
    test = pd.read_parquet(os.path.join(args.data, "pairs_test.parquet"))
    if args.limit:
        train, test = train.iloc[:args.limit], test.iloc[:args.limit]
    fit, held = train[train.fold == args.fold], train[train.fold != args.fold]
    log(f"fold {args.fold} on {device}: fit on {len(fit)} pairs ({fit.y.mean():.3f} true), "
        f"score {len(held)} held-out and {len(test)} test pairs")

    tok = AutoTokenizer.from_pretrained(args.model)
    # .float(): some checkpoints (DeBERTa-v3) load in half precision, which mixed-precision training cannot unscale
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).float().to(device)
    pad = tok.pad_token_id
    e_fit = encode(tok, fit.text_a.tolist(), fit.text_b.tolist())
    e_held = encode(tok, held.text_a.tolist(), held.text_b.tolist())
    e_test = encode(tok, test.text_a.tolist(), test.text_b.tolist())
    y = torch.tensor(fit.y.to_numpy(), dtype=torch.float32)
    log(f"tokens per pair: median {int(np.median([len(e[0]) for e in e_fit]))}, "
        f"share cut at {MAX_LEN}: {np.mean([len(e[0]) >= MAX_LEN for e in e_fit]):.3f}")

    steps = args.epochs * math.ceil(len(e_fit) / BATCH)
    decay = [p for n, p in model.named_parameters() if not any(k in n for k in ("bias", "LayerNorm"))]
    rest = [p for n, p in model.named_parameters() if any(k in n for k in ("bias", "LayerNorm"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": rest, "weight_decay": 0.0}], lr=args.lr)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    step, t0, run_loss = 0, time.time(), 0.0
    for epoch in range(args.epochs):
        for ids, seg, mask, idx in batches(e_fit, by_length(e_fit, BATCH * 64, rng), BATCH, pad):
            with torch.autocast(device_type=device.type, enabled=amp):
                z = model(input_ids=ids.to(device), token_type_ids=seg.to(device),
                          attention_mask=mask.to(device)).logits.squeeze(-1)
            loss = loss_fn(z.float(), y[idx].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss is {loss.item()} at step {step}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run_loss += loss.item()
            if step % 200 == 0:
                rate = step * BATCH / (time.time() - t0)
                log(f"fold {args.fold} epoch {epoch + 1} step {step}/{steps}  loss {run_loss / 200:.4f}  "
                    f"{rate:.0f} pairs/s  left {(steps - step) * BATCH / rate / 60:.0f} min")
                run_loss = 0.0
        # held-out scores after every epoch: the log shows whether another epoch still helps
        p_held = predict(model, e_held, device, pad, amp)
        yh = held.y.to_numpy()
        ll = -np.mean(yh * np.log(np.clip(p_held, 1e-6, 1)) + (1 - yh) * np.log(np.clip(1 - p_held, 1e-6, 1)))
        p3 = np.clip(held.p3.to_numpy(), 1e-6, 1 - 1e-6)
        ll3 = -np.mean(yh * np.log(p3) + (1 - yh) * np.log(1 - p3))
        log(f"fold {args.fold} epoch {epoch + 1}: held-out log loss {ll:.4f}  (pipeline so far: {ll3:.4f})")
        held[["q_gi", "s1_gi"]].assign(pt=p_held).to_parquet(os.path.join(args.out, f"part_held_{args.fold}.parquet"))
    p_test = predict(model, e_test, device, pad, amp)
    test[["q_gi", "s1_gi"]].assign(pt=p_test).to_parquet(os.path.join(args.out, f"part_test_{args.fold}.parquet"))
    log(f"fold {args.fold}: test scores written")
    model.half().save_pretrained(os.path.join(args.out, f"text_model_fold{args.fold}"))
    tok.save_pretrained(os.path.join(args.out, f"text_model_fold{args.fold}"))
    log(f"fold {args.fold} done")


def merge(args):
    held = pd.concat([pd.read_parquet(os.path.join(args.out, f"part_held_{f}.parquet")) for f in (0, 1)])
    held.to_parquet(os.path.join(args.out, f"text_scores_train{args.tag}.parquet"))
    t0, t1 = (pd.read_parquet(os.path.join(args.out, f"part_test_{f}.parquet")) for f in (0, 1))
    t = t0.merge(t1, on=["q_gi", "s1_gi"], suffixes=("_0", "_1"))
    t["pt"] = (t.pt_0 + t.pt_1) / 2
    t[["q_gi", "s1_gi", "pt"]].to_parquet(os.path.join(args.out, f"text_scores_test{args.tag}.parquet"))
    log(f"merged: {len(held)} training pairs (out-of-fold), {len(t)} test pairs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default=".")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--limit", type=int, default=0, help="first N pairs only (smoke test)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--lr", type=float, default=LR, help="peak learning rate (2e-5 for large models)")
    ap.add_argument("--tag", default="", help="suffix of the merged score files (one per model)")
    ap.add_argument("--mps", action="store_true", help="use the Apple GPU when there is no CUDA")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    merge(a) if a.merge else run(a)
