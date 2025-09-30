#!/usr/bin/env python3
import os, json, argparse, hashlib, glob
from typing import List, Dict, Iterable
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report, precision_recall_fscore_support

FOCUS_LABELS = ['math','graphs','strings','number theory','trees','geometry','games','probabilities']

# --------------------------- IO helpers ---------------------------

def load_thresholds(model_dir: str):
    path = os.path.join(model_dir, "thresholds.json")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj["labels"], obj["thresholds"]

def save_jsonl(rows: List[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

# --------------------------- preprocessing ---------------------------

def to_text_full(desc: str, code: str) -> str:
    parts = []
    if isinstance(desc, str) and desc.strip():
        parts += ["[DESCRIPTION]", desc.strip()]
    if isinstance(code, str) and code.strip():
        parts += ["[SOURCE CODE]", code.strip()]
    return "\n\n".join(parts)

def ensure_uid(uid: str, text_full: str) -> str:
    if isinstance(uid, str) and uid.strip():
        return uid
    # deterministic fallback
    return hashlib.md5(text_full.encode("utf-8")).hexdigest()

def load_and_preprocess_folder(path: str, drop_no_focus_for_eval: bool) -> List[dict]:
    """Read raw *.json recursively and return cleaned records like training."""
    patterns = [os.path.join(path, "**", "*.json"), os.path.join(path, "**", "*.JSON")]
    files = sorted({p for patt in patterns for p in glob.glob(patt, recursive=True)})
    if not files:
        print(f"[WARN] No *.json files found under: {path}")
        return []

    print(f"[INFO] Found {len(files)} JSON files under {path}")
    out = []
    no_focus = 0
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            print(f"[WARN] skipping {fp}: {e}")
            continue

        desc = raw.get("prob_desc_description", "") or ""
        code = raw.get("source_code", "") or ""
        text_full = to_text_full(desc, code)
        uid = ensure_uid(raw.get("src_uid", ""), text_full)

        # filter tags to focus set (if present)
        tags = raw.get("tags", [])
        if not isinstance(tags, list):
            tags = [tags] if tags else []
        tags_focus = [t for t in tags if t in FOCUS_LABELS]

        if not tags_focus:
            no_focus += 1
            # keep the row for prediction, but it won't be scored unless you want to
            rec = {"text_full": text_full, "src_uid": uid, "_file_name": os.path.relpath(fp, path)}
            out.append(rec)
            continue

        rec = {
            "text_full": text_full,
            "src_uid": uid,
            "tags": tags_focus,
            "_file_name": os.path.relpath(fp, path)
        }
        out.append(rec)

    print(f"[INFO] Prepared {len(out)} records (rows without focus tags: {no_focus})")
    if drop_no_focus_for_eval:
        # nothing to drop here—prediction keeps all; evaluation function will ignore rows without ground truth
        pass
    return out

def load_and_preprocess_jsonl(path: str) -> List[dict]:
    """Read JSONL that may contain raw-ish rows and normalize to training format."""
    out = []
    for row in iter_jsonl(path):
        text_full = row.get("text_full")
        if not text_full:
            # try to assemble from raw fields if present
            text_full = to_text_full(row.get("prob_desc_description", ""), row.get("source_code", ""))
        uid = ensure_uid(row.get("src_uid", ""), text_full)
        tags = row.get("tags", [])
        if not isinstance(tags, list):
            tags = [tags] if tags else []
        tags_focus = [t for t in tags if t in FOCUS_LABELS]
        rec = {"text_full": text_full, "src_uid": uid}
        if tags_focus:
            rec["tags"] = tags_focus
        out.append(rec)
    print(f"[INFO] Loaded {len(out)} JSONL rows")
    return out

# --------------------------- prediction ---------------------------

def apply_thresholds(prob_row: np.ndarray, labels: List[str], thresholds: Dict[str,float], kcap: int) -> List[str]:
    order = np.argsort(-prob_row)
    kept = [labels[j] for j in order if prob_row[j] >= thresholds.get(labels[j], 0.5)]
    if not kept:
        kept = [labels[int(order[0])]]
    chosen = []
    for j in order:
        lab = labels[j]
        if (prob_row[j] >= thresholds.get(lab, 0.5)) or lab == kept[0]:
            chosen.append(lab)
        if len(chosen) >= kcap:
            break
    return chosen

def predict_records(model, tok, labels, thresholds, records: List[dict],
                    max_len: int, batch_size: int, kcap: int,
                    avoid_repeats: bool, device: str) -> List[dict]:
    used_per_uid = defaultdict(set) if avoid_repeats else None
    outputs = []

    texts = [str(r.get("text_full", "") or "") for r in records]

    with torch.no_grad():
        for i in range(0, len(records), batch_size):
            chunk = records[i:i+batch_size]
            enc = tok(texts[i:i+batch_size], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            probs = torch.sigmoid(model(**enc).logits).detach().cpu().numpy()

            for r, p in zip(chunk, probs):
                uid = r.get("src_uid")
                order = np.argsort(-p)
                chosen = apply_thresholds(p, labels, thresholds, kcap)

                if used_per_uid is not None and uid:
                    used = used_per_uid[uid]
                    final = [lab for lab in chosen if lab not in used]
                    if len(final) < len(chosen):
                        for j in order:
                            cand = labels[j]
                            if cand in used or cand in final: continue
                            if p[j] >= thresholds.get(cand, 0.5) or not final:
                                final.append(cand)
                            if len(final) >= kcap: break
                    chosen = final[:kcap]
                    for lab in chosen: used.add(lab)

                out = {
                    "src_uid": uid,
                    "predicted_tags": chosen,
                    "probs": {labels[i]: float(p[i]) for i in range(len(labels))}
                }
                if "_file_name" in r: out["file_name"] = r["_file_name"]
                if "tags" in r: out["_true_tags"] = r["tags"]
                outputs.append(out)
    return outputs

# --------------------------- evaluation ---------------------------

def evaluate(pred_rows: List[dict], label_list: List[str]) -> Dict:
    y_true, y_pred = [], []
    for r in pred_rows:
        true = r.get("_true_tags")
        if not true:  # skip rows without GT
            continue
        y_true.append([t for t in true if t in label_list])
        y_pred.append([t for t in r.get("predicted_tags", []) if t in label_list])

    if not y_true:
        return {"note": "no ground-truth tags found; evaluation skipped", "n_eval": 0}

    mlb = MultiLabelBinarizer(classes=label_list)
    Y_true = mlb.fit_transform(y_true)
    Y_pred = mlb.transform(y_pred)

    macro = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
    micro = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    samples = f1_score(Y_true, Y_pred, average="samples", zero_division=0)

    p, r, f1, s = precision_recall_fscore_support(Y_true, Y_pred, average=None, zero_division=0)
    per_cls = {lab: {"precision": float(p[i]), "recall": float(r[i]),
                     "f1": float(f1[i]), "support": int(s[i])}
               for i, lab in enumerate(label_list)}
    report = classification_report(Y_true, Y_pred, target_names=label_list, zero_division=0)

    return {"macro_f1": float(macro), "micro_f1": float(micro), "samples_f1": float(samples),
            "per_class": per_cls, "report_text": report, "n_eval": int(Y_true.shape[0])}

# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Predict (and evaluate) with all preprocessing included.")
    ap.add_argument("--model_dir", required=True, help="Fine-tuned model dir with thresholds.json")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_dir", help="Folder of raw sample_*.json (recursive)")
    group.add_argument("--input_jsonl", help="JSONL; if raw-ish, will be normalized")
    ap.add_argument("--output", default="predictions.jsonl", help="Predictions JSONL path")
    ap.add_argument("--eval_out", default="eval_report.txt", help="Human-readable report")
    ap.add_argument("--eval_json", default="eval_report.json", help="JSON report")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--avoid_repeats_per_uid", action="store_true")
    ap.add_argument("--device", choices=["cpu","cuda"], default=None)
    ap.add_argument("--drop_no_focus_for_eval", action="store_true",
                    help="Rows without any focus tag are ignored in evaluation (they already are); predictions still saved.")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    labels, thresholds = load_thresholds(args.model_dir)
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    # Preprocess
    if args.input_dir:
        records = load_and_preprocess_folder(args.input_dir, args.drop_no_focus_for_eval)
    else:
        records = load_and_preprocess_jsonl(args.input_jsonl)

    if not records:
        print("[WARN] No records to process.")
        save_jsonl([], args.output)
        return

    # Predict
    preds = predict_records(
        model, tok, labels, thresholds, records,
        max_len=args.max_len, batch_size=args.batch_size,
        kcap=args.k, avoid_repeats=args.avoid_repeats_per_uid, device=device
    )
    save_jsonl(preds, args.output)
    print(f"✅ Wrote predictions to {args.output}  ({len(preds)} rows)")

    # Evaluate (only uses rows with ground-truth focus tags)
    eval_obj = evaluate(preds, FOCUS_LABELS)
    if eval_obj.get("n_eval", 0) > 0 and "report_text" in eval_obj:
        with open(args.eval_out, "w", encoding="utf-8") as f:
            f.write(eval_obj["report_text"])
        with open(args.eval_json, "w", encoding="utf-8") as f:
            json.dump(eval_obj, f, indent=2)
        print(f"✅ Wrote evaluation to {args.eval_out} and {args.eval_json}")
    else:
        print("ℹ️ No ground-truth focus tags found; evaluation skipped.")

if __name__ == "__main__":
    main()
