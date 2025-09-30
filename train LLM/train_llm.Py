# train_llm.py
import os, json, numpy as np, pandas as pd, torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)

# ----------------- Config -----------------
CLEAN_FILE = "clean_dataset.jsonl"   # from clean_dataset.py (columns: text_full, tags(list), src_uid)
TRAIN_FILE = "train.jsonl"           # will be written by this script
TEST_FILE  = "test.jsonl"            # will be written by this script

MODEL_NAME = "microsoft/codebert-base"     # or "distilbert-base-uncased"
MAX_LEN    = 512
LR         = 2e-5
EPOCHS     = 3
BATCH_TR   = 16
BATCH_EV   = 32
SEED       = 42
VAL_FRAC   = 0.15  # fraction of TRAIN reserved to learn thresholds & early eval

LABELS = ['math','graphs','strings','number theory','trees','geometry','games','probabilities']
OUT_DIR = "llm_ft_out"
MAX_PRED_TAGS = 3  # cap predicted tags per sample
# ------------------------------------------

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED); np.random.seed(SEED)

def ensure_splits():
    """Create group-aware train/test if not already existing."""
    if os.path.exists(TRAIN_FILE) and os.path.exists(TEST_FILE):
        return
    df = pd.read_json(CLEAN_FILE, lines=True)
    assert {"text_full","tags","src_uid"}.issubset(df.columns)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    tr_idx, te_idx = next(splitter.split(df, groups=df["src_uid"]))
    df.iloc[tr_idx].to_json(TRAIN_FILE, orient="records", lines=True, force_ascii=False)
    df.iloc[te_idx].to_json(TEST_FILE,  orient="records", lines=True, force_ascii=False)
    print(f"✅ Wrote {TRAIN_FILE} ({len(tr_idx)}), {TEST_FILE} ({len(te_idx)})")

def load_splits():
    train_df = pd.read_json(TRAIN_FILE, lines=True)
    test_df  = pd.read_json(TEST_FILE,  lines=True)
    return train_df, test_df

class MultiLabelDS(torch.utils.data.Dataset):
    def __init__(self, texts, Y, tok, max_len=512, with_labels=True):
        self.texts, self.Y = texts, Y
        self.tok, self.max_len, self.with_labels = tok, max_len, with_labels
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len)
        item = {k: torch.tensor(v) for k,v in enc.items()}
        if self.with_labels:
            item["labels"] = torch.tensor(self.Y[i], dtype=torch.float)
        return item

def learn_per_class_thresholds(probs, Y_true, labels):
    """Pick τ_c maximizing per-class F1 on a validation set."""
    thresholds = {}
    for j, lab in enumerate(labels):
        s = probs[:, j]
        y = Y_true[:, j]
        # candidate thresholds: unique rounded probs or quantiles (bounded)
        cand = np.unique(np.round(s, 4))
        if cand.size > 400:
            qs = np.linspace(0.02, 0.98, 200)
            cand = np.quantile(s, qs)
        best_f1, best_t = -1.0, 0.5
        for t in cand:
            f1 = f1_score(y, (s >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[lab] = best_t
    return thresholds

def apply_thresholds(prob_row, labels, thresholds, kcap=3):
    labs = [labels[j] for j,p in enumerate(prob_row) if p >= thresholds.get(labels[j], 0.5)]
    if not labs:  # ensure at least one
        labs = [labels[int(np.argmax(prob_row))]]
    # sort by prob desc and cap
    order = np.argsort(-prob_row)
    labs_sorted = []
    for j in order:
        lab = labels[j]
        if (prob_row[j] >= thresholds.get(lab, 0.5)) or lab == labs[0]:
            labs_sorted.append(lab)
        if len(labs_sorted) >= kcap:
            break
    return labs_sorted

def main():
    ensure_splits()
    train_df, test_df = load_splits()

    # ----- MultiLabel binarization -----
    mlb = MultiLabelBinarizer(classes=LABELS)
    Y_train_full = mlb.fit_transform(train_df["tags"].tolist())  # keeps order of LABELS

    # ----- carve small val from TRAIN (random; could do group-aware if needed) -----
    tr_idx, va_idx = train_test_split(
        np.arange(len(train_df)), test_size=VAL_FRAC,
        random_state=SEED, stratify=Y_train_full.argmax(1)
    )
    tr_df, va_df = train_df.iloc[tr_idx], train_df.iloc[va_idx]
    Y_tr, Y_va   = Y_train_full[tr_idx], Y_train_full[va_idx]

    # ----- Tokenizer, model, data -----
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), problem_type="multi_label_classification"
    )

    dtr = MultiLabelDS(tr_df["text_full"].astype(str).tolist(), Y_tr, tok, MAX_LEN, with_labels=True)
    dva = MultiLabelDS(va_df["text_full"].astype(str).tolist(), Y_va, tok, MAX_LEN, with_labels=True)
    dte = MultiLabelDS(test_df["text_full"].astype(str).tolist(), None, tok, MAX_LEN, with_labels=False)

    collate = DataCollatorWithPadding(tok, pad_to_multiple_of=None)

    # ----- Trainer -----
    args = TrainingArguments(
        output_dir=OUT_DIR,
        learning_rate=LR,
        weight_decay=0.01,
        per_device_train_batch_size=BATCH_TR,
        per_device_eval_batch_size=BATCH_EV,
        num_train_epochs=EPOCHS,
        logging_steps=50,
        seed=SEED
    )


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1/(1+np.exp(-logits))
        yhat = (probs >= 0.5).astype(int)  # temp; proper τ learned after
        return {
            "micro_f1": f1_score(labels, yhat, average="micro", zero_division=0),
            "macro_f1": f1_score(labels, yhat, average="macro", zero_division=0),
            "samples_f1": f1_score(labels, yhat, average="samples", zero_division=0),
        }

    trainer = Trainer(
        model=model, args=args,
        train_dataset=dtr, eval_dataset=dva,
        data_collator=collate, tokenizer=tok,
        compute_metrics=compute_metrics
    )
    trainer.train()

    metrics = trainer.evaluate()
    print(metrics)

    trainer.save_model(OUT_DIR); tok.save_pretrained(OUT_DIR)

    # ----- Learn thresholds on VA -----
    with torch.no_grad():
        va_logits = trainer.predict(dva).predictions
    va_probs = 1/(1+np.exp(-va_logits))
    thresholds = learn_per_class_thresholds(va_probs, Y_va, LABELS)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": LABELS, "thresholds": thresholds}, f, indent=2)
    print("Per-class thresholds:", thresholds)

    # ----- Evaluate on TEST with thresholds -----
    with torch.no_grad():
        te_logits = trainer.predict(dte).predictions
    te_probs = 1/(1+np.exp(-te_logits))

    y_pred = [apply_thresholds(p, LABELS, thresholds, kcap=MAX_PRED_TAGS) for p in te_probs]
    y_true = [[t for t in tags if t in LABELS] for tags in test_df["tags"].tolist()]
    # ensure at least one label in y_true (should be true already after cleaning)
    y_true = [labs if labs else [LABELS[0]] for labs in y_true]

    Y_true = mlb.transform(y_true)
    Y_pred = mlb.transform(y_pred)

    print("\nMacro F1:", f1_score(Y_true, Y_pred, average="macro", zero_division=0))
    print("Micro F1:", f1_score(Y_true, Y_pred, average="micro", zero_division=0))
    print("Samples F1:", f1_score(Y_true, Y_pred, average="samples", zero_division=0))
    print("\nPer-class report:\n", classification_report(Y_true, Y_pred, target_names=LABELS, zero_division=0))

    # ----- Save test predictions -----
    with open(os.path.join(OUT_DIR, "predictions_test.jsonl"), "w", encoding="utf-8") as f:
        for uid, probs, labs in zip(test_df.get("src_uid", [""]*len(test_df)), te_probs, y_pred):
            f.write(json.dumps({
                "src_uid": uid,
                "predicted_tags": labs,
                "probs": {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
            }, ensure_ascii=False) + "\n")
    print(f"✅ Done. Artifacts saved in {OUT_DIR}")

if __name__ == "__main__":
    main()
