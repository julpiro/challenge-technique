#!/usr/bin/env python3

import os
import json
import pandas as pd

# ---------------- Settings ----------------
RAW_DIR    = "../code_classification_dataset"   # folder with raw *.json files
OUT_FILE   = "clean_dataset.jsonl"           # output JSONL (multi-label)
FOCUS_TAGS = ['math', 'graphs', 'strings', 'number theory', 'trees', 'geometry', 'games', 'probabilities']
# ------------------------------------------

def read_folder_to_df(folder: str) -> pd.DataFrame:
    rows = []
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith('.json'):
            path = os.path.join(folder, fn)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    obj = json.load(f)
                    rows.append(obj)
            except Exception as e:
                print(f"[WARN] skip {fn}: {e}")
    if not rows:
        raise RuntimeError(f"No JSON files found in {folder}")
    return pd.DataFrame(rows)


def as_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    return [x]


def merge_text_only_desc_and_code(row) -> str:
    parts = []
    desc = row.get('prob_desc_description', '')
    code = row.get('source_code', '')
    if isinstance(desc, str) and desc.strip():
        parts.append('[DESCRIPTION]')
        parts.append(desc.strip())
    if isinstance(code, str) and code.strip():
        parts.append('[SOURCE CODE]')
        parts.append(code.strip())
    return "".join(parts)


def main():
    df = read_folder_to_df(RAW_DIR)

    for c in ['prob_desc_description', 'source_code', 'tags', 'src_uid']:
        if c not in df.columns:
            df[c] = '' if c != 'tags' else []

    # Normalize tags to list and filter to focus tags
    df['tags'] = df['tags'].apply(as_list)
    df['tags'] = df['tags'].apply(lambda lst: [t for t in lst if t in FOCUS_TAGS])

    df = df[df['tags'].map(len) > 0].reset_index(drop=True)

    # Build text_full with ONLY description + source_code
    df['text_full'] = df.apply(merge_text_only_desc_and_code, axis=1)

    if 'src_uid' not in df.columns or df['src_uid'].eq('').any():
        df['src_uid'] = pd.util.hash_pandas_object(df['text_full'], index=False).astype(str)

    out = df[['text_full', 'tags', 'src_uid']].copy()

    # Export single JSONL
    out.to_json(OUT_FILE, orient='records', lines=True, force_ascii=False)
    print(f"✅ Wrote {len(out)} samples to {OUT_FILE}")


if __name__ == '__main__':
    main()
