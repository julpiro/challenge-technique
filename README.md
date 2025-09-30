## Usage

Run the CLI on a folder of raw JSON problem files (e.g. sample_0.json, sample_1.json).

You just need to replace 'path/to/test_data':

```bash
python predict_folder_cli.py \
  --model_id julpiro/challenge_technique \
  --input_dir path/to/test_data \
  --output predictions.jsonl \
  --k 3 \
  --avoid_repeats_per_uid
```

## Key options

--model_id : Hugging Face model repo id.

--input_dir : Folder containing raw .json problem files.

--output : Path to save predictions (default: predictions.jsonl).

--k : Maximum number of tags per sample (default: 3).

--avoid_repeats_per_uid : Prevents predicting the same tag twice for the same problem.

## Environment

If needed in train LLM/environment.yml
