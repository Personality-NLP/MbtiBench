import json
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    raw_dataset_path = Path("downstream/Dreaddit/data/raw/new_dreaddit_test.csv")
    save_data_path = Path("downstream/Dreaddit/data/test.jsonl")
    all_data = pd.read_csv(raw_dataset_path, header=None, names=["id", "post", "question", "label"])

    max_examples = 500

    with open(save_data_path, "w") as f:
        for i in range(1, min(len(all_data), max_examples)):
            try:
                posts = all_data["post"][i]
                label = all_data["label"][i]
                assert label in ["yes", "no"]
                obj = {"id": i, "posts": posts, "label": label}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            except Exception as e:
                print(e)
