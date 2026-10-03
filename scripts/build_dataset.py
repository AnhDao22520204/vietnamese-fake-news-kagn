from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_json_folder(folder: str | Path, label: int):
    folder = Path(folder)
    rows = []

    for path in sorted(folder.glob("*.json")):
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
            text = str(content.get("text", "")).strip()
            if text:
                rows.append({"text": text, "label": label})
        except Exception as exc:
            print(f"Skipping {path}: {exc}")

    return rows


def parse_args():
    p = argparse.ArgumentParser(description="Build fnd_dataset.csv from VFND JSON folders.")
    p.add_argument("--real-dir", required=True)
    p.add_argument("--fake-article-dir", required=True)
    p.add_argument("--fake-social-dir", required=True)
    p.add_argument("--output", default="data/processed/fnd_dataset.csv")
    return p.parse_args()


def main():
    args = parse_args()

    rows = []
    rows += load_json_folder(args.real_dir, 1)
    rows += load_json_folder(args.fake_article_dir, 0)
    rows += load_json_folder(args.fake_social_dir, 0)

    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Saved {len(df)} rows to {out}")
    print(df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
