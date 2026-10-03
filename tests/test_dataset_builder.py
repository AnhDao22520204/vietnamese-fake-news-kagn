from pathlib import Path
import json

from scripts.build_dataset import load_json_folder


def test_load_json_folder(tmp_path: Path):
    (tmp_path / "a.json").write_text(
        json.dumps({"text": "Nội dung kiểm thử"}, ensure_ascii=False),
        encoding="utf-8",
    )
    rows = load_json_folder(tmp_path, label=1)
    assert rows == [{"text": "Nội dung kiểm thử", "label": 1}]
