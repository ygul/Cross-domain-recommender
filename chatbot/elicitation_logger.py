from __future__ import annotations

from pathlib import Path
from datetime import datetime
import csv
import json
import uuid


class ElicitationLogger:
    """
    Logs each elicitation session to:
    - CSV for easy Excel analysis
    - JSONL for programmatic analysis (one JSON object per line)
    """

    def __init__(self, base_dir: str = "logs") -> None:
        self.base = Path(base_dir)
        self.base.mkdir(exist_ok=True)

        self.csv_path = self.base / "elicitation_log.csv"
        self.jsonl_path = self.base / "elicitation_log.jsonl"

        # Create CSV header once
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "session_id",
                    "timestamp_utc",
                    "user_seed",
                    "question_1",
                    "answer_1",
                    "question_2",
                    "answer_2",
                    "final_query",
                    "item_types"
                ])

    def log(
        self,
        user_seed: str,
        turns: list,
        final_query: str,
        item_types: set | None
    ) -> None:
        session_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        q1 = turns[0].question if len(turns) > 0 else ""
        a1 = turns[0].answer if len(turns) > 0 else ""
        q2 = turns[1].question if len(turns) > 1 else ""
        a2 = turns[1].answer if len(turns) > 1 else ""

        item_types_str = ",".join(sorted(item_types)) if item_types else "ALL"

        # CSV
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                session_id,
                timestamp,
                user_seed,
                q1,
                a1,
                q2,
                a2,
                final_query,
                item_types_str
            ])

        # JSONL
        record = {
            "session_id": session_id,
            "timestamp_utc": timestamp,
            "user_seed": user_seed,
            "turns": [
                {"question": t.question, "answer": t.answer} for t in turns
            ],
            "final_query": final_query,
            "item_types": sorted(list(item_types)) if item_types else ["ALL"]
        }

        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
