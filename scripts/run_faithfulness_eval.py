#!/usr/bin/env python3
"""Run constrained vs naive faithfulness evaluation via ai_outputs API."""

from __future__ import annotations

import json
import os
import argparse
from pathlib import Path
from typing import Dict

import requests


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = ROOT / "faithfulness_eval_results.json"

DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://localhost:3001")
AI_OUTPUTS_URL = os.environ.get("AI_OUTPUTS_URL", "http://localhost:8002")
EVAL_USER = os.environ.get("EVAL_USER", "admin")
def print_table(naive: Dict[str, float], constrained: Dict[str, float]) -> None:
    print("\nMetric                         Naive    Constrained")
    print("-------------------------------------------------------")
    print(f"Grounding completeness         {naive['grounding_completeness']:.2f}         {constrained['grounding_completeness']:.2f}")
    print(f"Hallucination rate             {naive['hallucination_rate']:.2f}         {constrained['hallucination_rate']:.2f}")
    print(f"Citations per response         {naive['citations_per_response']:.1f}          {constrained['citations_per_response']:.1f}")
    print(f"Cross-method accuracy          {naive['cross_method_accuracy']:.2f}         {constrained['cross_method_accuracy']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run faithfulness evaluation via ai_outputs endpoint")
    parser.add_argument("--user_id", default=EVAL_USER, help="User ID to evaluate")
    parser.add_argument("--output", default=str(OUTPUT_FILE), help="Output JSON path")
    parser.add_argument("--max_questions", type=int, default=30, help="Number of evaluation prompts")
    args = parser.parse_args()

    payload = {"user_id": args.user_id, "max_questions": args.max_questions}
    resp = requests.post(f"{AI_OUTPUTS_URL}/evaluate-faithfulness", json=payload, timeout=1800)
    if resp.status_code != 200:
        raise SystemExit(f"Evaluation failed ({resp.status_code}): {resp.text[:500]}")

    result = resp.json()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    naive_metrics = result.get("naive_prompt", result.get("naive", {}))
    constrained_metrics = result.get("constrained_prompt", result.get("constrained", {}))
    print_table(naive_metrics, constrained_metrics)
    print(f"\nSaved results: {out_path}")


if __name__ == "__main__":
    main()
