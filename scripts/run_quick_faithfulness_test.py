#!/usr/bin/env python3
"""
Quick faithfulness evaluation test with a small test set
"""

import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Test with a minimal test set
minimal_test_set = {
    "metadata": {
        "created_at": "2026-02-05T20:00:00",
        "total_test_cases": 3,
        "categories": {
            "feature_importance": 1,
            "attention": 1,
            "confidence": 1
        }
    },
    "test_cases": [
        {
            "prompt": "What are the most important words for this prediction?",
            "category": "feature_importance",
            "expected_features": [("earnings", 0.15), ("growth", 0.12), ("revenue", 0.11)],
            "expected_tokens": [],
            "expected_values": {}
        },
        {
            "prompt": "Which tokens did the model pay most attention to?",
            "category": "attention",
            "expected_features": [],
            "expected_tokens": [("earnings", 0.18), ("growth", 0.16), ("revenue", 0.14)],
            "expected_values": {}
        },
        {
            "prompt": "What is the confidence score for this prediction?",
            "category": "confidence",
            "expected_features": [],
            "expected_tokens": [],
            "expected_values": {"confidence_score": 0.87}
        }
    ]
}

# Save minimal test set
test_set_path = "ai_outputs/test_set_minimal.json"
os.makedirs(os.path.dirname(test_set_path), exist_ok=True)
with open(test_set_path, 'w') as f:
    json.dump(minimal_test_set, f, indent=2)

print(f"Created minimal test set with {len(minimal_test_set['test_cases'])} test cases")
print(f"Saved to: {test_set_path}")
print("\nNow run the evaluation with:")
print(f"  curl -X POST http://localhost:8002/evaluate-faithfulness \\")
print(f"    -H 'Content-Type: application/json' \\")
print(f"    -d '{{\"user_id\": \"admin\", \"test_set_path\": \"{test_set_path}\"}}'")
