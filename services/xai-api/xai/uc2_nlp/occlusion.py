import io
import csv


_TEXT_KEYS = ("text", "title", "content", "sentence", "review")
_SENT_KEYS = ("finbert_sentiment", "sentiment", "sentiment_label", "label")


def _row_text(row: dict) -> str:
    for k in _TEXT_KEYS:
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _row_sentiment(row: dict) -> str:
    for k in _SENT_KEYS:
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def load_text_from_csv(csv_bytes: bytes, sample_index: int) -> tuple[str, str]:
    """Parse CSV bytes and return (text, true_sentiment) at sample_index."""
    reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8")))
    for i, row in enumerate(reader):
        if i == sample_index:
            return _row_text(row), _row_sentiment(row)
    raise IndexError(f"sample_index {sample_index} out of range")


def run_occlusion(text: str, model_predict, target_class: str = None) -> dict:
    """
    Occlusion-based word importance for text classification.

    Args:
        text: input text
        model_predict: callable(list[str]) -> list[dict]
                       each dict has keys: positive, negative, neutral
        target_class: class to explain. If None, uses the predicted class on full text.

    Returns:
        {
            "target_class": str,
            "baseline_score": float,
            "word_scores": [{"word": str, "index": int, "importance": float}, ...]
        }
    """
    words = text.split()

    # Baseline prediction on full text
    baseline = model_predict([text])[0]
    if target_class is None:
        target_class = max(baseline, key=baseline.get)
    baseline_score = baseline[target_class]

    # Build masked versions — one per word
    masked_texts = []
    for i, word in enumerate(words):
        masked = words[:i] + ["[MASK]"] + words[i + 1:]
        masked_texts.append(" ".join(masked))

    # Single batched model call
    masked_preds = model_predict(masked_texts)

    # Importance = drop in target class score when word is masked
    word_scores = []
    for i, (word, pred) in enumerate(zip(words, masked_preds)):
        importance = baseline_score - pred[target_class]
        word_scores.append({"word": word, "index": i, "importance": round(importance, 6)})

    # Sort by absolute importance descending
    word_scores_sorted = sorted(word_scores, key=lambda x: abs(x["importance"]), reverse=True)

    return {
        "target_class": target_class,
        "baseline_score": round(baseline_score, 6),
        "word_scores": word_scores_sorted,
    }
