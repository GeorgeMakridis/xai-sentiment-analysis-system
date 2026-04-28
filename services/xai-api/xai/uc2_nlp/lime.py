import numpy as np
from lime.lime_text import LimeTextExplainer


CLASSES = ["positive", "negative", "neutral"]


def run_lime(text: str, model_predict, num_features: int = 10, num_samples: int = 300) -> dict:
    """
    LIME-based word importance for text classification.

    Args:
        text: input text
        model_predict: callable(list[str]) -> list[dict]
                       each dict has keys: positive, negative, neutral
        num_features: number of top words to return
        num_samples: number of perturbed samples for LIME

    Returns:
        {
            "target_class": str,
            "word_scores": [{"word": str, "importance": float}, ...]
        }
    """
    def predict_proba(texts: list[str]):
        preds = model_predict(texts)
        return np.array([[p["positive"], p["negative"], p["neutral"]] for p in preds])

    explainer = LimeTextExplainer(class_names=CLASSES)
    baseline = model_predict([text])[0]
    target_class = max(baseline, key=baseline.get)
    target_index = CLASSES.index(target_class)

    explanation = explainer.explain_instance(
        text,
        predict_proba,
        num_features=num_features,
        num_samples=num_samples,
        labels=[target_index],
    )

    word_scores = [
        {"word": word, "importance": round(score, 6)}
        for word, score in explanation.as_list(label=target_index)
    ]
    word_scores_sorted = sorted(word_scores, key=lambda x: abs(x["importance"]), reverse=True)

    return {
        "target_class": target_class,
        "word_scores": word_scores_sorted,
    }
