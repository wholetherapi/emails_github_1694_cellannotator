import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def summarise(predictions, labels):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }


def calibrated_confidence(logits):
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return probs.max(axis=1)
