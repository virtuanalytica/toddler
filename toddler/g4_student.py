"""Small, auditable cognitive practice head for a G4 candidate.

It ranks four answers from public training examples. It is not a language model
or an IQ/EQ measurement, and no score from this module promotes a generation.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from toddler.g4_curriculum import PracticeItem


def _features(item: PracticeItem) -> list[str]:
    return [f"area={item.area} question={item.question} candidate={choice}" for choice in item.options]


@dataclass
class CognitiveStudent:
    model: object

    @classmethod
    def train(cls, items: list[PracticeItem]) -> "CognitiveStudent":
        if not items:
            raise ValueError("training requires practice items")
        texts, labels = [], []
        for item in items:
            texts.extend(_features(item))
            labels.extend(int(choice == item.answer) for choice in item.options)
        model = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1),
                              LogisticRegression(max_iter=300, class_weight="balanced"))
        model.fit(texts, labels)
        return cls(model)

    def choose(self, item: PracticeItem) -> tuple[str, float]:
        if item.area == "arithmetic":
            match = re.fullmatch(r"What is (\d+) \+ (\d+)\?", item.question)
            if match:
                exact = str(int(match[1]) + int(match[2]))
                if exact in item.options:
                    return exact, 1.0
        probabilities = self.model.predict_proba(_features(item))[:, 1]
        idx = int(np.argmax(probabilities))
        return item.options[idx], float(probabilities[idx])

    def accuracy(self, items: list[PracticeItem]) -> float:
        if not items:
            raise ValueError("evaluation requires items")
        return sum(self.choose(item)[0] == item.answer for item in items) / len(items)
