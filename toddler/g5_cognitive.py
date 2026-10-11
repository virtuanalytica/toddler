"""Verified-feedback arithmetic learner layered beside a frozen G4 policy.

This is a public G5 development component, not a promotion evaluator. A trusted
Teacher supplies answer-keyed practice. The learner eliminates arithmetic
hypotheses until one rule remains for an exact question form. It never reads a
private test bank or changes any G4 navigation weight.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Callable

from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline

from toddler.g4_curriculum import PracticeItem
from toddler.g4_student import CognitiveStudent

RULES: dict[str, Callable[[int, int], int]] = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "max": max,
    "min": min,
}

FORMS = {
    "english_plus": ("arithmetic", re.compile(r"What is (\d+) plus (\d+)\?")),
    "english_minus": ("arithmetic", re.compile(r"What is (\d+) minus (\d+)\?")),
    "english_times": ("arithmetic", re.compile(r"What is (\d+) times (\d+)\?")),
    "dutch_plus": ("arithmetic", re.compile(r"Wat is (\d+) plus (\d+)\?")),
    "dutch_minus": ("arithmetic", re.compile(r"Wat is (\d+) min (\d+)\?")),
    "dutch_times": ("arithmetic", re.compile(r"Wat is (\d+) keer (\d+)\?")),
    "cantonese_plus": ("arithmetic", re.compile(r"計算 (\d+) 加 (\d+)。")),
    "cantonese_minus": ("arithmetic", re.compile(r"計算 (\d+) 減 (\d+)。")),
    "cantonese_times": ("arithmetic", re.compile(r"計算 (\d+) 乘 (\d+)。")),
}


def parse_form(item: PracticeItem) -> tuple[str, int, int] | None:
    for form, (area, pattern) in FORMS.items():
        if item.area == area and (match := pattern.fullmatch(item.question)):
            return form, int(match[1]), int(match[2])
    return None


def masked_question(question: str) -> str:
    """Keep operation words while removing memorisable operand values."""
    return re.sub(r"\d+", "<n>", question.casefold())


@dataclass
class SemanticOperationExpert:
    """Learn an arithmetic operation from verified answers across question forms.

    New public feedback can be appended and the classifier refitted. Training
    labels are inferred from the answer and operands, not a Teacher operation
    label. The expert abstains on non-arithmetic and low-confidence questions.
    """

    architecture: str = "tfidf-char-logreg"
    confidence_floor: float = 0.4
    examples: list[tuple[str, str]] = field(default_factory=list)
    seen_sources: set[str] = field(default_factory=set)
    model: object | None = None

    def learn_verified(self, item: PracticeItem, *, source_id: str, key_source: str) -> bool:
        if not source_id.startswith("public-g5-") or source_id in self.seen_sources:
            raise ValueError("semantic learner needs a new public G5 source ID")
        if key_source != "deterministic_integer_arithmetic_v1":
            raise ValueError("semantic feedback needs a deterministic verifier")
        numbers = re.findall(r"\d+", item.question)
        if item.area != "arithmetic" or len(numbers) != 2:
            return False
        a, b = map(int, numbers)
        try:
            expected = int(item.answer)
        except ValueError as exc:
            raise ValueError("semantic feedback answer must be an integer") from exc
        matching = [name for name in ("add", "subtract", "multiply")
                    if RULES[name](a, b) == expected]
        if len(matching) != 1:
            raise ValueError("verified feedback does not identify one arithmetic operation")
        self.examples.append((masked_question(item.question), matching[0]))
        self.seen_sources.add(source_id)
        self.model = None
        return True

    def fit(self) -> None:
        if {operation for _, operation in self.examples} != {"add", "subtract", "multiply"}:
            raise ValueError("semantic learner needs verified examples for all three operations")
        if self.architecture == "tfidf-char-logreg":
            self.model = make_pipeline(
                TfidfVectorizer(analyzer="char", ngram_range=(2, 5)),
                LogisticRegression(max_iter=1000, random_state=42))
        elif self.architecture == "hashing-char-sgd":
            self.model = make_pipeline(
                HashingVectorizer(analyzer="char", ngram_range=(2, 5),
                                  n_features=2**16, alternate_sign=False),
                SGDClassifier(loss="log_loss", alpha=1e-4, random_state=42,
                              max_iter=2000, tol=1e-4))
        else:
            raise ValueError("unknown semantic arithmetic architecture")
        self.model.fit([text for text, _ in self.examples],
                       [operation for _, operation in self.examples])

    def choose(self, item: PracticeItem) -> tuple[str, float] | None:
        numbers = re.findall(r"\d+", item.question)
        if item.area != "arithmetic" or len(numbers) != 2 or self.model is None:
            return None
        a, b = map(int, numbers)
        probabilities = self.model.predict_proba([masked_question(item.question)])[0]
        labels = self.model.classes_
        index = int(probabilities.argmax())
        confidence = float(probabilities[index])
        if confidence < self.confidence_floor:
            return None
        answer = str(RULES[str(labels[index])](a, b))
        return (answer, confidence) if answer in item.options else None

    def manifest(self) -> dict:
        return {"schema": "toddler-g5-semantic-operation/v1",
                "architecture": self.architecture, "confidence_floor": self.confidence_floor,
                "examples": len(self.examples),
                "source_id_sha256": hashlib.sha256(json.dumps(
                    sorted(self.seen_sources), separators=(",", ":")).encode()).hexdigest(),
                "masked_question_sha256": hashlib.sha256(json.dumps(
                    sorted(self.examples), ensure_ascii=False,
                    separators=(",", ":")).encode()).hexdigest()}


@dataclass
class FeedbackArithmetic:
    """A bounded, transparent online learner whose updates require verified keys."""

    hypotheses: dict[str, set[str]] = field(default_factory=dict)
    seen_sources: set[str] = field(default_factory=set)
    accepted_examples: int = 0

    def learn_verified(self, item: PracticeItem, *, source_id: str,
                       key_source: str) -> bool:
        if not source_id.startswith("public-g5-") or source_id in self.seen_sources:
            raise ValueError("training example needs a new public G5 source ID")
        if key_source != "deterministic_integer_arithmetic_v1":
            raise ValueError("arithmetic feedback needs the deterministic key verifier")
        parsed = parse_form(item)
        if parsed is None:
            return False
        form, a, b = parsed
        try:
            expected = int(item.answer)
        except ValueError as exc:
            raise ValueError("verified arithmetic answer must be an integer") from exc
        candidates = self.hypotheses.get(form, set(RULES))
        remaining = {name for name in candidates if RULES[name](a, b) == expected}
        if not remaining:
            raise ValueError("verified key contradicts all current arithmetic hypotheses")
        self.hypotheses[form] = remaining
        self.seen_sources.add(source_id)
        self.accepted_examples += 1
        return True

    def choose(self, item: PracticeItem) -> tuple[str, float] | None:
        parsed = parse_form(item)
        if parsed is None:
            return None
        form, a, b = parsed
        candidates = self.hypotheses.get(form, set())
        if len(candidates) != 1:
            return None
        answer = str(RULES[next(iter(candidates))](a, b))
        return (answer, 1.0) if answer in item.options else None

    def manifest(self) -> dict:
        data = {"schema": "toddler-g5-feedback-arithmetic/v1",
                "hypotheses": {form: sorted(names) for form, names in sorted(self.hypotheses.items())},
                "accepted_examples": self.accepted_examples,
                "source_id_sha256": hashlib.sha256(
                    json.dumps(sorted(self.seen_sources), separators=(",", ":")).encode()).hexdigest()}
        data["sha256"] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        return data


@dataclass
class G5CognitiveStudent:
    """Keep the G4 factual fallback while adding verified online rule learning."""

    base: CognitiveStudent
    arithmetic: FeedbackArithmetic = field(default_factory=FeedbackArithmetic)
    semantic: SemanticOperationExpert | None = None

    def choose(self, item: PracticeItem) -> tuple[str, float]:
        return (self.arithmetic.choose(item)
                or (getattr(self, "semantic", None) and self.semantic.choose(item))
                or self.base.choose(item))

    def accuracy(self, items: list[PracticeItem]) -> float:
        if not items:
            raise ValueError("evaluation needs at least one item")
        return sum(self.choose(item)[0] == item.answer for item in items) / len(items)
