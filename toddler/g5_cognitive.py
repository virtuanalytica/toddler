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

    def choose(self, item: PracticeItem) -> tuple[str, float]:
        return self.arithmetic.choose(item) or self.base.choose(item)

    def accuracy(self, items: list[PracticeItem]) -> float:
        if not items:
            raise ValueError("evaluation needs at least one item")
        return sum(self.choose(item)[0] == item.answer for item in items) / len(items)
