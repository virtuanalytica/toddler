from toddler.g4_curriculum import (KNOWLEDGE_AREAS, MONTESSORI_AREAS,
                                    NAVIGATION_AREAS, areas_covered, practice_items)
from toddler.g4_student import CognitiveStudent


def test_curriculum_covers_requested_areas():
    items = practice_items(seed=3, arithmetic_count=20)
    assert set(KNOWLEDGE_AREAS + MONTESSORI_AREAS) <= areas_covered(items)
    assert NAVIGATION_AREAS == ("unlock", "unlockpickup")
    assert all(item.answer in item.options and len(set(item.options)) == 4 for item in items)


def test_cognitive_head_trains_and_returns_one_of_four_choices():
    items = practice_items(seed=4, arithmetic_count=20)
    student = CognitiveStudent.train(items)
    answer, confidence = student.choose(items[0])
    assert answer in items[0].options and 0 <= confidence <= 1
    assert 0 <= student.accuracy(items) <= 1
    unseen_arithmetic = [item for item in practice_items(seed=777, arithmetic_count=50)
                         if item.area == "arithmetic"]
    assert student.accuracy(unseen_arithmetic) == 1.0
