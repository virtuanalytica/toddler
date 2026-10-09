"""Public G4 practice curriculum. These items are training material, never a promotion test.

The cognitive head is separate from G3's MiniGrid policy. Answer keys below are
manually specified; generated model answers must not be treated as ground truth.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class PracticeItem:
    area: str
    question: str
    options: tuple[str, str, str, str]
    answer: str

    def __post_init__(self) -> None:
        if len(set(self.options)) != 4 or self.answer not in self.options:
            raise ValueError("practice item needs four unique options and a known answer")


# Six Montessori areas are explicit. This is a design taxonomy, not a claim of
# Montessori accreditation or proof of transfer to real-world child learning.
MONTESSORI_AREAS = (
    "montessori_practical_life", "montessori_sensorial", "montessori_language",
    "montessori_mathematics", "montessori_cultural", "montessori_grace_courtesy",
)

KNOWLEDGE_AREAS = (
    "arithmetic", "english", "nederlands", "cantonese", "history", "geography",
    "trivia", "culture", "manners", "puzzles", "iq_reasoning", "eq_social",
)

NAVIGATION_AREAS = ("unlock", "unlockpickup")

# (area, question, correct answer, three distractors). Items stay small and
# auditable. A future Teacher may propose variants, but its keys need an
# independent verifier before they enter training.
_FACTS: tuple[tuple[str, str, str, tuple[str, str, str]], ...] = (
    ("english", "Which word means a place to read books?", "library", ("kitchen", "river", "garden")),
    ("english", "Choose the plural of child.", "children", ("childs", "childes", "child")),
    ("nederlands", "Wat is het meervoud van kind?", "kinderen", ("kinden", "kindjesen", "kind")),
    ("nederlands", "Welke zin is een beleefde vraag?", "Mag ik u iets vragen?", ("Geef dat.", "Nu meteen.", "Weg daar.")),
    ("cantonese", "廣東話「唔該」通常用來表示甚麼？", "勞煩或道謝", ("告別", "數數", "生氣")),
    ("cantonese", "廣東話「早晨」是甚麼意思？", "早上好", ("晚安", "謝謝", "再見")),
    ("history", "Which civilization built the pyramids at Giza?", "Ancient Egyptians", ("Romans", "Vikings", "Aztecs")),
    ("history", "In which century did the printing press spread through Europe?", "15th century", ("5th century", "19th century", "21st century")),
    ("geography", "What is the capital of the Netherlands?", "Amsterdam", ("Utrecht", "Rotterdam", "Eindhoven")),
    ("geography", "Which continent contains Kenya?", "Africa", ("Asia", "Europe", "Oceania")),
    ("trivia", "How many sides does a triangle have?", "3", ("4", "5", "6")),
    ("trivia", "Which planet is called the red planet?", "Mars", ("Venus", "Jupiter", "Neptune")),
    ("culture", "Which celebration is associated with lanterns and the lunar new year?", "Lunar New Year", ("King's Day", "Oktoberfest", "Thanksgiving")),
    ("culture", "What is a respectful first step when you do not know a custom?", "Ask and listen", ("Assume everyone agrees", "Mock the custom", "Ignore the person")),
    ("manners", "You accidentally interrupt someone. What should you do?", "Apologize and let them finish", ("Speak louder", "Walk away", "Interrupt again")),
    ("manners", "Someone lends you a book. What is considerate?", "Return it as agreed", ("Keep it forever", "Damage it", "Forget the lender")),
    ("puzzles", "A key is inside a locked room and the door is open. What should you do first?", "Enter the room", ("Break the wall", "Throw away the key", "Lock the door")),
    ("puzzles", "You have three boxes and one is labeled empty. Which first check is simplest?", "Inspect the labeled box", ("Burn all boxes", "Guess without looking", "Move house")),
    ("iq_reasoning", "Complete the pattern: 2, 4, 6, ?", "8", ("5", "7", "10")),
    ("iq_reasoning", "Which shape differs: square, rectangle, triangle, cube?", "cube", ("square", "rectangle", "triangle")),
    ("eq_social", "A friend looks upset and asks for space. What is respectful?", "Give space and check later", ("Demand an explanation", "Laugh", "Tell everyone")),
    ("eq_social", "You notice you are angry before replying. What can help?", "Pause and breathe", ("Shout at once", "Blame a stranger", "Ignore safety")),
    ("montessori_practical_life", "After pouring water, what is the next caring step if some spills?", "Wipe the spill", ("Leave it slippery", "Hide the cup", "Pour more")),
    ("montessori_practical_life", "How do you carry a fragile glass?", "With steady hands", ("By throwing it", "With eyes closed", "On a moving chair")),
    ("montessori_sensorial", "Which pair compares texture?", "Rough and smooth", ("Yesterday and tomorrow", "North and south", "Yes and no")),
    ("montessori_sensorial", "Which object is the largest: 2 cm, 5 cm, 9 cm, 3 cm?", "9 cm", ("2 cm", "5 cm", "3 cm")),
    ("montessori_language", "Which letter starts the English word apple?", "a", ("b", "c", "d")),
    ("montessori_language", "Which Dutch word rhymes with kat?", "mat", ("boom", "huis", "vis")),
    ("montessori_mathematics", "One bead plus two beads makes how many?", "3", ("1", "2", "4")),
    ("montessori_mathematics", "Which number comes after 6?", "7", ("5", "8", "9")),
    ("montessori_cultural", "A globe is a model of what?", "Earth", ("The Moon", "A city", "A tree")),
    ("montessori_cultural", "Which is a living thing?", "Tree", ("Rock", "Spoon", "Chair")),
    ("montessori_grace_courtesy", "What can you say when asking to join an activity?", "May I join?", ("Move away!", "I own this!", "Be quiet!")),
    ("montessori_grace_courtesy", "Someone is working quietly. What shows respect?", "Wait before interrupting", ("Shout", "Grab their work", "Push them")),
)


def practice_items(seed: int = 0, arithmetic_count: int = 80) -> list[PracticeItem]:
    rng = random.Random(seed)
    out = []
    for area, question, answer, wrong in _FACTS:
        options = list((answer, *wrong))
        rng.shuffle(options)
        out.append(PracticeItem(area, question, tuple(options), answer))
    for _ in range(arithmetic_count):
        a, b = rng.randint(1, 50), rng.randint(1, 50)
        answer = a + b
        distractors = {answer + 1, answer - 1, answer + 10}
        options = [str(answer), *(str(x) for x in distractors)]
        rng.shuffle(options)
        out.append(PracticeItem("arithmetic", f"What is {a} + {b}?", tuple(options), str(answer)))
    return out


def areas_covered(items: list[PracticeItem]) -> set[str]:
    return {item.area for item in items}
