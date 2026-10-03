"""Historical and current AI/NN concepts, their relations, and where Toddler uses them.

Content is never written here: every concept is fetched from a real source (Wikipedia
article summaries, CC BY-SA 4.0) by corpus.py. This file only holds the curated structure:
which concepts, how they relate, and which Toddler module or mapping row uses them.
"""

from __future__ import annotations

# id -> (Wikipedia title, year of origin (approx.), era)
CONCEPTS: dict[str, tuple[str, int, str]] = {
    "perceptron": ("Perceptron", 1958, "historic"),
    "hopfield": ("Hopfield_network", 1982, "historic"),
    "backprop": ("Backpropagation", 1986, "historic"),
    "td_learning": ("Temporal_difference_learning", 1988, "historic"),
    "q_learning": ("Q-learning", 1989, "historic"),
    "cnn": ("Convolutional_neural_network", 1989, "historic"),
    "lstm": ("Long_short-term_memory", 1997, "historic"),
    "svm": ("Support_vector_machine", 1995, "historic"),
    "cca": ("Canonical_correlation", 1936, "historic"),
    "reinforcement_learning": ("Reinforcement_learning", 1980, "historic"),
    "loss_function": ("Loss_function", 1950, "historic"),
    "deep_learning": ("Deep_learning", 2012, "current"),
    "neural_network": ("Neural_network_(machine_learning)", 1943, "historic"),
    "attention": ("Attention_(machine_learning)", 2014, "current"),
    "transformer": ("Transformer_(deep_learning_architecture)", 2017, "current"),
    "meta_learning": ("Meta-learning_(computer_science)", 1987, "current"),
    "curriculum_learning": ("Curriculum_learning", 2009, "current"),
    "intrinsic_motivation": ("Intrinsic_motivation_(artificial_intelligence)", 2005, "current"),
    "rlhf": ("Reinforcement_learning_from_human_feedback", 2017, "current"),
    "reward_hacking": ("Reward_hacking", 2016, "current"),
    "ai_alignment": ("AI_alignment", 2016, "current"),
    "vla": ("Vision-language-action_model", 2023, "current"),
    "developmental_robotics": ("Developmental_robotics", 2001, "current"),
    "lidar": ("Lidar", 1961, "historic"),
    "slam": ("Simultaneous_localization_and_mapping", 1986, "historic"),
    "industrial_robot": ("Industrial_robot", 1961, "historic"),
    "unitree": ("Unitree_Robotics", 2016, "current"),
    "clustering_coefficient": ("Clustering_coefficient", 1998, "historic"),
    "louvain": ("Louvain_method", 2008, "current"),
    "diffusion_mri": ("Diffusion_MRI", 1985, "historic"),
    "amygdala": ("Amygdala", 1819, "historic"),
    "inferior_frontal_gyrus": ("Inferior_frontal_gyrus", 1861, "historic"),
    "insular_cortex": ("Insular_cortex", 1796, "historic"),
    "cbcl": ("Child_Behavior_Checklist", 1991, "historic"),
    "peer_to_peer": ("Peer-to-peer", 1999, "historic"),
    "proof_of_work": ("Proof_of_work", 1993, "historic"),
}

# (from, to, kind)
RELATIONS: tuple[tuple[str, str, str], ...] = (
    ("perceptron", "neural_network", "is_precursor_of"),
    ("backprop", "deep_learning", "is_precursor_of"),
    ("neural_network", "deep_learning", "is_precursor_of"),
    ("cnn", "deep_learning", "is_part_of"),
    ("lstm", "transformer", "is_superseded_by"),
    ("attention", "transformer", "is_part_of"),
    ("td_learning", "q_learning", "is_precursor_of"),
    ("reinforcement_learning", "rlhf", "is_precursor_of"),
    ("reinforcement_learning", "reward_hacking", "has_risk"),
    ("rlhf", "ai_alignment", "serves"),
    ("reward_hacking", "ai_alignment", "is_problem_for"),
    ("transformer", "vla", "is_part_of"),
    ("developmental_robotics", "curriculum_learning", "uses"),
    ("intrinsic_motivation", "developmental_robotics", "is_part_of"),
    ("lidar", "slam", "is_input_of"),
    ("diffusion_mri", "clustering_coefficient", "is_input_of"),
    ("clustering_coefficient", "louvain", "is_input_of"),
    ("svm", "cca", "is_complemented_by"),
    ("peer_to_peer", "proof_of_work", "uses"),
)

# concept -> Toddler module or design anchor (wee2017 rows where applicable)
TODDLER_LINKS: dict[str, tuple[str, str]] = {
    "loss_function": ("docs/design/brain.md#1", "learned components minimise a loss; it is not the decision rule"),
    "reinforcement_learning": ("toddler/objective.py", "reward per judge-confirmed task"),
    "reward_hacking": ("toddler/objective.py", "false 'done' is the worst reward; judge, not Toddler, confirms"),
    "rlhf": ("toddler/evaluation.py", "human/judge ratings as T-scores (row 2)"),
    "ai_alignment": ("toddler/stop.py", "STOP rules as constraints, not penalties (row 20)"),
    "meta_learning": ("docs/design/brain.md#5", "few-shot learning target from the toddler benchmark"),
    "curriculum_learning": ("docs/design/brain.md#5", "phase order C0-C3 (row 21)"),
    "intrinsic_motivation": ("toddler/objective.py", "+0.1 reward for new useful knowledge"),
    "developmental_robotics": ("docs/design/wee2017-mapping.md", "early structure shapes later behaviour (row 25)"),
    "clustering_coefficient": ("toddler/structure.py", "clustering_profile (row 9)"),
    "louvain": ("toddler/structure.py", "consensus_partition (row 11)"),
    "svm": ("toddler/evaluation.py", "nested_feature_selection, SVM-RFE (row 15)"),
    "cca": ("toddler/evaluation.py", "cca_test with Bonferroni (row 17)"),
    "diffusion_mri": ("toddler/structure.py", "tractography analogue: module graph from traced paths (rows 7-8)"),
    "amygdala": ("toddler/fastpath.py", "threat detector first and best connected (row 19)"),
    "inferior_frontal_gyrus": ("toddler/stop.py", "response inhibition layer (row 20)"),
    "insular_cortex": ("toddler/fastpath.py", "integration of body/sensor state into the reflex (row 20)"),
    "cbcl": ("toddler/evaluation.py", "two behaviour axes, independent rater (rows 2-3)"),
    "vla": ("docs/design/brain.md#5", "fine-tune open VLA models for physical understanding"),
    "unitree": ("docs/design/brain.md#5", "learned locomotion/manipulation policies, sim-to-real"),
    "industrial_robot": ("toddler/fastpath.py", "KUKA-grade precision and hard force/speed limits"),
    "lidar": ("toddler/fastpath.py", "person-distance sensing for the hard limits"),
    "slam": ("docs/design/brain.md#5", "3D scene from camera + lidar"),
    "peer_to_peer": ("toddler/relay.py", "knitweb peers for Pulse relay"),
    "proof_of_work": ("toddler/relay.py", "proof-of-useful-work verification by sampling"),
    "deep_learning": ("docs/design/brain.md#1", "energy cost term: 20 W brain vs ~1 MW AlphaGo"),
}
