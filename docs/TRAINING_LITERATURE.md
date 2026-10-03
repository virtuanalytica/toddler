# Training literature

The papers Toddler's training and evaluation build on, each with what it changed in the code.
Only primary sources are listed; identifiers are arXiv ids or DOIs.

## Learning algorithm

| Source | What Toddler takes from it | Where |
| ------ | -------------------------- | ----- |
| Schulman, Wolski, Dhariwal, Radford, Klimov (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347 | Clipped surrogate objective, minibatch epochs | `toddler/learn/ppo.py` |
| Schulman, Moritz, Levine, Jordan, Abbeel (2015). High-Dimensional Continuous Control Using Generalized Advantage Estimation. arXiv:1506.02438 | GAE(gamma, lambda) advantages | `toddler/learn/ppo.py` |
| Pardo, Tavakoli, Levdik, Kormushev (2018). Time Limits in Reinforcement Learning. arXiv:1712.00378 | A time-limit truncation is not a termination: bootstrap from the value of the truncated state and never across a reset (the bug fixed in PR #9) | `toddler/learn/ppo.py` |
| Bellemare, Srinivasan, Ostrovski, Schaul, Saxton, Munos (2016). Unifying Count-Based Exploration and Intrinsic Motivation. arXiv:1606.01868 | Count-based novelty bonus (1 / sqrt(count)) with a decaying weight; the "+0.1 for new knowledge" principle | `toddler/learn/ppo.py` (`_novelty`) |
| Pathak, Agrawal, Efros, Darrell (2017). Curiosity-driven Exploration by Self-supervised Prediction. arXiv:1705.05363 | Curiosity as intrinsic reward; a candidate replacement for the count bonus on larger state spaces | backlog |
| Bengio, Louradour, Collobert, Weston (2009). Curriculum Learning. ICML. doi:10.1145/1553374.1553380 | Easy-to-hard ordering of tasks and phases (C0-C3) | phase order C0 -> C3: `docs/design/wee2017-mapping.md` row 21; task order: `docs/TASK_PLAN.md` |

## Toddlers teaching toddlers

| Source | What Toddler takes from it | Where |
| ------ | -------------------------- | ----- |
| Hinton, Vinyals, Dean (2015). Distilling the Knowledge in a Neural Network. arXiv:1503.02531 | Teacher-to-student distillation | `toddler/learn/ppo.py` (`teacher=`), PR #11 |
| Rusu et al. (2016). Policy Distillation. arXiv:1511.06295 | Distilling a policy (KL on action distributions) | PR #11 (NULL in our setting) |
| Ross, Gordon, Bagnell (2011). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger). AISTATS. arXiv:1011.0686 | Behaviour cloning and its compounding-error limits; motivates the cloning warm start followed by self-reinforcement | `toddler/learn/peer.py` (`behaviour_clone`), PR #12 |
| Furlanello, Lipton, Tschannen, Itti, Anandkumar (2018). Born-Again Neural Networks. arXiv:1805.04770 | Generations of students trained from a teacher of the same size | `toddler/learn/generations.py` |
| Jaderberg et al. (2017). Population Based Training of Neural Networks. arXiv:1711.09846 | Population of toddlers, exploit/explore between members | backlog |
| Silver et al. (2017). Mastering the game of Go without human knowledge. Nature 550, 354-359. doi:10.1038/nature24270 | Self-play: learning from copies of oneself | backlog (two-player task) |
| Wang, Lehman, Clune, Stanley (2019). Paired Open-Ended Trailblazer (POET). arXiv:1901.01753 | Co-evolving tasks and agents | backlog |

## Comparing generations fairly

| Source | What Toddler takes from it | Where |
| ------ | -------------------------- | ----- |
| Agarwal, Schwarzer, Castro, Courville, Bellemare (2021). Deep Reinforcement Learning at the Edge of the Statistical Precipice. arXiv:2108.13264 | Normalised scores, aggregate IQM (mean of per-task IQMs), stratified bootstrap confidence intervals, probability of improvement | `toddler/learn/scoring.py` |
| Henderson, Islam, Bachman, Pineau, Precup, Meger (2018). Deep Reinforcement Learning That Matters. arXiv:1709.06560 | Seed variance is large: always several seeds, never one run; report the variance | `toddler/learn/generations.py` (min 5 seeds) |
| Towers et al. (2024). Gymnasium: A Standard Interface for Reinforcement Learning Environments. arXiv:2407.17032 | The task environments and their solve thresholds | `toddler/learn/tasks.py` |

## Physical rules and calibration (Jev, FQ)

| Source | What Toddler takes from it | Where |
| ------ | -------------------------- | ----- |
| Bisk, Zellers, Le Bras, Gao, Choi (2020). PIQA: Reasoning about Physical Commonsense in Natural Language. AAAI. arXiv:1911.11641 | Labelled physical-commonsense questions for measuring and calibrating Jev | `jevserver/calibrate.py` |
| Guo, Pleiss, Sun, Weinberger (2017). On Calibration of Modern Neural Networks. ICML. arXiv:1706.04599 | Expected calibration error (ECE); modern networks are overconfident | `jevserver/calibrate.py` |
| Zadrozny, Elkan (2002). Transforming Classifier Scores into Accurate Multiclass Probability Estimates. KDD. doi:10.1145/775047.775151 | Isotonic regression as calibration layer | `jevserver/calibrate.py` |

## Structure and development

| Source | What Toddler takes from it | Where |
| ------ | -------------------------- | ----- |
| Wee et al. (2017). Neonatal neural networks predict children behavioral profiles later in life. Human Brain Mapping 38(3), 1362-1373. doi:10.1002/hbm.23459 | All 25 analysis steps as design analogies | `docs/design/wee2017-mapping.md` |
| Blondel, Guillaume, Lambiotte, Lefebvre (2008). Fast unfolding of communities in large networks. J. Stat. Mech. arXiv:0803.0476 | Louvain community detection (consensus over runs) | `toddler/structure.py` |

## Statistics already in the merged code (`toddler/evaluation.py`, `toddler/structure.py`)

| Source | What Toddler takes from it | Where |
| ------ | -------------------------- | ----- |
| Peng, Long, Ding (2005). Feature selection based on mutual information: criteria of max-dependency, max-relevance, and min-redundancy. IEEE TPAMI 27(8), 1226-1238. doi:10.1109/TPAMI.2005.159 | mRMR pre-selection of features | `evaluation._mrmr` |
| Guyon, Weston, Barnhill, Vapnik (2002). Gene Selection for Cancer Classification using Support Vector Machines. Machine Learning 46, 389-422. doi:10.1023/A:1012487302797 | SVM-RFE with a linear SVM | `evaluation.nested_feature_selection` |
| Varma, Simon (2006). Bias in error estimation when using cross-validation for model selection. BMC Bioinformatics 7, 91. doi:10.1186/1471-2105-7-91 | Nested cross-validation: select features in inner folds only | `evaluation.nested_feature_selection` |
| Hotelling (1936). Relations between two sets of variates. Biometrika 28(3/4), 321-377. doi:10.2307/2333955 | Canonical correlation analysis | `evaluation.cca_test` |
| Bartlett (1941). The statistical significance of canonical correlations. Biometrika 32(1), 29-37. doi:10.2307/2332254 | Chi-square test of the canonical correlation | `evaluation.cca_test` |
| Dunn (1961). Multiple comparisons among means. JASA 56(293), 52-64. doi:10.1080/01621459.1961.10482090 | Bonferroni correction over several tests | `evaluation.cca_test` (`p_bonferroni`) |
| Welch (1947). The generalization of "Student's" problem when several different population variances are involved. Biometrika 34(1/2), 28-35. doi:10.2307/2332510 | Unequal-variance t-test for best vs worst groups | `evaluation.compare_extremes` |
| McCall (1922). How to Measure in Education. New York: Macmillan | T-scale (50 + 10 z) | `evaluation.t_scores` |
| Yerushalmy (1947). Statistical problems in assessing methods of medical diagnosis, with special reference to X-ray techniques. Public Health Reports 62(40), 1432-1449. doi:10.2307/4586294 | Sensitivity and specificity as separate error rates (Wee et al. report sens 0.954 / spec 0.765, mapping row 16) | `evaluation.sens_spec`, FQ reflex scores in `toddler/quotients.py` |
| Mann, Whitney (1947). On a test of whether one of two random variables is stochastically larger than the other. Ann. Math. Statist. 18(1), 50-60. doi:10.1214/aoms/1177730491 | One-sided rank test between groups of runs | `toddler/learn/peer.py`, `generations.decide_promotion` |
| Efron (1979). Bootstrap methods: another look at the jackknife. Ann. Statist. 7(1), 1-26. doi:10.1214/aos/1176344552 | Bootstrap confidence intervals (stratified per task as in Agarwal et al.) | `toddler/learn/scoring.bootstrap_ci` |
| Dixon, Mood (1946). The statistical sign test. JASA 41(236), 557-566. doi:10.1080/01621459.1946.10501898 | Sign test per edge across runs | `structure.sign_test_edges` |
| Watts, Strogatz (1998). Collective dynamics of "small-world" networks. Nature 393, 440-442. doi:10.1038/30918 | Local clustering coefficient | `structure.clustering_profile` |
| Newman (2006). Modularity and community structure in networks. PNAS 103(23), 8577-8582. doi:10.1073/pnas.0601602103 | Modularity Q | `structure.consensus_partition`, `structure.modularity_null_test` |
| Lancichinetti, Fortunato (2012). Consensus clustering in complex networks. Scientific Reports 2, 336. doi:10.1038/srep00336 | Consensus over many stochastic Louvain runs (co-assignment) | `structure.consensus_partition` |
| Maslov, Sneppen (2002). Specificity and stability in topology of protein networks. Science 296(5569), 910-913. doi:10.1126/science.1065103 | Degree-preserving edge-swap null model | `structure.modularity_null_test` |

## Lessons already learned in this repository

- A plausible-looking gain can come from the training change itself, not from the teacher: in PR #11 the real teacher beat "no teacher" (p 0.047) but not a random teacher (p 0.85). Always include a control that receives the same mechanism without the knowledge.
- Register the protocol before running a follow-up (PR #12): the cloning result then means something, and it replicated after the PPO fix.
- Fix the learner before trusting any comparison: the truncation bug (Pardo et al.) changed which runs could be compared; every affected experiment was re-run.
