# G4 automatic audit handoff

The Teacher timer writes searched cohorts under `~/.local/share/teacher/g4-search/`. On a successful Teacher service exit, `teacher-g4-audit.service` runs `python3 -m scripts.g4_auto_audit`. It accepts only the newest completed cohort from the last six hours; an incomplete newest run cannot be replaced by an older one.

The handoff validates the five frozen children, inherited weights, route selection and hashes. If fewer than five children improve on both public development and public check, it records `public_gate_failed` without generating private seeds. The current 2026-10-10 cohort selected four of five and therefore remains a research candidate. G3 stays official.

If all five pass, the handoff reserves the single `G4-search` confirmatory family, writes two committed private seed sets, evaluates both, and promotes only if the frozen replication gate passes. The family lock prevents a new cohort from retrying the same hidden test. Interrupted runs resume the same protocol and existing trial files. A changed manifest, invalid trial or missing protocol yields an explicit failure status; inspect and repair the existing attempt rather than creating fresh seeds.

Statuses are stored with owner-only permissions under `~/.local/share/toddler/g4-search-private/status/`. A failed service appears in `systemctl --user status teacher-g4-audit.service`; `journalctl --user -u teacher-g4-audit.service` shows the status path. The official generation is only `G4-search` after a verified `survived` lineage verdict. Future G5+ audits need a new preregistered statistical family and independent hidden sets.
