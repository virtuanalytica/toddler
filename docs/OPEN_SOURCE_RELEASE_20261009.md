# Public release of Toddler, 9 October 2026

The originator requested that Toddler be open for crowdsourced contributions.
The repository's original code, documentation and generated reports now use
the canonical [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.html)
text in `LICENSE`; `pyproject.toml` declares `Apache-2.0`. `NOTICE` preserves
VirtualV Holding B.V.'s attribution, identifies the Wikipedia extracts under
CC BY-SA 4.0, and states that externally stored model weights, private data,
private seeds and third-party models are not redistributed or relicensed by
this repository. Each Wikipedia extract also retains its source URL, revision,
license and extract hash.

Before the visibility change, the remote Git object graph was scanned: 46
remote branches, 1,260 reachable objects and 428 blobs. The scan found no
private-key headers, common GitHub/OpenAI/Anthropic/AWS token patterns, bearer
tokens, credential-like environment assignments or secret-looking file paths.
The workflow history was inspected for secrets injected into GitHub Actions;
the repository test workflow uses no secrets. This is a pattern-based audit,
not proof that no sensitive information exists anywhere in prose or past
Actions output. Public visibility exposes branches, history and Actions logs,
as [GitHub documents](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/setting-repository-visibility).

`CONTRIBUTING.md`, the Toddler PR template and `docs/learn/DISTRIBUTED_TRIALS.md`
describe a public fork-to-PR path. A new contributor must disclose all public
or private models influencing the method, training commands, data and
interaction provenance, and the exact trial protocol/report hashes. CI checks
the submission format and hash references with
`scripts/validate_contribution.py`. An independent evaluator still runs fresh
hidden tests before a candidate can be promoted. The ancestor gate protects
skills mastered by surviving predecessors. G3 remains the archived final
originator-signed pivot; later promotions rely on auditable trials and the
lineage record.

The default test workflow has read-only `contents` permission and checkout
does not persist its credential for code submitted through fork PRs.
