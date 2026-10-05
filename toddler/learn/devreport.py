"""Development report (ontwikkelverslag) per surviving generation, as a PDF.

Built only from recorded facts: the generation register (weights, meta.json), the evolution ledger
(lineage.jsonl: births, inheritance, selection, derived agents) and the generation's JSON report.
Nothing is typed in by hand, so the PDF proves what hardware, software, code, data and inherited
weights went into the generation, and every weights file is re-hashed while the report is built.

Only generations with the ledger verdict "survived" get a report; extinct generations and control
arms stay in the ledger (they are evidence) but are not documented as a line of descent.
"""

from __future__ import annotations

import hashlib
import html
import json
import time
from pathlib import Path

from toddler.learn import generations as G
from toddler.learn import lineage as L

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; background: #fbf8f1;
        @bottom-center { content: "Toddler ontwikkelverslag " string(gen) "  -  " counter(page) "/" counter(pages);
                         font: 8pt Charter, 'Bitstream Charter', Georgia, serif; color: #6b6355; } }
body { font: 10pt/1.45 Charter, 'Bitstream Charter', Georgia, 'DejaVu Serif', serif; color: #1f1d1a; background: #fbf8f1; }
h1 { string-set: gen attr(data-gen); font-size: 22pt; color: #1d3a6e; margin: 0 0 2mm; }
h2 { font-size: 13pt; color: #1d3a6e; border-bottom: 0.6pt solid #c9bfa9; padding-bottom: 1mm; margin-top: 7mm; }
.sub { color: #6b6355; margin-bottom: 5mm; }
table { border-collapse: collapse; width: 100%; font-size: 8.4pt; margin: 2mm 0; }
th, td { border-bottom: 0.4pt solid #ddd3bf; padding: 1.2mm 1.6mm; text-align: left; vertical-align: top; }
th { background: #efe8d8; }
code, .mono { font-family: 'DejaVu Sans Mono', monospace; font-size: 7.4pt; word-break: break-all; }
.verdict { display: inline-block; padding: 0.8mm 3mm; border-radius: 2mm; background: #1d3a6e; color: #fff; }
.note { color: #6b6355; font-size: 8.4pt; }
"""


def _e(x) -> str:
    return html.escape(str(x))


def _table(rows: list[list], head: list[str]) -> str:
    th = "".join(f"<th>{_e(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{th}</tr>{body}</table>"


def _tree_svg(led: L.Ledger, generation: str) -> str:
    """Family tree of this generation: every member and all its ancestors, one column per generation."""
    nodes, edges = {}, []
    for m in led.members(generation):
        nodes[m["ref"]] = m
        for a in led.ancestry(m["ref"]):
            nodes.setdefault(a["parent"], led.birth(a["parent"]) or {"ref": a["parent"]})
            edges.append((a["parent"], a["child"]))
    gens = sorted({r.split("/", 1)[0] for r in nodes}, key=lambda g: (len(g), g))
    cols = {g: sorted(r for r in nodes if r.startswith(g + "/")) for g in gens}
    w, h = 170 * len(gens) + 40, 28 * max(len(c) for c in cols.values()) + 40
    pos = {r: (30 + 170 * i, 30 + 28 * j) for i, g in enumerate(gens) for j, r in enumerate(cols[g])}
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="Georgia" font-size="10">']
    for g, i in zip(gens, range(len(gens))):
        out.append(f'<text x="{30 + 170 * i}" y="16" fill="#1d3a6e" font-weight="bold">{_e(g)}</text>')
    for p, c in dict.fromkeys(edges):
        (x1, y1), (x2, y2) = pos[p], pos[c]
        out.append(f'<line x1="{x1 + 70}" y1="{y1 - 4}" x2="{x2}" y2="{y2 - 4}" stroke="#8a7f69" stroke-width="0.8"/>')
    for r, (x, y) in pos.items():
        fill = "#1d3a6e" if r.startswith(generation + "/") else "#6b6355"
        out.append(f'<text x="{x}" y="{y}" fill="{fill}">{_e(r.split("/", 1)[1])}</text>')
    out.append("</svg>")
    return "".join(out)


def build_html(root: Path, generation: str, report_json: Path | None = None) -> str:
    root = Path(root)
    led = L.Ledger(root)
    verdict = led.verdict(generation)
    if not verdict or verdict["verdict"] != "survived":
        raise PermissionError(f"{generation}: a development report is made only for surviving generations "
                              f"(ledger verdict: {verdict['verdict'] if verdict else 'none'})")
    ok_chain, bad_seq = led.verify()
    reg = G.Registry(root)
    members = led.members(generation)
    rep = json.loads(report_json.read_text()) if report_json and report_json.exists() else {}

    # integrity: re-hash every weights file now and compare with ledger and meta.json
    integ = []
    for m in members:
        tid = m["ref"].split("/", 1)[1]
        f = root / generation / tid / "weights.pt"
        now = hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else "missing"
        rec = reg.generation(generation)
        meta_sha = next((r.weights_sha256 for r in rec if r.toddler_id == tid), "missing")
        integ.append([_e(m["ref"]), f"<code>{_e(now)}</code>",
                      "ok" if now == m["weights_sha256"] == meta_sha else "<b>AFWIJKING</b>"])

    first = members[0] if members else {}
    hw_sets = {json.dumps(m["hardware"], sort_keys=True) for m in members}
    sw_sets = {json.dumps(m["software"], sort_keys=True) for m in members}
    code_sets = {json.dumps(m["code"], sort_keys=True) for m in members}
    data = first.get("data", {})

    parts = [f"<style>{CSS}</style>",
             f'<h1 data-gen="{_e(generation)}">Ontwikkelverslag {_e(generation)}</h1>',
             f'<div class="sub">Opgemaakt {time.strftime("%Y-%m-%d %H:%M %Z")} uit het generatieregister en het '
             f'evolutieregister. Status: <span class="verdict">{_e(verdict["verdict"])}</span></div>',
             "<h2>1. Selectie</h2>",
             f"<p>Besluit vastgelegd in het evolutieregister (rij {verdict['_seq']}). Onderbouwing:</p>",
             f"<pre class='mono'>{_e(json.dumps(verdict['evidence'], indent=1))}</pre>",
             "<h2>2. Populatie</h2>",
             _table([[_e(m["ref"]), _e(m["role"]), _e(m["_event"]),
                      "<br>".join(f"{_e(i['parent'])}: {_e(', '.join(i['modules']))}" for i in m["inherits"]) or "geen (oorsprong)"]
                     for m in members], ["toddler", "rol", "registratie", "geërfd van (modules)"]),
             "<h2>3. Overerving (generatiedata)</h2>",
             _table([[_e(m["ref"]), _e(i["parent"]), f"<code>{_e(i['parent_weights_sha256'])}</code>"]
                     for m in members for i in m["inherits"]] or [["-", "-", "geen ouders: oorsprongsgeneratie"]],
                    ["kind", "ouder", "sha256 oudergewichten"]),
             "<p class='note'>Elke geërfde ouder is bij de geboorte gecontroleerd tegen zijn eigen registratie; een "
             "afwijkende hash wordt door het register geweigerd.</p>",
             "<h2>4. Stamboom</h2>", _tree_svg(led, generation),
             "<h2>5. Hardware</h2>",
             *[f"<pre class='mono'>{_e(json.dumps(json.loads(h), indent=1))}</pre>" for h in sorted(hw_sets)],
             "<h2>6. Software en code</h2>",
             *[f"<pre class='mono'>{_e(json.dumps(json.loads(x), indent=1))}</pre>" for x in sorted(sw_sets | code_sets)],
             "<h2>7. Trainingsdata en evaluatie</h2>",
             f"<pre class='mono'>{_e(json.dumps(data, indent=1))}</pre>",
             "<p class='note'>Een verborgen seed-set wordt alleen met zijn commitment vermeld tot het venster sluit; "
             "daarna is hij te controleren met <code>python3 -m toddler.learn.secret_seeds reveal &lt;set_id&gt;</code>.</p>",
             "<h2>8. Budget</h2>",
             _table([[_e(m["ref"]), f"<code>{_e(json.dumps(m['budget']))}</code>"] for m in members], ["toddler", "budget"]),
             ]
    if rep:
        keep = {k: rep[k] for k in rep if k not in ("hardware", "software", "per_toddler")}
        parts += ["<h2>9. Resultaten (generatierapport)</h2>", f"<pre class='mono'>{_e(json.dumps(keep, indent=1))}</pre>"]
    agents = [a for a in led._events("derived_agent", "transmorphed")
              if any(r.startswith(generation + "/") for r in _foundations(led, a["agent_id"]))]
    parts += ["<h2>10. Afgeleide agents en transmorfoses</h2>",
              _table([[_e(a["agent_id"]), _e(a["_event"]), _e(a.get("purpose", "")), _e(a.get("approved_by", ""))]
                      for a in agents] or [["-", "-", "nog geen agents op deze generatie gebouwd", "-"]],
                     ["agent", "soort", "doel", "goedgekeurd door"]),
              "<h2>11. Integriteit</h2>",
              _table(integ, ["toddler", "sha256 gewichten (nu gemeten)", "register + meta.json"]),
              f"<p>Hashketen van het evolutieregister: {'intact' if ok_chain else f'GEBROKEN bij rij {bad_seq}'}; "
              f"{len(led.log.entries)} rijen; laatste rijhash <code>{_e(led.log.entries[-1].row_hash)}</code>.</p>"]
    return "".join(parts)


def _foundations(led: L.Ledger, agent_id: str) -> list[str]:
    try:
        return list(led.trace_agent(agent_id)["foundations"])
    except KeyError:
        return []


def write_pdf(root: Path, generation: str, out: Path, report_json: Path | None = None) -> Path:
    from weasyprint import HTML

    out.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=build_html(root, generation, report_json)).write_pdf(out)
    return out
