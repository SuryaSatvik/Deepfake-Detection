"""
Assembles notebooks/model analysis.ipynb from the cell source file
scripts/_nb_cells.txt.

The cell bodies contain their own triple-quoted strings, so the file cannot be
imported as Python. It is segmented textually instead: a cell starts at a line
that is exactly `md(r\"\"\"` or `co(r\"\"\"`, and ends at the LAST line equal to
`\"\"\")` before the next cell start (or before `dump()`).
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "scripts" / "_nb_cells.txt").read_text().splitlines()

START = re.compile(r'^(md|co)\(r"""$')
starts = [i for i, l in enumerate(src) if START.match(l)]
end_of_all = next((i for i, l in enumerate(src) if l.strip() == "dump()"), len(src))
bounds = starts + [end_of_all]

cells = []
for k, i in enumerate(starts):
    kind = "markdown" if START.match(src[i]).group(1) == "md" else "code"
    seg = src[i + 1: bounds[k + 1]]
    close = max(j for j, l in enumerate(seg) if l.rstrip() == '""")')
    body = "\n".join(seg[:close]).strip("\n")
    cells.append((kind, body))

out_cells = []
for kind, body in cells:
    c = {"cell_type": kind, "metadata": {},
         "source": body.splitlines(keepends=True)}
    if kind == "code":
        c["execution_count"] = None
        c["outputs"] = []
    out_cells.append(c)

nb = {"cells": out_cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}

out = ROOT / "notebooks" / "model analysis.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out}  ({len(out_cells)} cells: "
      f"{sum(1 for k,_ in cells if k=='markdown')} md / "
      f"{sum(1 for k,_ in cells if k=='code')} code)")

# every code cell must at least parse
import ast
bad = 0
for n, (kind, body) in enumerate(cells):
    if kind != "code":
        continue
    try:
        ast.parse(body)
    except SyntaxError as e:
        bad += 1
        print(f"  SYNTAX ERROR in code cell {n}: line {e.lineno}: {e.msg}")
        print("   >", (body.splitlines()[e.lineno-1] if e.lineno and e.lineno <= len(body.splitlines()) else ""))
print("all code cells parse" if not bad else f"{bad} cell(s) failed to parse")
sys.exit(1 if bad else 0)
