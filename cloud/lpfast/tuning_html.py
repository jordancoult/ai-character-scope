"""TUNING.md -> TUNING.html (styled, light/dark). Run after editing TUNING.md:  python3 cloud/lpfast/tuning_html.py"""
import html
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1f;--mut:#6e6e73;--line:#e3e3e0;--acc:#c2185b;--code:#f1f0ee}
@media(prefers-color-scheme:dark){:root{--bg:#161617;--fg:#ececec;--mut:#a1a1a6;--line:#2e2e30;--acc:#ff6fa8;--code:#232325}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif}
main{max-width:1000px;margin:0 auto;padding:40px 24px 80px}
h1{font-size:30px;margin:0 0 6px;letter-spacing:-.02em}h1+p{color:var(--mut);margin-top:4px}
h2{font-size:19px;margin:38px 0 10px;padding-top:14px;border-top:1px solid var(--line);color:var(--acc)}
p{margin:8px 0;max-width:80ch}
table{width:100%;border-collapse:collapse;margin:12px 0 4px;font-size:14px}
th{text-align:left;font-weight:600;color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.04em;border-bottom:2px solid var(--line);padding:8px 10px}
td{border-bottom:1px solid var(--line);padding:8px 10px;vertical-align:top}
tr:hover td{background:color-mix(in srgb,var(--acc) 5%,transparent)}
code{font:13px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--code);padding:1px 6px;border-radius:5px;white-space:nowrap}
td:first-child code{color:var(--acc);font-weight:600}
td{font-variant-numeric:tabular-nums}
"""


def inline(t):
    parts = re.split(r"(`(?:[^`\\]|\\.)*`)", t)
    out = []
    for p in parts:
        if p.startswith("`") and p.endswith("`") and len(p) > 1:
            out.append("<code>" + html.escape(p[1:-1].replace("\\|", "|")) + "</code>")
        else:
            p = html.escape(p.replace("\\|", "|"))
            p = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", p)
            out.append(re.sub(r"\*(.+?)\*", r"<em>\1</em>", p))
    return "".join(out)


def cells(row):
    row = row.strip().strip("|")
    return [c.strip() for c in re.split(r"(?<!\\)\|", row)]


def convert(md):
    lines, out, i, para = md.splitlines(), [], 0, []

    def flush():
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para.clear()
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("#"):
            flush()
            n = len(ln) - len(ln.lstrip("#"))
            out.append(f"<h{n}>{inline(ln[n:].strip())}</h{n}>")
        elif ln.startswith("|"):
            flush()
            head = cells(ln)
            i += 2
            body = []
            while i < len(lines) and lines[i].startswith("|"):
                body.append(cells(lines[i]))
                i += 1
            out.append("<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead><tbody>"
                       + "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body)
                       + "</tbody></table>")
            continue
        elif not ln.strip():
            flush()
        else:
            para.append(ln.strip())
        i += 1
    flush()
    return "".join(out)


def main():
    md = (HERE / "TUNING.md").read_text()
    title = md.splitlines()[0].lstrip("# ").strip()
    (HERE / "TUNING.html").write_text(
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
        f"content='width=device-width,initial-scale=1'><title>{title}</title><style>{CSS}</style></head>"
        f"<body><main>{convert(md)}</main></body></html>\n")
    print("wrote", HERE / "TUNING.html")


if __name__ == "__main__":
    main()
