from __future__ import annotations
import html
import json
from pathlib import Path
from typing import Iterable, Tuple
import textwrap


def _read_section(path: Path) -> Tuple[str, str]:
    text = path.read_text(
        encoding='utf-8',
        errors='ignore',
    ).strip()
    kind = 'text'
    if path.suffix.lower() == '.json':
        try:
            payload = json.loads(text)
            text = json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            )
            kind = 'json'
        except Exception:
            kind = 'text'
    return (text, kind)


def write_summary_html_report(
    *,
    output_path: str | Path,
    sections: Iterable[Tuple[str, str | Path]],
    title: str = 'Legal Emotion OT Summary',
) -> str:
    out_p = Path(output_path)
    out_p.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    blocks = []
    for label, path in sections:
        if not label:
            continue
        p = Path(path)
        if not p.exists():
            continue
        (
            body,
            kind,
        ) = _read_section(p)
        blocks.append({
                'label': str(label),
                'path': str(p),
                'body': body,
                'kind': kind,
            })
    if not blocks:
        raise ValueError('error: ValueError')
    sections_html = '\n'.join((
            f"\n        <section class=\"card\">\n          <div class=\"card-header\">\n            <div class=\"card-title\">{html.escape(block['label'])}</div>\n            <div class=\"card-path\">{html.escape(block['path'])}</div>\n          </div>\n          <pre class=\"card-body {block['kind']}\">{html.escape(block['body'])}</pre>\n        </section>\n        "
            for block in blocks
        ))
    doc = textwrap.dedent(f"""\
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>{html.escape(title)}</title>
            <style>
              :root {{
                --ink: #14121b;
                --muted: #52606d;
                --accent: #e07a5f;
                --bg-top: #fdf6ec;
                --bg-bottom: #eff2f6;
                --card: #ffffff;
                --border: rgba(15, 23, 42, 0.08);
              }}
              * {{
                box-sizing: border-box;
              }}
              body {{
                margin: 0;
                font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
                color: var(--ink);
                background: linear-gradient(180deg, var(--bg-top), var(--bg-bottom));
              }}
              header {{
                padding: 20px 24px 16px;
                background: radial-gradient(circle at top left, #ffe8d1 0%, #f0f4f8 60%);
                border-bottom: 1px solid var(--border);
              }}
              header h1 {{
                margin: 0;
                font-family: ui-serif, Georgia, Cambria, "Times New Roman", Times, serif;
                font-size: 28px;
                letter-spacing: 0.3px;
              }}
              header p {{
                margin: 6px 0 0;
                color: var(--muted);
                font-size: 14px;
              }}
              main {{
                padding: 18px 24px 30px;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 16px;
              }}
              .card {{
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 16px;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
                overflow: hidden;
                display: flex;
                flex-direction: column;
                min-height: 220px;
              }}
              .card-header {{
                padding: 14px 16px;
                border-bottom: 1px dashed var(--border);
                background: rgba(224, 122, 95, 0.08);
              }}
              .card-title {{
                font-weight: 700;
                font-size: 16px;
              }}
              .card-path {{
                margin-top: 4px;
                font-size: 12px;
                color: var(--muted);
              }}
              .card-body {{
                padding: 12px 16px 16px;
                margin: 0;
                font-size: 12px;
                line-height: 1.5;
                white-space: pre-wrap;
                overflow: auto;
                max-height: 360px;
              }}
              .card-body.json {{
                color: #1f2937;
              }}
              @media (max-width: 720px) {{
                header {{
                  padding: 16px 18px 14px;
                }}
                main {{
                  padding: 16px 18px 24px;
                }}
              }}
            </style>
          </head>
          <body>
            <header>
              <h1>{html.escape(title)}</h1>
            </header>
            <main>
              {sections_html}
            </main>
          </body>
        </html>
        """)
    out_p.write_text(
        doc,
        encoding='utf-8',
    )
    return str(out_p)
