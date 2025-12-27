from __future__ import annotations
import base64
from pathlib import Path
from typing import Iterable, Tuple
import textwrap


def _html_to_data_url(html: str) -> str:
    payload = base64.b64encode(html.encode('utf-8')).decode('ascii')
    return f'data:text/html;base64,{payload}'


def write_combined_html_report(
    *,
    output_path: str | Path,
    sections: Iterable[Tuple[str, str | Path]],
    title: str = 'Legal Emotion OT Report',
) -> str:
    out_p = Path(output_path)
    out_p.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    entries = []
    for label, html_path in sections:
        if not label:
            continue
        p = Path(html_path)
        if not p.exists():
            raise FileNotFoundError('error: FileNotFoundError')
        html = p.read_text(
            encoding='utf-8',
            errors='ignore',
        )
        entries.append((str(label), _html_to_data_url(html)))
    if not entries:
        raise ValueError('error: ValueError')
    labels = [label for label, _ in entries]
    has_doc_ot = any((
            'doc-level ot' in label.lower()
            for label in labels
        ))
    has_token_ot = any(('token-ot' in label.lower() for label in labels))
    has_group_shift = any(('group shift' in label.lower() for label in labels))
    has_summary = any((
            label.lower() == 'summary'
            or 'summary' in label.lower()
            for label in labels
        ))
    has_most_emotional = any((
            'most emotional' in label.lower()
            for label in labels
        ))
    tab_buttons = '\n'.join((
            f'<button class="tab" data-tab="tab-{i}">{label}</button>'
            for i, (label, _) in enumerate(entries)
        ))
    tab_frames = '\n'.join((
            f'<div class="panel" id="tab-{i}"><iframe src="{url}" loading="lazy"></iframe></div>'
            for i, (_, url) in enumerate(entries)
        ))
    guide_bits = []
    if has_doc_ot and has_token_ot:
        guide_bits.append('<p>Use the tabs to move between the <b>doc-level OT</b> view (primary) and the <b>token-level OT</b> views (experimental). Doc-level OT compares predicted emotion distributions; token-level OT is interpretable but can be noisy on legal text.</p>')
    if has_doc_ot:
        guide_bits.append('\n      <h3>Doc-level OT Neighbours tab</h3>\n      <ul>\n        <li>Distance compares the emotion mix predicted for each document. Lower means a more similar mix.</li>\n        <li>Mass and mass_diff show intensity. Low distance with high mass_diff means similar mix but different intensity.</li>\n        <li>High distance means different emotion profiles regardless of intensity. (this happens regardless of intensity).</li>\n      </ul>\n            '.rstrip())
    if has_token_ot:
        guide_bits.append('\n      <h3>Token-OT Neighbours tab (experimental)</h3>\n      <ol>\n        <li>Pick a document from the left list or use search.</li>\n        <li>Read Top terms to see the emotional language extracted from that document.</li>\n        <li>Check selected_ratio and n_selected_tokens to judge coverage. Very low values mean the cloud is sparse.</li>\n        <li>Use Nearest neighbours to compare documents. Lower distance means more similar extracted emotional language.</li>\n        <li>Select a neighbour to open Pair explanation. Top transport flows show which terms align and which terms differ.</li>\n        <li>If top terms or flows look procedural/boilerplate, expand stopwords or raise vad_threshold and rerun.</li>\n      </ol>\n            '.rstrip())
    if has_group_shift:
        guide_bits.append('\n      <h3>Group Shift tab (token-OT, experimental)</h3>\n      <ul>\n        <li>Compares two document groups. Flows show which emotional terms shift between groups.</li>\n      </ul>\n            '.rstrip())
    guide_html = '\n'.join(guide_bits).strip()
    html = textwrap.dedent(f"""\
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>{title}</title>
            <style>
              :root {{
                --bg-deep: #1b1f2a;
                --panel: #ffffff;
                --muted: #5b6b7a;
                --border: rgba(15, 23, 42, 0.08);
                --ink: #111827;
                --accent: #e07a5f;
                --accent-dark: #c9654a;
                --sheet: #f4f1ed;
              }}
              * {{ box-sizing: border-box; }}
              body {{
                margin: 0;
                font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
                color: var(--ink);
                background: linear-gradient(160deg, #fdf6ec 0%, #eff2f6 70%);
              }}
              header {{
                background: radial-gradient(circle at top left, #ffe8d1 0%, #f0f4f8 60%);
                color: #1f2937;
                padding: 16px 20px 14px;
                border-bottom: 1px solid var(--border);
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
                animation: fadeIn 0.6s ease;
              }}
              header .title {{
                font-family: ui-serif, Georgia, Cambria, "Times New Roman", Times, serif;
                font-weight: 700;
                font-size: 20px;
                letter-spacing: 0.3px;
              }}
              header .hint {{
                font-size: 12px;
                color: var(--muted);
              }}
              .guide {{
                margin: 14px 18px 0;
                padding: 12px 16px;
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid var(--border);
                border-radius: 14px;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
                font-size: 13px;
                color: var(--ink);
              }}
              .guide h2 {{
                margin: 0 0 6px;
                font-size: 14px;
                font-weight: 700;
              }}
              .guide ul {{
                margin: 0;
                padding-left: 18px;
              }}
              .tabs {{
                display: flex;
                gap: 8px;
                padding: 10px 14px;
                border-bottom: 1px solid var(--border);
                background: rgba(255, 255, 255, 0.85);
                backdrop-filter: blur(6px);
                position: sticky;
                top: 0;
                z-index: 2;
                animation: slideUp 0.5s ease;
              }}
              .tab {{
                border: 1px solid var(--border);
                background: #fff7ed;
                color: #4b5563;
                padding: 8px 14px;
                border-radius: 999px;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.2s ease;
              }}
              .tab.active {{
                background: var(--accent);
                color: #fff;
                border-color: var(--accent-dark);
                box-shadow: 0 8px 18px rgba(224, 122, 95, 0.25);
              }}
              .panel {{
                display: none;
                height: calc(100vh - 106px);
                animation: fadeIn 0.25s ease;
              }}
              .panel.active {{
                display: block;
              }}
              iframe {{
                width: 100%;
                height: 100%;
                border: 0;
                background: #fff;
              }}
              @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(6px); }}
                to {{ opacity: 1; transform: translateY(0); }}
              }}
              @keyframes slideUp {{
                from {{ opacity: 0; transform: translateY(8px); }}
                to {{ opacity: 1; transform: translateY(0); }}
              }}
            </style>
          </head>
          <body>
            <header>
              <div class="title">{title}</div>
            </header>
            <section class="guide">
              <h2>How to read this report</h2>
        {guide_html}
            </section>
            <div class="tabs">
              {tab_buttons}
            </div>
            {tab_frames}
            <script>
              const tabs = Array.from(document.querySelectorAll('.tab'));
              const panels = Array.from(document.querySelectorAll('.panel'));
              function setActive(idx) {{
                tabs.forEach((t, i) => t.classList.toggle('active', i === idx));
                panels.forEach((p, i) => p.classList.toggle('active', i === idx));
              }}
              tabs.forEach((t, i) => t.addEventListener('click', () => setActive(i)));
              setActive(0);
            </script>
          </body>
        </html>
        """)
    out_p.write_text(
        html,
        encoding='utf-8',
    )
    return str(out_p)
