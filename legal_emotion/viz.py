from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
import textwrap


def _safe_js_object_literal(payload: Dict[str, Any]) -> str:
    s = json.dumps(
        payload,
        ensure_ascii=False,
    )
    return s.replace(
        '</',
        '<\\/',
    )


def write_compare_html_report(
    *,
    output_path: str | Path,
    payload: Dict[str, Any],
) -> str:
    out_p = Path(output_path)
    out_p.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    data_js = _safe_js_object_literal(payload)
    html = textwrap.dedent(f"""\
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>Legal Emotion OT Report</title>
            <style>
              :root {{
                --bg: #0b1020;
                --panel: #ffffff;
                --muted: #64748b;
                --border: #e2e8f0;
                --ink: #0f172a;
                --blue: #4f46e5;
                --green: #10b981;
                --red: #ef4444;
                --heat: #2563eb;
              }}
        
              * {{ box-sizing: border-box; }}
              body {{
                margin: 0;
                font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
                color: var(--ink);
                background: #f8fafc;
              }}
        
              header {{
                padding: 12px 16px;
                background: var(--bg);
                color: #fff;
                border-bottom: 1px solid rgba(255, 255, 255, 0.12);
              }}
        
              header .row {{
                display: flex;
                align-items: baseline;
                justify-content: space-between;
                gap: 12px;
              }}
        
              header .title {{
                font-weight: 700;
                letter-spacing: 0.2px;
              }}
        
              header .subtitle {{
                font-weight: 500;
                color: rgba(255, 255, 255, 0.75);
                margin-left: 10px;
              }}
        
              header .stats {{
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                font-size: 12px;
                color: rgba(255, 255, 255, 0.75);
              }}
        
              main {{
                display: grid;
                grid-template-columns: 320px 1fr;
                min-height: calc(100vh - 54px);
              }}
        
              #sidebar {{
                background: var(--panel);
                border-right: 1px solid var(--border);
                padding: 12px;
                overflow: auto;
              }}
        
              #content {{
                padding: 16px;
                overflow: auto;
              }}
        
              .search {{
                width: 100%;
                padding: 10px 12px;
                border-radius: 10px;
                border: 1px solid var(--border);
                outline: none;
              }}
        
              .doc-item {{
                padding: 8px 10px;
                border-radius: 10px;
                cursor: pointer;
                border: 1px solid transparent;
                margin-top: 8px;
              }}
        
              .doc-item:hover {{
                background: #f1f5f9;
              }}
        
              .doc-item.active {{
                background: #eef2ff;
                border-color: #c7d2fe;
              }}
        
              .doc-item .small {{
                font-size: 12px;
                color: var(--muted);
                margin-top: 2px;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
              }}
        
              h2 {{
                margin: 0 0 6px 0;
                font-size: 20px;
              }}
        
              h3 {{
                margin: 18px 0 8px 0;
                font-size: 14px;
                color: #111827;
              }}
        
              .meta {{
                border: 1px solid var(--border);
                background: #fff;
                border-radius: 12px;
                padding: 10px 12px;
              }}
        
              .meta pre {{
                margin: 0;
                font-size: 12px;
                overflow: auto;
              }}
        
              .grid2 {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
              }}
        
              .panel {{
                border: 1px solid var(--border);
                background: #fff;
                border-radius: 12px;
                padding: 10px 12px;
              }}
        
              details.panel summary {{
                cursor: pointer;
                font-weight: 600;
                color: #111827;
              }}
        
              details.panel pre {{
                margin: 10px 0 0;
                font-size: 12px;
                line-height: 1.45;
                white-space: pre-wrap;
                max-height: 520px;
                overflow: auto;
              }}
        
              .bar-row {{
                display: grid;
                grid-template-columns: 110px 1fr 86px;
                gap: 8px;
                align-items: center;
                margin: 6px 0;
              }}
        
              .bar-row .label {{
                font-size: 12px;
                color: #111827;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
              }}
        
              .bar-row .bar {{
                height: 10px;
                background: #eef2ff;
                border-radius: 999px;
                overflow: hidden;
              }}
        
              .bar-row .bar > div {{
                height: 100%;
                background: var(--blue);
              }}
        
              .bar-row .value {{
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                font-size: 12px;
                color: var(--muted);
                text-align: right;
              }}
        
              table {{
                width: 100%;
                border-collapse: collapse;
                background: #fff;
                border: 1px solid var(--border);
                border-radius: 12px;
                overflow: hidden;
              }}
        
              th, td {{
                padding: 8px 10px;
                border-bottom: 1px solid var(--border);
                font-size: 12px;
                vertical-align: top;
              }}
        
              th {{
                background: #f8fafc;
                color: #111827;
                font-weight: 600;
              }}
        
              tr:last-child td {{
                border-bottom: none;
              }}
        
              tr.clickable:hover td {{
                background: #f8fafc;
                cursor: pointer;
              }}
        
              .pill {{
                display: inline-block;
                padding: 2px 8px;
                border-radius: 999px;
                background: #f1f5f9;
                color: #0f172a;
                font-size: 12px;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
              }}
        
              .muted {{
                color: var(--muted);
              }}
        
              .heatmap {{
                margin-top: 10px;
                overflow: auto;
                border: 1px solid var(--border);
                border-radius: 12px;
              }}
        
              .heatmap table {{
                border: none;
                border-radius: 0;
              }}
        
              .heatmap th, .heatmap td {{
                border-bottom: 1px solid var(--border);
                border-right: 1px solid var(--border);
                text-align: center;
                padding: 6px;
                font-size: 11px;
                white-space: nowrap;
              }}
        
              .heatmap th:last-child, .heatmap td:last-child {{
                border-right: none;
              }}
        
              .heatcell {{
                min-width: 36px;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
              }}
            </style>
          </head>
          <body>
            <header>
              <div class="row">
                <div>
                  <span class="title">Legal Emotion OT Report</span>
                  <span class="subtitle" id="subtitle"></span>
                </div>
                <div class="stats" id="stats"></div>
              </div>
            </header>
            <main>
              <aside id="sidebar">
                <input id="search" class="search" placeholder="Filter docs (index / id / path)..." />
                <div id="docList"></div>
              </aside>
              <section id="content">
                <h2 id="docTitle"></h2>
                <div class="meta">
                  <pre id="docMeta"></pre>
                </div>
        
                <details id="textBlock" class="panel" style="margin-top: 12px;">
                  <summary>Document text</summary>
                  <pre id="docText" class="muted">(text not embedded)</pre>
                </details>
        
                <div class="grid2" style="margin-top: 12px;">
                  <div class="panel">
                    <div class="muted" id="primaryLabel"></div>
                    <div id="primaryBars"></div>
                  </div>
                  <div class="panel">
                    <div class="muted" id="styleLabel"></div>
                    <div id="styleBars"></div>
                  </div>
                </div>
        
                <h3>Neighbours</h3>
                <div id="neighboursWrap"></div>
        
                <h3>Pair Explanation</h3>
                <div id="pairWrap" class="panel muted">Select a neighbour row to view the OT breakdown.</div>
              </section>
            </main>
        
            <script>
              const DATA = {data_js};
        
              function safeJson(x) {{
                try {{
                  return JSON.stringify(x, null, 2);
                }} catch (e) {{
                  return String(x);
                }}
              }}
        
              function formatNum(x) {{
                if (x === null || x === undefined) return "";
                if (typeof x !== "number") return String(x);
                if (!Number.isFinite(x)) return String(x);
                const abs = Math.abs(x);
                if (abs !== 0 && (abs < 1e-3 || abs >= 1e4)) return x.toExponential(3);
                return x.toFixed(6).replace(/0+$/, "").replace(/\\.$/, "");
              }}
        
              function docLabel(doc) {{
                const meta = doc.meta || {{}};
                if (meta.id) return `${{doc.index}}: ${{meta.id}}`;
                if (meta.path) {{
                  const p = String(meta.path);
                  const base = p.split(/[\\\\/]/).pop();
                  return `${{doc.index}}: ${{base}}`;
                }}
                return String(doc.index);
              }}
        
              function docSub(doc) {{
                const meta = doc.meta || {{}};
                if (meta.path) return String(meta.path);
                const keys = Object.keys(meta);
                if (keys.length) return keys.slice(0, 3).map(k => `${{k}}=${{meta[k]}}`).join(" ");
                return "";
              }}
        
              function renderBars(container, labels, values, opts) {{
                container.innerHTML = "";
                const diverging = !!(opts && opts.diverging);
                const absMax = Math.max(1e-12, ...values.map(v => Math.abs(Number(v) || 0)));
                for (let i = 0; i < labels.length; i++) {{
                  const v = Number(values[i]) || 0;
                  const row = document.createElement("div");
                  row.className = "bar-row";
        
                  const label = document.createElement("div");
                  label.className = "label";
                  label.textContent = labels[i];
        
                  const bar = document.createElement("div");
                  bar.className = "bar";
        
                  const inner = document.createElement("div");
                  inner.style.width = `${{Math.min(100, (Math.abs(v) / absMax) * 100)}}%`;
                  if (diverging) {{
                    inner.style.background = v >= 0 ? "var(--green)" : "var(--red)";
                    bar.style.background = "#f1f5f9";
                  }} else {{
                    inner.style.background = "var(--blue)";
                  }}
                  bar.appendChild(inner);
        
                  const val = document.createElement("div");
                  val.className = "value";
                  val.textContent = formatNum(v);
        
                  row.appendChild(label);
                  row.appendChild(bar);
                  row.appendChild(val);
                  container.appendChild(row);
                }}
              }}
        
              function emoLabel(e) {{
                if (DATA.emotion_labels && typeof DATA.emotion_labels === "object" && DATA.emotion_labels[e]) {{
                  return String(DATA.emotion_labels[e]);
                }}
                return String(e);
              }}
        
              function renderExplain(explain, emotions, emotionLabels) {{
                const wrap = document.createElement("div");
                if (!explain) {{
                  wrap.className = "muted";
                  wrap.textContent = "No explanation data available.";
                  return wrap;
                }}
                if (explain.error) {{
                  wrap.className = "muted";
                  wrap.textContent = explain.error;
                  return wrap;
                }}
        
                const meta = document.createElement("div");
                meta.innerHTML = `
                  <div style="display:flex; gap: 10px; flex-wrap: wrap; align-items: center;">
                    <span class="pill">${{explain.mode}}</span>
                    <span class="pill">${{explain.measure}}</span>
                    <span class="pill">dist=${{formatNum(explain.distance)}}</span>
                    <span class="pill">Delta=${{formatNum(explain.distance_minus_calc)}}</span>
                  </div>
                  <div style="margin-top: 8px;" class="muted">
                    transport_ab=${{formatNum(explain.transport_ab)}} &nbsp; kl_a=${{formatNum(explain.kl_a)}} &nbsp; kl_b=${{formatNum(explain.kl_b)}} &nbsp;
                    mass_a=${{formatNum(explain.mass_a)}} &nbsp; mass_b=${{formatNum(explain.mass_b)}} &nbsp; mass_plan=${{formatNum(explain.mass_plan)}}
                  </div>
                `;
                wrap.appendChild(meta);
        
                const flows = Array.isArray(explain.top_transport_cost_contrib) ? explain.top_transport_cost_contrib : [];
        
                if (flows.length) {{
                  const table = document.createElement("table");
                  table.style.marginTop = "10px";
                  table.innerHTML = `
                    <thead>
                      <tr>
                        <th>from</th>
                        <th>to</th>
                        <th>mass</th>
                        <th>cost</th>
                        <th>contribution</th>
                      </tr>
                    </thead>
                    <tbody></tbody>
                  `;
                  const tbody = table.querySelector("tbody");
                  for (const f of flows) {{
                    const tr = document.createElement("tr");
                    tr.innerHTML = `
                      <td>${{emoLabel(f.from)}}</td>
                      <td>${{emoLabel(f.to)}}</td>
                      <td class="muted">${{formatNum(f.mass)}}</td>
                      <td class="muted">${{formatNum(f.cost)}}</td>
                      <td class="muted">${{formatNum(f.contribution)}}</td>
                    `;
                    tbody.appendChild(tr);
                  }}
                  wrap.appendChild(table);
        
                  const idx = new Map(emotions.map((e, i) => [e, i]));
                  const n = emotions.length;
                  const mat = Array.from({{length: n}}, () => Array(n).fill(0));
                  let maxV = 0;
                  for (const f of flows) {{
                    const i = idx.get(f.from);
                    const j = idx.get(f.to);
                    if (i === undefined || j === undefined) continue;
                    const v = Number(f.contribution) || 0;
                    mat[i][j] += v;
                    maxV = Math.max(maxV, Math.abs(v));
                  }}
        
                  const heat = document.createElement("div");
                  heat.className = "heatmap";
                  const t = document.createElement("table");
                  const thead = document.createElement("thead");
                  const hr = document.createElement("tr");
                  hr.appendChild(document.createElement("th"));
                  for (const e of emotionLabels) {{
                    const th = document.createElement("th");
                    th.textContent = e;
                    hr.appendChild(th);
                  }}
                  thead.appendChild(hr);
                  t.appendChild(thead);
        
                  const tb = document.createElement("tbody");
                  for (let i = 0; i < n; i++) {{
                    const tr = document.createElement("tr");
                    const th = document.createElement("th");
                    th.textContent = emotionLabels[i] || emotions[i];
                    tr.appendChild(th);
                    for (let j = 0; j < n; j++) {{
                      const v = mat[i][j];
                      const cell = document.createElement("td");
                      cell.className = "heatcell";
                      const a = maxV > 0 ? Math.min(1, Math.abs(v) / maxV) : 0;
                      const alpha = 0.10 + 0.60 * a;
                      cell.style.background = v > 0 ? `rgba(37, 99, 235, ${{alpha}})` : "#fff";
                      cell.textContent = v > 0 ? formatNum(v) : "";
                      tr.appendChild(cell);
                    }}
                    tb.appendChild(tr);
                  }}
                  t.appendChild(tb);
                  heat.appendChild(t);
                  wrap.appendChild(heat);
                }}
                return wrap;
              }}
        
              const docs = Array.isArray(DATA.docs) ? DATA.docs : [];
              const emotions = Array.isArray(DATA.emotions) ? DATA.emotions : [];
              const emotionLabels = emotions.map(e => emoLabel(e));
              const docsByIndex = new Map(docs.map(d => [d.index, d]));
              const neighboursByIndex = new Map((DATA.neighbours || []).map(r => [r.index, r.neighbours || []]));
        
              function setDoc(index) {{
                const doc = docsByIndex.get(index);
                if (!doc) return;
                document.getElementById("docTitle").textContent = docLabel(doc);
                document.getElementById("docMeta").textContent = safeJson(doc.meta || {{}});
        
                const textBlock = document.getElementById("textBlock");
                const textEl = document.getElementById("docText");
                if (textBlock && textEl) {{
                  if (typeof doc.text === "string" && doc.text.length) {{
                    textEl.textContent = doc.text;
                    textEl.className = "";
                  }} else {{
                    textEl.textContent = "(text not embedded in this report)";
                    textEl.className = "muted";
                  }}
                }}
        
                const primaryVec = Array.isArray(doc.vector) ? doc.vector : [];
                const styleVec = Array.isArray(doc.style_vector) ? doc.style_vector : [];
        
                const primaryLabel = document.getElementById("primaryLabel");
                primaryLabel.textContent = `Primary: mode=${{DATA.mode}} measure=${{DATA.measure}}  mass=${{formatNum(doc.mass)}}`;
                const styleLabel = document.getElementById("styleLabel");
                if (DATA.style && DATA.style.measure) {{
                  styleLabel.textContent = `Style: mode=${{DATA.style.mode}} measure=${{DATA.style.measure}}`;
                }} else {{
                  styleLabel.textContent = "Style: (disabled)";
                }}
        
                renderBars(document.getElementById("primaryBars"), emotionLabels, primaryVec, {{diverging:false}});
                if (DATA.style && styleVec.length === emotions.length) {{
                  renderBars(document.getElementById("styleBars"), emotionLabels, styleVec, {{diverging:false}});
                }} else {{
                  document.getElementById("styleBars").innerHTML = "<div class='muted'>No style vector available.</div>";
                }}
        
                const neigh = neighboursByIndex.get(index) || [];
                renderNeighbours(index, neigh);
                document.getElementById("pairWrap").textContent = "Select a neighbour row to view the OT breakdown.";
                document.getElementById("pairWrap").className = "panel muted";
              }}
        
              function renderNeighbours(i, neigh) {{
                const wrap = document.getElementById("neighboursWrap");
                if (!neigh.length) {{
                  wrap.innerHTML = "<div class='panel muted'>No neighbours (need at least 2 docs).</div>";
                  return;
                }}
                const table = document.createElement("table");
                table.innerHTML = `
                  <thead>
                    <tr>
                      <th>neighbour</th>
                      <th>distance</th>
                      <th>style</th>
                      <th>mass_diff</th>
                      <th>saturation_diff</th>
                    </tr>
                  </thead>
                  <tbody></tbody>
                `;
                const tbody = table.querySelector("tbody");
                for (const n of neigh) {{
                  const j = n.index;
                  const tr = document.createElement("tr");
                  tr.className = "clickable";
                  const sat = n.saturation_diff_per_1k_words !== null && n.saturation_diff_per_1k_words !== undefined
                    ? formatNum(n.saturation_diff_per_1k_words)
                    : (n.saturation_diff_per_1k_tokens !== null && n.saturation_diff_per_1k_tokens !== undefined ? formatNum(n.saturation_diff_per_1k_tokens) : "");
                  tr.innerHTML = `
                    <td><span class="pill">${{j}}</span></td>
                    <td class="muted">${{formatNum(n.distance)}}</td>
                    <td class="muted">${{formatNum(n.style_distance)}}</td>
                    <td class="muted">${{formatNum(n.mass_diff)}}</td>
                    <td class="muted">${{sat}}</td>
                  `;
                  tr.addEventListener("click", () => renderPair(i, n));
                  tbody.appendChild(tr);
                }}
                wrap.innerHTML = "";
                wrap.appendChild(table);
              }}
        
              function renderPair(i, neighbour) {{
                const j = neighbour.index;
                const a = docsByIndex.get(i);
                const b = docsByIndex.get(j);
                const wrap = document.getElementById("pairWrap");
                wrap.className = "panel";
                wrap.innerHTML = "";
        
                const head = document.createElement("div");
                head.innerHTML = `
                  <div style="display:flex; gap: 10px; flex-wrap: wrap; align-items: center;">
                    <span class="pill">pair ${{i}} -> ${{j}}</span>
                    <span class="pill">dist=${{formatNum(neighbour.distance)}}</span>
                    <span class="pill">style=${{formatNum(neighbour.style_distance)}}</span>
                    <span class="pill">mass_diff=${{formatNum(neighbour.mass_diff)}}</span>
                  </div>
                `;
                wrap.appendChild(head);
        
                if (a && b && DATA.style && Array.isArray(a.style_vector) && Array.isArray(b.style_vector) && a.style_vector.length === emotions.length) {{
                  const delta = b.style_vector.map((v, k) => (Number(v) || 0) - (Number(a.style_vector[k]) || 0));
                  const sec = document.createElement("div");
                  sec.style.marginTop = "10px";
                  sec.innerHTML = "<div class='muted'>Style delta (neighbour - doc)</div>";
                  const bars = document.createElement("div");
                  renderBars(bars, emotionLabels, delta, {{diverging:true}});
                  sec.appendChild(bars);
                  wrap.appendChild(sec);
                }}
        
                const grid = document.createElement("div");
                grid.className = "grid2";
                grid.style.marginTop = "12px";
        
                const p = document.createElement("div");
                p.className = "panel";
                p.innerHTML = "<div class='muted'>Primary OT explanation</div>";
                p.appendChild(renderExplain(neighbour.primary_explain, emotions, emotionLabels));
        
                const s = document.createElement("div");
                s.className = "panel";
                s.innerHTML = "<div class='muted'>Style OT explanation</div>";
                s.appendChild(renderExplain(neighbour.style_explain, emotions, emotionLabels));
        
                grid.appendChild(p);
                grid.appendChild(s);
                wrap.appendChild(grid);
              }}
        
              function renderDocList(filterText) {{
                const list = document.getElementById("docList");
                list.innerHTML = "";
                const q = (filterText || "").trim().toLowerCase();
                for (const doc of docs) {{
                  const label = docLabel(doc);
                  const sub = docSub(doc);
                  const blob = `${{doc.index}} ${{label}} ${{sub}}`.toLowerCase();
                  if (q && !blob.includes(q)) continue;
                  const item = document.createElement("div");
                  item.className = "doc-item";
                  item.dataset.index = String(doc.index);
                  item.innerHTML = `<div>${{label}}</div><div class="small">${{sub}}</div>`;
                  item.addEventListener("click", () => {{
                    document.querySelectorAll(".doc-item.active").forEach(x => x.classList.remove("active"));
                    item.classList.add("active");
                    setDoc(doc.index);
                  }});
                  list.appendChild(item);
                }}
              }}
        
              document.getElementById("subtitle").textContent =
                `mode=${{DATA.mode}}  cost=${{DATA.cost}}  measure=${{DATA.measure}}`;
              document.getElementById("stats").textContent =
                `${{docs.length}} docs * topk=${{DATA.topk}} * explain=${{DATA.include_explain ? "on" : "off"}}`;
        
              const search = document.getElementById("search");
              search.addEventListener("input", () => renderDocList(search.value));
              renderDocList("");
              if (docs.length) {{
                const first = document.querySelector(".doc-item");
                if (first) first.click();
              }}
            </script>
          </body>
        </html>
        """)
    out_p.write_text(
        html,
        encoding='utf-8',
    )
    return str(out_p)
