from __future__ import annotations
import html
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(
        'r',
        encoding='utf-8',
        errors='ignore',
    ) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(
                obj,
                dict,
            ):
                yield obj


def _as_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return float(v)


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    xs = sorted((float(v) for v in values))
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return float(xs[mid])
    return float(0.5 * (xs[mid - 1] + xs[mid]))


def _pick_vector(
    row: Dict[str, Any],
    key: str,
    n: int,
) -> Optional[List[float]]:
    v = row.get(key)
    if not isinstance(
        v,
        list,
    ) or len(v) != n:
        return None
    out: List[float] = []
    for item in v:
        fv = _as_float(item)
        out.append(0.0 if fv is None else float(fv))
    return out


@dataclass
class _Bucket:
    totals: List[float]
    emos: List[List[float]]


def _aggregate_scores_by_year(scores_jsonl: Path) -> Tuple[
    List[str],
    List[str],
    Dict[str, Dict[str, Dict[str, Any]]],
    Dict[str, Any],
]:
    first: Optional[Dict[str, Any]] = None
    buckets: Dict[str, Dict[str, _Bucket]] = {}
    doc_types: set[str] = set()
    total_rows = 0
    used_rows = 0
    for row in _iter_jsonl(scores_jsonl):
        total_rows += 1
        if first is None:
            first = row
        meta = (
            row.get('meta')
            if isinstance(
                row.get('meta'),
                dict,
            )
            else {}
        )
        date = meta.get('date')
        if not isinstance(
            date,
            str,
        ) or len(date) < 4:
            continue
        year = date[:4]
        if not year.isdigit():
            continue
        doc_type = (
            meta.get('doc_type')
            if isinstance(
                meta.get('doc_type'),
                str,
            )
            else 'UNK'
        )
        doc_type = str(doc_type).strip() or 'UNK'
        doc_types.add(doc_type)
        emotions = (
            first.get('emotions')
            if isinstance(
                first.get('emotions'),
                list,
            )
            else []
        )
        if not emotions:
            continue
        emotions = [str(e) for e in emotions]
        n = len(emotions)
        vec = _pick_vector(
            row,
            'pred_mixscaled_per_1k_words',
            n,
        )
        if vec is None:
            continue
        total = _as_float(row.get('emotion_signal_per_1k_words'))
        if total is None:
            total = _as_float(row.get('pred_per_1k_words'))
        if total is None:
            total = float(sum(vec))
        by_type = buckets.setdefault(
            year,
            {},
        )
        b = by_type.get(doc_type)
        if b is None:
            b = _Bucket(
                totals=[],
                emos=[[] for _ in range(n)],
            )
            by_type[doc_type] = b
        b.totals.append(float(total))
        for i, v in enumerate(vec):
            b.emos[i].append(float(v))
        used_rows += 1
    if first is None:
        raise ValueError('error: ValueError')
    emotions = first.get('emotions')
    if not isinstance(
        emotions,
        list,
    ) or not emotions:
        raise ValueError('error: ValueError')
    emotions = [str(e) for e in emotions]
    n = len(emotions)
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    years_sorted = sorted(buckets.keys())
    doc_types_sorted = sorted(doc_types)
    for year in years_sorted:
        out[year] = {}
        for doc_type in doc_types_sorted:
            b = buckets.get(
                year,
                {},
            ).get(doc_type)
            if b is None or not b.totals:
                continue
            totals = b.totals
            out[year][doc_type] = {
                'n': int(len(totals)),
                'mean_total': float(sum(totals) / float(len(totals))),
                'median_total': float(_median(totals)),
                'mean_emotions': [
                    (
                        float(sum(b.emos[i])
                            / float(len(b.emos[i])))
                        if b.emos[i]
                        else 0.0
                    )
                    for i in range(n)
                ],
                'median_emotions': [
                    (
                        float(_median(b.emos[i]))
                        if b.emos[i]
                        else 0.0
                    )
                    for i in range(n)
                ],
            }
        all_totals: List[float] = []
        all_emos: List[List[float]] = [[] for _ in range(n)]
        for doc_type, b in buckets.get(
            year,
            {},
        ).items():
            all_totals.extend(b.totals)
            for i in range(n):
                all_emos[i].extend(b.emos[i])
        if all_totals:
            out[year]['ALL'] = {
                'n': int(len(all_totals)),
                'mean_total': float(sum(all_totals) / float(len(all_totals))),
                'median_total': float(_median(all_totals)),
                'mean_emotions': [
                    (
                        float(sum(all_emos[i])
                            / float(len(all_emos[i])))
                        if all_emos[i]
                        else 0.0
                    )
                    for i in range(n)
                ],
                'median_emotions': [
                    (
                        float(_median(all_emos[i]))
                        if all_emos[i]
                        else 0.0
                    )
                    for i in range(n)
                ],
            }
    meta = {
        'scores_jsonl': str(scores_jsonl),
        'rows_total': int(total_rows),
        'rows_used': int(used_rows),
        'years': years_sorted,
        'doc_types': doc_types_sorted,
    }
    return (emotions, doc_types_sorted, out, meta)


def write_emotion_timeline_html_report(
    *,
    output_path: str | Path,
    series: Sequence[Tuple[str, str | Path]],
    title: str = 'ICJ Emotionality Timeline',
    emotion_labels: Dict[str, str] | None = None,
) -> str:
    out_p = Path(output_path)
    out_p.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    if not series:
        raise ValueError('error: ValueError')
    datasets: List[Dict[str, Any]] = []
    feelings_ref: Optional[List[str]] = None
    doc_types_union: set[str] = set()
    years_union: set[str] = set()
    for label, path in series:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError('error: FileNotFoundError')
        (
            emotions,
            doc_types,
            by_year,
            meta,
        ) = (
            _aggregate_scores_by_year(p)
        )
        if feelings_ref is None:
            feelings_ref = list(emotions)
        elif list(emotions) != list(feelings_ref):
            raise ValueError('error: ValueError')
        doc_types_union.update(doc_types)
        years_union.update(by_year.keys())
        datasets.append({
                'label': str(label),
                'by_year': by_year,
                'meta': meta,
            })
    emotions = feelings_ref or []
    feeling_labels_map = (
        {
            str(e): str(emotion_labels.get(
                    str(e),
                    str(e),
                ))
            for e in emotions
        }
        if isinstance(
            emotion_labels,
            dict,
        )
        else None
    )
    years_sorted = sorted(years_union)
    doc_types_sorted = ['ALL'] + sorted((dt for dt in doc_types_union if dt != 'ALL'))
    payload = {
        'title': str(title),
        'emotions': emotions,
        'emotion_labels': feeling_labels_map,
        'years': years_sorted,
        'doc_types': doc_types_sorted,
        'datasets': datasets,
    }
    data_js = json.dumps(
        payload,
        ensure_ascii=False,
    ).replace(
        '</',
        '<\\/',
    )
    doc = textwrap.dedent(f"""\
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>{html.escape(title)}</title>
            <style>
              :root {{
                --ink: #0f172a;
                --muted: #64748b;
                --border: rgba(15, 23, 42, 0.10);
                --card: #ffffff;
                --bg-top: #fdf6ec;
                --bg-bottom: #eff2f6;
                --accent: #e07a5f;
              }}
              * {{ box-sizing: border-box; }}
              body {{
                margin: 0;
                font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
                color: var(--ink);
                background: linear-gradient(180deg, var(--bg-top), var(--bg-bottom));
              }}
              header {{
                padding: 18px 22px 14px;
                background: radial-gradient(circle at top left, #ffe8d1 0%, #f0f4f8 60%);
                border-bottom: 1px solid var(--border);
              }}
              header h1 {{
                margin: 0;
                font-size: 20px;
                letter-spacing: 0.2px;
              }}
              header p {{
                margin: 6px 0 0;
                color: var(--muted);
                font-size: 13px;
                line-height: 1.4;
                max-width: 980px;
              }}
              main {{
                padding: 16px 22px 26px;
                max-width: 1200px;
                margin: 0 auto;
              }}
              .card {{
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 14px;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
                padding: 12px 14px;
                margin: 12px 0;
              }}
              .controls {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px 14px;
                align-items: center;
              }}
              .controls label {{
                font-size: 12px;
                color: var(--muted);
                display: flex;
                gap: 8px;
                align-items: center;
                white-space: nowrap;
              }}
              select {{
                padding: 6px 10px;
                border: 1px solid var(--border);
                border-radius: 10px;
                background: #fff;
                color: var(--ink);
                font-size: 12px;
              }}
              .checks {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                align-items: center;
              }}
              .checks label {{
                font-size: 12px;
                color: var(--muted);
                display: flex;
                gap: 6px;
                align-items: center;
              }}
              canvas {{
                width: 100%;
                height: 380px;
                border: 1px solid var(--border);
                border-radius: 14px;
                background: #fff;
              }}
              .tooltip {{
                position: absolute;
                pointer-events: none;
                background: rgba(15, 23, 42, 0.92);
                color: #fff;
                padding: 8px 10px;
                border-radius: 10px;
                font-size: 12px;
                line-height: 1.35;
                max-width: 360px;
                box-shadow: 0 12px 26px rgba(0, 0, 0, 0.22);
                display: none;
                z-index: 10;
              }}
              table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 12px;
              }}
              th, td {{
                padding: 8px 10px;
                border-bottom: 1px solid var(--border);
                font-size: 12px;
                vertical-align: top;
              }}
              th {{
                text-align: left;
                color: #111827;
                font-weight: 700;
                background: #f8fafc;
              }}
              td.mono {{
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                color: var(--muted);
              }}
              .legend {{
                display: flex;
                flex-wrap: wrap;
                gap: 12px;
                margin-top: 10px;
                align-items: center;
                color: var(--muted);
                font-size: 12px;
              }}
              .swatch {{
                width: 12px;
                height: 12px;
                border-radius: 4px;
                display: inline-block;
                margin-right: 6px;
              }}
              .muted {{
                color: var(--muted);
              }}
            </style>
          </head>
          <body>
            <header>
              <h1>{html.escape(title)}</h1>
              <p>
                Filenames don't start with the year, but ICJ filenames contain a <code>YYYY-MM-DD</code> date token and scoring extracts it into <code>meta.date</code>.
                This tab aggregates per-document intensities (per 1k words) into an annual time series.
              </p>
            </header>
            <main>
              <section class="card">
                <div class="controls">
                  <label>Metric
                    <select id="metric"></select>
                  </label>
                  <label>Aggregation
                    <select id="agg">
                      <option value="mean">Mean</option>
                      <option value="median">Median</option>
                    </select>
                  </label>
                  <label>Doc type
                    <select id="docType"></select>
                  </label>
                </div>
                <div class="checks" id="datasetChecks" style="margin-top: 10px;"></div>
                <div style="position: relative; margin-top: 12px;">
                  <canvas id="chart"></canvas>
                  <div class="tooltip" id="tooltip"></div>
                </div>
                <div class="legend" id="legend"></div>
                <div class="muted" style="margin-top: 6px;">
                  Hover the chart to see per-year values and document counts.
                </div>
                <div id="tableWrap"></div>
              </section>
            </main>
            <script>
              const DATA = {data_js};
        
              function $(id) {{ return document.getElementById(id); }}
        
              function emoLabel(key) {{
                if (DATA.emotion_labels && DATA.emotion_labels[key]) return String(DATA.emotion_labels[key]);
                return String(key);
              }}
        
              function formatNum(x) {{
                if (x === null || x === undefined) return "";
                const v = Number(x);
                if (!Number.isFinite(v)) return String(x);
                const abs = Math.abs(v);
                if (abs !== 0 && (abs < 1e-3 || abs >= 1e4)) return v.toExponential(3);
                return v.toFixed(4).replace(/0+$/, "").replace(/\\.$/, "");
              }}
        
              function htmlEscape(s) {{
                return String(s)
                  .replaceAll("&", "&amp;")
                  .replaceAll("<", "&lt;")
                  .replaceAll(">", "&gt;")
                  .replaceAll('"', "&quot;")
                  .replaceAll("'", "&#39;");
              }}
        
              function yearsUnion(selectedDatasets) {{
                const ys = new Set();
                for (const ds of selectedDatasets) {{
                  const byYear = ds.by_year || {{}};
                  for (const y of Object.keys(byYear)) ys.add(y);
                }}
                return Array.from(ys).sort();
              }}
        
              function valueAt(ds, year, docType, metricKey, aggKey) {{
                const byYear = ds.by_year || {{}};
                const y = byYear[String(year)];
                if (!y) return null;
                const b = y[String(docType)];
                if (!b) return null;
                const isTotal = metricKey === "__TOTAL__";
                if (isTotal) {{
                  return aggKey === "median" ? b.median_total : b.mean_total;
                }}
                const idx = DATA.emotions.indexOf(metricKey);
                if (idx < 0) return null;
                const arr = aggKey === "median" ? b.median_emotions : b.mean_emotions;
                if (!Array.isArray(arr) || idx >= arr.length) return null;
                return Number(arr[idx]);
              }}
        
              function countAt(ds, year, docType) {{
                const byYear = ds.by_year || {{}};
                const y = byYear[String(year)];
                if (!y) return 0;
                const b = y[String(docType)];
                if (!b) return 0;
                return Number(b.n) || 0;
              }}
        
              function buildControls() {{
                const metric = $("metric");
                metric.innerHTML = "";
                const optTotal = document.createElement("option");
                optTotal.value = "__TOTAL__";
                optTotal.textContent = "Total emotionality (sum)";
                metric.appendChild(optTotal);
                for (const e of DATA.emotions) {{
                  const opt = document.createElement("option");
                  opt.value = e;
                  opt.textContent = emoLabel(e);
                  metric.appendChild(opt);
                }}
        
                const docType = $("docType");
                docType.innerHTML = "";
                for (const dt of DATA.doc_types) {{
                  const opt = document.createElement("option");
                  opt.value = dt;
                  opt.textContent = dt;
                  docType.appendChild(opt);
                }}
        
                const checks = $("datasetChecks");
                checks.innerHTML = "";
                for (let i = 0; i < DATA.datasets.length; i++) {{
                  const ds = DATA.datasets[i];
                  const label = document.createElement("label");
                  const input = document.createElement("input");
                  input.type = "checkbox";
                  input.checked = true;
                  input.dataset.idx = String(i);
                  label.appendChild(input);
                  const span = document.createElement("span");
                  span.textContent = ds.label;
                  label.appendChild(span);
                  checks.appendChild(label);
                }}
              }}
        
              function selectedDatasets() {{
                const checks = Array.from($("datasetChecks").querySelectorAll("input[type=checkbox]"));
                const out = [];
                for (const c of checks) {{
                  if (!c.checked) continue;
                  const idx = Number(c.dataset.idx);
                  const ds = DATA.datasets[idx];
                  if (ds) out.push(ds);
                }}
                return out;
              }}
        
              function draw() {{
                const dsSel = selectedDatasets();
                const metricKey = $("metric").value;
                const aggKey = $("agg").value;
                const docType = $("docType").value;
        
                const years = yearsUnion(dsSel);
                const canvas = $("chart");
                const ctx = canvas.getContext("2d");
        
                const rect = canvas.getBoundingClientRect();
                const dpr = window.devicePixelRatio || 1;
                canvas.width = Math.max(1, Math.floor(rect.width * dpr));
                canvas.height = Math.max(1, Math.floor(rect.height * dpr));
                ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        
                const W = rect.width;
                const H = rect.height;
                const m = {{l: 46, r: 18, t: 18, b: 34}};
                const plotW = Math.max(10, W - m.l - m.r);
                const plotH = Math.max(10, H - m.t - m.b);
        
                const series = [];
                let maxV = 0;
                for (let i = 0; i < dsSel.length; i++) {{
                  const ds = dsSel[i];
                  const vals = years.map(y => valueAt(ds, y, docType, metricKey, aggKey));
                  for (const v of vals) {{
                    if (v !== null && Number(v) > maxV) maxV = Number(v);
                  }}
                  series.push({{ds, vals}});
                }}
                const yMax = maxV > 0 ? maxV * 1.1 : 1;
        
                function xAt(i) {{
                  if (years.length <= 1) return m.l + plotW / 2;
                  return m.l + (i / (years.length - 1)) * plotW;
                }}
                function yAt(v) {{
                  const vv = Number(v) || 0;
                  return m.t + plotH - (vv / yMax) * plotH;
                }}
        
                ctx.clearRect(0, 0, W, H);
                ctx.fillStyle = "#ffffff";
                ctx.fillRect(0, 0, W, H);
        
                ctx.strokeStyle = "rgba(15,23,42,0.10)";
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(m.l, m.t);
                ctx.lineTo(m.l, m.t + plotH);
                ctx.lineTo(m.l + plotW, m.t + plotH);
                ctx.stroke();
        
                const yTicks = 5;
                ctx.fillStyle = "#64748b";
                ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, Courier New, monospace";
                for (let t = 0; t <= yTicks; t++) {{
                  const v = (t / yTicks) * yMax;
                  const y = yAt(v);
                  ctx.strokeStyle = "rgba(15,23,42,0.08)";
                  ctx.beginPath();
                  ctx.moveTo(m.l, y);
                  ctx.lineTo(m.l + plotW, y);
                  ctx.stroke();
                  ctx.fillText(formatNum(v), 6, y + 4);
                }}
        
                ctx.fillStyle = "#64748b";
                ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, Courier New, monospace";
                const maxLabels = 12;
                const step = Math.max(1, Math.ceil(years.length / maxLabels));
                for (let i = 0; i < years.length; i += step) {{
                  const x = xAt(i);
                  const y = m.t + plotH + 18;
                  ctx.fillText(String(years[i]), x - 14, y);
                }}
        
                const colors = ["#e07a5f", "#2563eb", "#10b981", "#7c3aed", "#f59e0b", "#0ea5e9"];
                for (let s = 0; s < series.length; s++) {{
                  const col = colors[s % colors.length];
                  const vals = series[s].vals;
                  ctx.strokeStyle = col;
                  ctx.lineWidth = 2;
                  ctx.beginPath();
                  let started = false;
                  for (let i = 0; i < vals.length; i++) {{
                    const v = vals[i];
                    if (v === null || !Number.isFinite(Number(v))) {{
                      started = false;
                      continue;
                    }}
                    const x = xAt(i);
                    const y = yAt(v);
                    if (!started) {{
                      ctx.moveTo(x, y);
                      started = true;
                    }} else {{
                      ctx.lineTo(x, y);
                    }}
                  }}
                  ctx.stroke();
        
                  ctx.fillStyle = col;
                  for (let i = 0; i < vals.length; i++) {{
                    const v = vals[i];
                    if (v === null || !Number.isFinite(Number(v))) continue;
                    const x = xAt(i);
                    const y = yAt(v);
                    ctx.beginPath();
                    ctx.arc(x, y, 2.6, 0, Math.PI * 2);
                    ctx.fill();
                  }}
                }}
        
                const legend = $("legend");
                legend.innerHTML = "";
                for (let s = 0; s < series.length; s++) {{
                  const item = document.createElement("div");
                  const sw = document.createElement("span");
                  sw.className = "swatch";
                  sw.style.background = colors[s % colors.length];
                  item.appendChild(sw);
                  const txt = document.createElement("span");
                  txt.textContent = series[s].ds.label;
                  item.appendChild(txt);
                  legend.appendChild(item);
                }}
        
                const tableWrap = $("tableWrap");
                let th = "<th>Year</th>";
                for (const s of series) {{
                  th += `<th>${{htmlEscape(s.ds.label)}}</th><th>${{htmlEscape(s.ds.label)}} n</th>`;
                }}
                let rows = "";
                for (let i = 0; i < years.length; i++) {{
                  const y = years[i];
                  let td = `<td class='mono'>${{y}}</td>`;
                  for (let s = 0; s < series.length; s++) {{
                    const v = series[s].vals[i];
                    const n = countAt(series[s].ds, y, docType);
                    td += `<td class='mono'>${{formatNum(v)}}</td><td class='mono'>${{n}}</td>`;
                  }}
                  rows += `<tr>${{td}}</tr>`;
                }}
                tableWrap.innerHTML = `
                  <table>
                    <thead><tr>${{th}}</tr></thead>
                    <tbody>${{rows}}</tbody>
                  </table>
                `;
        
                const tooltip = $("tooltip");
                function onMove(ev) {{
                  if (!years.length || !series.length) return;
                  const r = canvas.getBoundingClientRect();
                  const x = ev.clientX - r.left;
                  const y = ev.clientY - r.top;
                  if (x < m.l || x > m.l + plotW || y < m.t || y > m.t + plotH) {{
                    tooltip.style.display = "none";
                    return;
                  }}
                  const tt = (x - m.l) / Math.max(1e-9, plotW);
                  const idx = Math.round(tt * (years.length - 1));
                  const year = years[Math.max(0, Math.min(years.length - 1, idx))];
                  let body = `<div><b>${{year}}</b></div>`;
                  for (let s = 0; s < series.length; s++) {{
                    const ds = series[s].ds;
                    const v = series[s].vals[idx];
                    const n = countAt(ds, year, docType);
                    const sw = `<span class='swatch' style='background:${{colors[s % colors.length]}}; display:inline-block;'></span>`;
                    body += `<div style='margin-top:4px;'>${{sw}} ${{htmlEscape(ds.label)}}: <b>${{formatNum(v)}}</b> <span style='opacity:0.85'>(n=${{n}})</span></div>`;
                  }}
                  tooltip.innerHTML = body;
                  tooltip.style.display = "block";
                  tooltip.style.left = `${{Math.min(r.width - 10, x + 12)}}px`;
                  tooltip.style.top = `${{Math.min(r.height - 10, y + 12)}}px`;
                }}
                canvas.onmousemove = onMove;
                canvas.onmouseleave = () => {{ tooltip.style.display = "none"; }};
              }}
        
              function attach() {{
                buildControls();
                $("metric").addEventListener("change", draw);
                $("agg").addEventListener("change", draw);
                $("docType").addEventListener("change", draw);
                $("datasetChecks").addEventListener("change", draw);
                window.addEventListener("resize", draw);
                draw();
              }}
        
              attach();
            </script>
          </body>
        </html>
        """)
    out_p.write_text(
        doc,
        encoding='utf-8',
    )
    return str(out_p)
