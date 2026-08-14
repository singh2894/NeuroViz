# app/report.py — self-contained HTML report export (NeuroViz Identity styling)

import base64
import datetime
import html
import json


def _chart_png_uri(spec: dict, scale: float = 2.0) -> str | None:
    """Vega-Lite spec -> PNG data URI. Returns None if conversion fails."""
    try:
        import vl_convert as vlc

        png = vlc.vegalite_to_png(json.dumps(spec), scale=scale)
        return "data:image/png;base64," + base64.b64encode(png).decode()
    except Exception:
        return None


def build_report(
    title: str,
    kpis: list[tuple[str, str]],
    charts: list[tuple[dict, str]],
    stats_html: str = "",
) -> str:
    """One printable, self-contained HTML file: KPI cards, chart grid
    (as embedded PNGs), and the exact statistics table. Open it, print it,
    or email it — no app needed."""
    kpi_html = "".join(
        f'<div class="kpi"><div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value">{html.escape(value)}</div></div>'
        for label, value in kpis
    )
    chart_html = ""
    for spec, caption in charts:
        uri = _chart_png_uri(spec)
        if uri is None:
            continue
        chart_html += (
            f'<figure><img src="{uri}" alt="{html.escape(caption)}">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        )
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>{html.escape(title)} — NeuroViz report</title>
<style>
  body {{ font-family: 'JetBrains Mono', ui-monospace, monospace;
         background: #F4F4F1; color: #1C1C1C; margin: 0;
         padding: 48px 24px; }}
  main {{ max-width: 960px; margin: 0 auto; }}
  h1 {{ font-weight: 500; letter-spacing: -0.055em; margin: 0; }}
  .meta {{ color: #7A7A74; font-size: 12px; margin-bottom: 32px; }}
  .rule {{ border-top: 1px solid #DDDDD7; margin: 24px 0; }}
  .kpis {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .kpi {{ background: #FFFFFF; border: 1px solid #DDDDD7; padding: 16px 20px;
         min-width: 160px; }}
  .kpi-label {{ font-size: 10px; text-transform: uppercase;
               letter-spacing: 0.12em; color: #7A7A74; }}
  .kpi-value {{ font-size: 26px; font-weight: 500; letter-spacing: -0.045em; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  figure {{ background: #FFFFFF; border: 1px solid #DDDDD7; padding: 12px;
           margin: 0; }}
  figure img {{ width: 100%; height: auto; }}
  figcaption {{ font-size: 12px; color: #55554E; padding-top: 8px; }}
  table {{ border-collapse: collapse; font-size: 12px; background: #FFFFFF;
          width: 100%; }}
  th, td {{ border: 1px solid #DDDDD7; padding: 6px 10px; text-align: right; }}
  th {{ color: #55554E; }}
  @media print {{ body {{ background: #FFFFFF; padding: 0; }} }}
  @media (max-width: 700px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
<div class="meta">NeuroViz report · generated {stamp} · computed locally</div>
<div class="kpis">{kpi_html}</div>
<div class="rule"></div>
<div class="grid">{chart_html}</div>
{f'<div class="rule"></div><h3>Exact statistics</h3>{stats_html}' if stats_html else ""}
</main></body></html>"""
