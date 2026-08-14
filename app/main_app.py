# app/main_app.py

import io
import sys
from pathlib import Path

import altair as alt
import polars as pl
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.compilers.altair_compile import (  # noqa: E402
    build_category_chart,
    build_distribution_chart,
    build_scatter_chart,
    build_trend_chart,
    infer_schema,
    pick_category_col,
)
from app.compilers.altair_compile import compile as compile_chart  # noqa: E402
from app.components.filters import apply_filters, build_filters  # noqa: E402
from app.parsers.nlp import Intent, answer_question, parse_intent  # noqa: E402


def get_datasets() -> dict[str, pl.DataFrame]:
    """Session-only dataset store: nothing is ever written to disk."""
    return st.session_state.setdefault("datasets", {})


def select_dataset() -> pl.DataFrame | None:
    """Sidebar dataset picker shared by all pages; remembers the selection."""
    datasets = get_datasets()
    if not datasets:
        st.warning("No datasets in this session — upload one first.")
        if st.button("Go to Upload →"):
            st.switch_page(upload)
        return None

    names = list(datasets)
    current = st.session_state.get("dataset", "")
    selected = st.sidebar.selectbox(
        "Dataset",
        names,
        index=names.index(current) if current in names else 0,
    )
    st.session_state["dataset"] = selected
    return datasets[selected]


def save_upload(uploaded) -> str | None:
    """Load an uploaded CSV/Excel/Parquet file into session memory."""
    name = Path(uploaded.name).stem
    suffix = Path(uploaded.name).suffix.lower()

    try:
        if suffix == ".parquet":
            df = pl.read_parquet(io.BytesIO(uploaded.getvalue()))
        elif suffix == ".csv":
            df = pl.read_csv(uploaded, infer_schema_length=10000)
        elif suffix in (".xlsx", ".xls"):
            df = pl.read_excel(io.BytesIO(uploaded.getvalue()))
        else:
            st.error("Please upload a CSV, Excel, or Parquet file.")
            return None
    except Exception as exc:
        st.error(f"Could not read that file: {exc}")
        return None
    get_datasets()[name] = df
    return name


def stats_table(df: pl.DataFrame) -> None:
    """Exact descriptive statistics: count, unique, mode (top), mean, std,
    min, quartiles, max — computed, not LLM-generated."""
    st.caption(f"{df.height:,} rows × {df.width} columns")
    st.dataframe(
        df.to_pandas().describe(include="all").T,
        use_container_width=True,
    )


# -----------------------------------------------------
# Page 1: Upload
# -----------------------------------------------------
def upload_page():
    st.title("Data")
    st.write(
        "Bring any tabular dataset — sales, health, climate, anything. "
        "Data stays in memory for this session only; nothing is saved to "
        "disk. Close or refresh the tab and it's gone."
    )

    uploaded = st.file_uploader(
        "Drop a file here",
        type=["csv", "xlsx", "xls", "parquet"],
    )
    if uploaded is not None:
        name = save_upload(uploaded)
        if name:
            st.session_state["dataset"] = name
            st.success(f"Loaded {name} — this session only")

    datasets = get_datasets()
    if not datasets:
        st.info("No datasets yet — upload a file to get started.")
        return

    st.subheader("This session's datasets")
    names = list(datasets)
    current = st.session_state.get("dataset", "")
    selected = st.selectbox(
        "Preview a dataset",
        names,
        index=names.index(current) if current in names else 0,
    )
    st.session_state["dataset"] = selected

    df = datasets[selected]
    st.caption(f"{df.height:,} rows × {df.width} columns")
    st.dataframe(df.head(20).to_pandas(), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Open dashboard →", type="primary"):
            st.switch_page(dashboard)
    with col2:
        if st.button("Remove from session"):
            datasets.pop(selected, None)
            st.session_state.pop("dataset", None)
            st.rerun()


# -----------------------------------------------------
# Page 2: Dashboard (auto-generated BI overview)
# -----------------------------------------------------
def dashboard_page():
    st.title("Dashboard")
    df = select_dataset()
    if df is None:
        return

    chosen = build_filters(df)
    df = apply_filters(df, chosen)
    schema = infer_schema(df)
    cat_col = pick_category_col(schema, df)

    # Cross-filter: read the bar-click selection made on the previous rerun
    selected: list[str] = []
    state = st.session_state.get("dash_cat")
    if cat_col and state is not None:
        try:
            selected = [p[cat_col] for p in state.selection["cat_sel"]]
        except (KeyError, AttributeError, TypeError):
            selected = []
    fdf = df.filter(pl.col(cat_col).is_in(selected)) if selected else df

    # KPI cards (respect cross-filter)
    kpis = st.columns(4)
    kpis[0].metric("Rows", f"{fdf.height:,}", border=True)
    kpis[1].metric("Columns", str(fdf.width), border=True)
    for slot, col in zip(kpis[2:], schema["numeric_cols"][:2], strict=False):
        total = fdf.get_column(col).sum()
        slot.metric(
            f"Total {col}",
            f"{total:,.0f}" if total is not None else "–",
            border=True,
        )
    if selected:
        st.caption(f"Cross-filtered to: {', '.join(map(str, selected))}")

    with st.expander("Dataset summary"):
        stats_table(fdf)
        if st.button("Generate commentary"):
            with st.spinner("reading the data profile..."):
                st.session_state["dash_summary"] = answer_question(
                    "Summarize this dataset for an analyst: what it contains, "
                    "key figures, notable patterns or outliers. Short bullets.",
                    fdf,
                )
        if "dash_summary" in st.session_state:
            if st.session_state["dash_summary"]:
                st.markdown(st.session_state["dash_summary"])
            else:
                st.info("Commentary needs the LLM configured (set LLM_API_URL).")

    # Auto chart grid — reuses the same builders the AI path uses
    blank = Intent(kind="trend", text="")
    cat_chart, cat_caption = build_category_chart(df, blank, schema, {})
    others = [
        c
        for c in (
            build_trend_chart(fdf, blank, schema, {}),
            build_distribution_chart(fdf, blank, schema, {}),
            build_scatter_chart(fdf, schema, {}),
        )
        if c[0] is not None
    ]
    if cat_chart is None and not others:
        st.info("Not enough structure in this dataset to auto-build charts.")
        return

    row = st.columns(2)
    slots = list(row)
    if cat_chart is not None:
        with slots.pop(0):
            st.altair_chart(
                cat_chart,
                use_container_width=True,
                on_select="rerun",
                key="dash_cat",
                theme=None,
            )
            st.caption(f"{cat_caption} — click bars to cross-filter")
    while len(slots) < len(others):
        slots.extend(st.columns(2))
    for slot, (chart, caption) in zip(slots, others, strict=False):
        with slot:
            st.altair_chart(chart, use_container_width=True, theme=None)
            st.caption(caption)

    # Drill-down: one category clicked + another categorical level available
    if len(selected) == 1:
        drill_cols = [c for c in schema["categorical_cols"] if c != cat_col]
        if drill_cols:
            sub_chart, sub_caption = build_category_chart(
                fdf, blank, schema, {"categorical": [drill_cols[0]], "numeric": []}
            )
            if sub_chart is not None:
                st.altair_chart(sub_chart, use_container_width=True, theme=None)
                st.caption(f"Drill-down into {selected[0]}: {sub_caption}")


# -----------------------------------------------------
# Page 3: Explore (manual chart builder, Tableau-style)
# -----------------------------------------------------
def explore_page():
    st.title("Build")
    df = select_dataset()
    if df is None:
        return

    chosen = build_filters(df)
    df = apply_filters(df, chosen)
    schema = infer_schema(df)
    numeric = schema["numeric_cols"]
    all_cols = df.columns
    cat_options = ["(none)"] + schema["categorical_cols"]

    tab_chart, tab_pivot = st.tabs(["Chart", "Pivot table"])

    with tab_pivot:
        p1, p2, p3, p4 = st.columns(4)
        rows = p1.selectbox("Rows", all_cols, key="pv_rows")
        col_dim = p2.selectbox("Columns", cat_options, key="pv_cols")
        val = p3.selectbox("Values", numeric, key="pv_vals") if numeric else None
        pagg = p4.selectbox(
            "Aggregate",
            ["sum", "mean", "min", "max", "median", "count"],
            key="pv_agg",
        )
        if val is None:
            st.info("Pivot needs a numeric column.")
        else:
            try:
                if col_dim == "(none)":
                    pv = (
                        df.group_by(rows)
                        .agg(getattr(pl.col(val), pagg)().alias(val))
                        .sort(rows)
                    )
                else:
                    pv = df.pivot(
                        index=rows,
                        on=col_dim,
                        values=val,
                        aggregate_function=pagg,
                    ).sort(rows)
                st.dataframe(pv.to_pandas(), use_container_width=True)
                st.download_button(
                    "Download CSV",
                    data=pv.write_csv(),
                    file_name="pivot.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Pivot failed: {exc}")

    with tab_chart:
        _chart_builder(df, schema, numeric, all_cols, cat_options)


def _chart_builder(df, schema, numeric, all_cols, cat_options):
    c1, c2, c3, c4, c5 = st.columns(5)
    chart_type = c1.selectbox(
        "Chart", ["Bar", "Line", "Area", "Scatter", "Histogram", "Box", "Pie"]
    )
    x = c2.selectbox(
        "Category" if chart_type == "Pie" else "X axis",
        all_cols,
    )
    y = c3.selectbox("Y axis", numeric) if chart_type != "Histogram" else None
    agg = (
        c4.selectbox("Aggregate", ["sum", "mean", "min", "max", "median", "count"])
        if chart_type in ("Bar", "Line", "Area", "Pie")
        else None
    )
    color = c5.selectbox("Color by", cat_options) if chart_type != "Pie" else "(none)"
    facet = (
        st.selectbox("Facet by (small multiples)", cat_options)
        if chart_type != "Pie"
        else "(none)"
    )
    bar_mode = (
        st.radio("Bar mode", ["stacked", "grouped"], horizontal=True)
        if chart_type == "Bar" and color != "(none)"
        else None
    )

    if x is None or (y is None and chart_type != "Histogram"):
        st.info("This chart needs a numeric column for the Y axis.")
        return

    if df.height > 50_000:
        # ponytail: sample for responsiveness; server-side aggregation if needed
        df = df.sample(50_000, seed=0)
        st.caption("Showing a 50,000-row sample for responsiveness.")
    pdf = df.to_pandas()

    marks = {
        "Bar": alt.Chart(pdf).mark_bar(),
        "Line": alt.Chart(pdf).mark_line(strokeWidth=2.5),
        "Area": alt.Chart(pdf).mark_area(opacity=0.7),
        "Scatter": alt.Chart(pdf).mark_circle(size=60, opacity=0.6),
        "Histogram": alt.Chart(pdf).mark_bar(opacity=0.7),
        "Box": alt.Chart(pdf).mark_boxplot(),
        "Pie": alt.Chart(pdf).mark_arc(innerRadius=60),
    }
    base = marks[chart_type]

    if chart_type == "Pie":
        enc = {
            "theta": alt.Theta(y, aggregate=agg),
            "color": alt.Color(x),
            "tooltip": [x, alt.Tooltip(y, aggregate=agg, format=",.2f")],
        }
    elif chart_type == "Histogram":
        enc = {"x": alt.X(x, bin=alt.Bin(maxbins=30)), "y": alt.Y("count()")}
    elif agg:
        enc = {
            "x": x,
            "y": alt.Y(y, aggregate=agg, axis=alt.Axis(format="~s")),
            "tooltip": [x, alt.Tooltip(y, aggregate=agg, format=",.2f")],
        }
    else:
        enc = {
            "x": x,
            "y": alt.Y(y, axis=alt.Axis(format="~s")),
            "tooltip": [x, alt.Tooltip(y, format=",.2f")],
        }
    if color != "(none)":
        enc["color"] = color
    if bar_mode == "grouped":
        enc["xOffset"] = color

    if facet != "(none)":
        enc["facet"] = alt.Facet(facet, columns=3)
        chart = base.encode(**enc).properties(height=220, width=280)
    else:
        chart = base.encode(**enc).properties(height=460)
        if chart_type in ("Scatter", "Line", "Area"):
            chart = chart.interactive()  # zoom + pan like BI tools
    st.altair_chart(chart, use_container_width=True, theme=None)


# -----------------------------------------------------
# Page 4: Ask AI (natural language)
# -----------------------------------------------------
def visualize_page():
    st.markdown(
        """
        <style>
        .neuroviz-hero { text-align: center; margin-bottom: 1.5rem; }
        .neuroviz-hero h1 {
            font-size: clamp(2.4rem, 4vw, 3.4rem); margin-bottom: 0.25rem;
        }
        .neuroviz-hero p { color: #55554E; }
        .query-input .stTextInput>div { margin: 0 auto; width: min(900px, 95vw); }
        .query-input .stTextInput input { min-height: 2.5rem; font-size: 1.1rem; }
        </style>
        <div class="neuroviz-hero">
            <h1>Ask in English. Get a real chart.</h1>
            <p>Runs on your machine. No key required.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = select_dataset()
    if df is None:
        return

    mode = st.radio(
        "mode",
        ["chart", "answer in words"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown('<div class="query-input">', unsafe_allow_html=True)
    query = st.text_input(
        " ",
        label_visibility="collapsed",
        placeholder=(
            "Ask a question about the data (e.g., 'BMI vs age') — press Enter"
        ),
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if query and mode == "answer in words":
        with st.spinner("reading the data profile..."):
            answer = answer_question(query, df)
        if answer:
            st.markdown(
                '<div style="display:flex;align-items:center;gap:8px;'
                'font-size:12px;color:#55554E;justify-content:center;">'
                '<span style="width:8px;height:8px;background:#4A7C63;'
                'display:inline-block;"></span>'
                "answered from schema · summary statistics · 5 sample rows</div>",
                unsafe_allow_html=True,
            )
            st.markdown(answer)
        else:
            st.info(
                "Text answers need the LLM configured (set LLM_API_URL). "
                "Charts work offline — switch to chart mode."
            )
        with st.expander("Exact statistics"):
            stats_table(df)
        return

    st.write("### Data preview")
    st.dataframe(df.head(10).to_pandas(), use_container_width=True)

    chosen = build_filters(df)
    df = apply_filters(df, chosen)
    if chosen:
        st.write(f"### Filtered data ({df.height} rows)")
        st.dataframe(df.to_pandas(), use_container_width=True)

    if query:
        intent = parse_intent(query, columns=df.columns)
        chart, caption = compile_chart(intent, df)

        # Voice: show the reasoning — state what was parsed
        parts = [f"kind={intent.kind}"]
        if intent.metric:
            parts.append(f"metric={intent.metric}")
        if intent.agg:
            parts.append(f"agg={intent.agg}")
        if intent.time_grain:
            parts.append(f"grain={intent.time_grain}")
        if intent.columns:
            parts.append("cols=" + ",".join(intent.columns))
        dot = "#4A7C63" if chart else "#A8663F"
        label = "parsed" if chart else "no chart"
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;'
            'font-size:12px;color:#55554E;justify-content:center;">'
            f'<span style="width:8px;height:8px;background:{dot};'
            'display:inline-block;"></span>'
            f"{label} · {' · '.join(parts)}</div>",
            unsafe_allow_html=True,
        )

        if chart:
            st.altair_chart(chart, use_container_width=True, theme=None)
            st.caption(f"Read as: {caption}")

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download JSON",
                    data=chart.to_json().encode(),
                    mime="application/json",
                    file_name="chart.json",
                )
            with col2:
                with st.expander("Altair JSON"):
                    st.code(chart.to_json()[:2000], language="json")
        else:
            st.info(caption)


# -----------------------------------------------------
# Navigation
# -----------------------------------------------------
_LOGO = str(Path(__file__).resolve().parent / "assets" / "logo.svg")

st.set_page_config(page_title="NeuroViz", page_icon=_LOGO, layout="wide")
st.logo(_LOGO)

# "NeuroViz Identity": JetBrains Mono everywhere; hierarchy from size,
# weight and rule-work (colors/radius live in .streamlit/config.toml).
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    html, body, p, li, h1, h2, h3, h4, h5, h6,
    .stMarkdown, .stTextInput input, .stSelectbox, .stButton button,
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
    [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"] {
        font-family: 'JetBrains Mono', ui-monospace, monospace;
    }
    h1 { font-weight: 500; letter-spacing: -0.055em; }
    h2, h3 { font-weight: 600; letter-spacing: -0.035em; }
    [data-testid="stMetricLabel"] p {
        text-transform: uppercase; letter-spacing: 0.12em;
        font-size: 10px; color: #7A7A74;
    }
    [data-testid="stMetricValue"] {
        font-weight: 500; letter-spacing: -0.045em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

upload = st.Page(upload_page, title="Data", icon=":material/database:", default=True)
dashboard = st.Page(dashboard_page, title="Dashboard", icon=":material/monitoring:")
explore = st.Page(explore_page, title="Build", icon=":material/stacked_bar_chart:")
visualize = st.Page(visualize_page, title="Ask", icon=":material/terminal:")

st.navigation([upload, dashboard, explore, visualize]).run()
