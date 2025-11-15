# app/main_app.py

from pathlib import Path

import polars as pl
import streamlit as st
from compilers.altair_compile import compile as compile_chart
from components.filters import apply_filters, build_filters
from parsers.nlp import parse_intent


def handle_dataset_upload() -> None:
    """Allow users to drop a CSV/Parquet file and store it under data/."""
    uploaded = st.sidebar.file_uploader(
        "Upload CSV or Parquet",
        type=["csv", "parquet"],
        help="CSV files are converted to Parquet automatically.",
    )
    if uploaded is None:
        return

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    incoming = Path(uploaded.name)
    suffix = incoming.suffix.lower()

    try:
        if suffix == ".parquet":
            target = data_dir / incoming.name
            with target.open("wb") as fh:
                fh.write(uploaded.getbuffer())
        elif suffix == ".csv":
            df_uploaded = pl.read_csv(uploaded)
            target = data_dir / f"{incoming.stem}.parquet"
            df_uploaded.write_parquet(target)
        else:
            st.sidebar.error("Please upload a CSV or Parquet file.")
            return
    except Exception as exc:  # pragma: no cover - Streamlit UI only
        st.sidebar.error(f"Failed to save file: {exc}")
        return

    st.sidebar.success(f"Saved dataset to {target}")
    st.experimental_rerun()


# -----------------------------------------------------
# Page setup
# -----------------------------------------------------
st.set_page_config(
    page_title="NeuroViz",
    page_icon="📈",
    layout="wide",
)

st.sidebar.header("Seed data")
handle_dataset_upload()

# -----------------------------------------------------
# Load data from /data as Parquet
# -----------------------------------------------------
parquet_files = list(Path("data").glob("*.parquet"))
dataset = st.sidebar.selectbox(
    "Choose Parquet",
    parquet_files,
    index=0 if parquet_files else None,
)

df = pl.read_parquet(dataset) if dataset else None

st.markdown(
    """
    <style>
    .neuroviz-hero {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .neuroviz-hero h1 {
        font-size: clamp(2.4rem, 4vw, 3.4rem);
        margin-bottom: 0.25rem;
    }
    .neuroviz-hero p {
        font-size: 1rem;
        color: rgba(255,255,255,0.75);
        margin-bottom: 0.5rem;
    }
    .query-input .stTextInput>div {
        margin: 0 auto;
        width: min(900px, 95vw);
    }
    .query-input .stTextInput input {
        min-height: 2.5rem;
        font-size: 1.1rem;
        padding-top: 0.35rem;
        padding-bottom: 0.35rem;
    }
    </style>
    <div class="neuroviz-hero">
        <h1>NeuroViz (Data Visualization Assistant) 📊</h1>
        <p>Ask a question about the data (e.g., 'trend of churn by month in 2021')</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="query-input">', unsafe_allow_html=True)
query = st.text_input(
    " ",
    label_visibility="collapsed",
    placeholder="Ask a question about the data (e.g., 'BMI vs age')",
)
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------
# If we have data, show preview and filters
# -----------------------------------------------------
if df is not None:
    st.write("### Data preview")
    st.dataframe(df.head(10).to_pandas())

    original_df = df
    # Sidebar filters
    chosen = build_filters(original_df)
    df = apply_filters(original_df, chosen)

    if chosen:
        # Show full filtered dataset so users can review every row
        filtered_pdf = df.to_pandas()
        st.write(f"### Filtered data ({len(filtered_pdf)} rows)")
        st.dataframe(filtered_pdf, use_container_width=True)
    # -------------------------------------------------
    # Handle query → intent → chart
    # -------------------------------------------------
    if query:
        intent = parse_intent(query)
        chart, caption = compile_chart(intent, df)

        if chart:
            st.altair_chart(chart, use_container_width=True)
            st.caption(caption)

            # Buttons under the chart
            col1, col2, col3 = st.columns(3)

            with col1:
                st.download_button(
                    "Download JSON",
                    data=chart.to_json().encode(),
                    mime="application/json",
                    file_name="chart.json",
                )

            with col2:
                st.code(chart.to_json()[:400] + " ...", language="json")

            with col3:
                st.text_area("Altair JSON", chart.to_json(), height=120)
        else:
            st.info(caption)
else:
    st.warning("Drop a Parquet file into the /data folder to get started.")
