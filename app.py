from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from html import escape

import pandas as pd
import streamlit as st

from data import make_sample_data
from security_utils import MAX_UPLOAD_SIZE_BYTES, prepare_csv_export
from ticket_intelligence import (
    analyze_sentiment,
    assess_urgency,
    batch_analyze,
    classify_ticket,
    discover_topics,
    evaluate_classifier,
    find_column,
    normalize_dataset,
    suggest_department,
    train_classifier,
)


st.set_page_config(
    page_title="Signal Desk | Ticket Intelligence",
    page_icon="SD",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
        """
    <style>
    :root {
            --ink: #172522;
            --muted: #64716c;
            --paper: #f3f5f0;
            --panel: #ffffff;
            --sidebar: #e9eee6;
            --line: #dfe5dc;
            --green: #a9d77a;
            --green-dark: #365d3d;
            --button-background: #172522;
            --button-foreground: #ffffff;
            --coral: #e77d61;
            --yellow: #f0c765;
            --blue: #83b8c6;
    }
        html, body, [class*="css"] { font-family: "Segoe UI", sans-serif; color: var(--ink); }
        .stApp {
            --ink: light-dark(#172522, #edf4ef);
            --muted: light-dark(#64716c, #a8b7ac);
            --paper: light-dark(#f3f5f0, #101815);
            --panel: light-dark(#ffffff, #1b2520);
            --sidebar: light-dark(#e9eee6, #141e19);
            --line: light-dark(#dfe5dc, #36463c);
            --green-dark: light-dark(#365d3d, #c2e6a0);
            --button-background: light-dark(#172522, #a9d77a);
            --button-foreground: light-dark(#ffffff, #172522);
            background: var(--paper);
            color: var(--ink);
        }
        [data-testid="stSidebar"] { background: var(--sidebar); border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--muted); }
    .block-container { padding-bottom: 3rem; max-width: 1440px; }
    [data-testid="stMainBlockContainer"] { padding-top: 4rem; }
    h1, h2, h3 { letter-spacing: 0 !important; color: var(--ink); }
    h1 { font-size: 2rem !important; font-weight: 800 !important; }
    h2 { font-size: 1.2rem !important; font-weight: 800 !important; }
    h3 { font-size: 1rem !important; font-weight: 800 !important; }
        .eyebrow { color: var(--green-dark); font: 500 .72rem monospace; text-transform: uppercase; letter-spacing: 0; }
        .metric-card { background: var(--panel); color: var(--ink); border: 1px solid var(--line); border-radius: 7px; padding: 1rem 1.1rem; min-height: 112px; }
    .metric-label { color: var(--muted); font-size: .78rem; font-weight: 700; }
    .metric-value { color: var(--ink); font-size: 1.8rem; font-weight: 800; margin-top: .35rem; }
    .metric-note { color: var(--muted); font-size: .72rem; margin-top: .15rem; }
    .result-band { border-radius: 7px; padding: 1.1rem 1.2rem; margin: .35rem 0 .7rem; }
    .mono { font-family: 'DM Mono', monospace; }
        div[data-testid="stMetric"] { background: var(--panel); color: var(--ink); border: 1px solid var(--line); border-radius: 7px; padding: .8rem 1rem; }
    div[data-testid="stMetricLabel"] p { color: var(--muted); }
        div[data-testid="stMetricValue"] { color: var(--ink); }
        .stButton > button { border-radius: 5px; border: 1px solid var(--line); background: var(--panel); color: var(--ink); font-weight: 700; }
        .stButton > button[kind="primary"] { background: var(--button-background); color: var(--button-foreground); border-color: var(--button-background); }
    .stTabs [data-baseweb="tab-list"] { gap: 1.2rem; }
    .stTabs [data-baseweb="tab"] { height: 3rem; }
    </style>
        """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Evaluating the selected model…", max_entries=12)
def cached_evaluation(frame: pd.DataFrame, model_name: str) -> dict:
    return evaluate_classifier(frame, model_name)


def sample_frame(frame: pd.DataFrame, limit: int, seed: int = 42) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame
    return frame.sample(n=limit, random_state=seed).reset_index(drop=True)


def evaluation_fingerprint(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(
        frame[["narrative", "category"]], index=True
    ).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def session_model(frame: pd.DataFrame, model_name: str):
    cache_key = (model_name, evaluation_fingerprint(frame))
    if st.session_state.get("_model_cache_key") != cache_key:
        st.session_state["_model_cache"] = train_classifier(frame, model_name)
        st.session_state["_model_cache_key"] = cache_key
    return st.session_state["_model_cache"]


def metric(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{escape(str(label))}</div>'
        f'<div class="metric-value">{escape(str(value))}</div>'
        f'<div class="metric-note">{escape(str(note))}</div></div>',
        unsafe_allow_html=True,
    )


def page_heading(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="eyebrow">{escape(kicker)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


input_error = None

with st.sidebar:
    st.markdown('<div class="eyebrow">SIGNAL DESK / 01</div>', unsafe_allow_html=True)
    st.markdown("## Ticket intelligence")
    st.caption("A focused workbench for complaint triage and emerging issues.")
    view = st.radio(
        "Workspace",
        ["Overview", "Live triage", "Recurring issues", "Model lab"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("### Data source")
    source = st.radio("Source", ["Sample dataset", "Upload CFPB CSV"], label_visibility="collapsed")
    uploaded = None
    selected_label = None
    if source == "Upload CFPB CSV":
        uploaded = st.file_uploader("Complaint export", type=["csv"])
        st.caption("CSV uploads are limited to 25 MB. Use a CFPB CSV with complaint narratives and a category field.")
        if uploaded is not None and uploaded.size > MAX_UPLOAD_SIZE_BYTES:
            input_error = "This CSV exceeds the 25 MB upload limit. Export a smaller date range and try again."
        elif uploaded is not None:
            try:
                preview = pd.read_csv(uploaded, nrows=5)
                suggested = find_column(preview, "product")
                options = list(preview.columns)
                default_index = options.index(suggested) if suggested in options else 0
                selected_label = st.selectbox("Training label", options, index=default_index)
            except Exception:
                input_error = "Could not read the CSV header. Confirm it is a valid UTF-8 CSV file."


if source == "Upload CFPB CSV" and uploaded is None:
    st.info("Upload a CFPB CSV or select Sample dataset to continue.")
    st.stop()
if input_error:
    st.error(input_error)
    st.stop()

if source == "Upload CFPB CSV":
    try:
        uploaded.seek(0)
        raw_data = pd.read_csv(uploaded, low_memory=False)
        dataset = normalize_dataset(raw_data, label_column=selected_label)
        dataset_source = "Uploaded CFPB data"
    except Exception:
        st.error(
            "The uploaded CSV could not be prepared. Check its text and label columns, "
            "then upload it again. No sample data was substituted."
        )
        st.stop()
else:
    dataset = make_sample_data()
    dataset_source = "Bundled illustrative dataset"

# Bound expensive local text work on a very large upload; headline counts still use all rows.
analysis_data = sample_frame(dataset, 5000)
training_data = sample_frame(dataset, 20000)
model_options = ["Logistic Regression", "Linear SVM", "Naive Bayes"]
model_name = st.session_state.get("model_name", "Logistic Regression")
model = session_model(training_data, model_name) if view != "Model lab" else None


if view == "Overview":
    page_heading("OPERATIONS / OVERVIEW", "Complaint signal, at a glance", "A live read on incoming volume, customer tone, and where work is landing.")
    sentiment_rows = batch_analyze(analysis_data["narrative"].tolist(), model)
    urgent_count = int(sentiment_rows["urgency"].isin(["Urgent", "High"]).sum())
    negative_share = float(sentiment_rows["sentiment"].eq("Negative").mean()) if len(sentiment_rows) else 0
    category_count = dataset["category"].nunique()

    columns = st.columns(4, gap="medium")
    with columns[0]:
        metric("COMPLAINTS IN DATA", f"{len(dataset):,}", dataset_source)
    with columns[1]:
        metric("CATEGORIES", f"{category_count:,}", "Distinct training labels")
    with columns[2]:
        metric("HIGH / URGENT", f"{urgent_count:,}", f"Of {len(analysis_data):,} analyzed narratives")
    with columns[3]:
        metric("NEGATIVE SENTIMENT", f"{negative_share:.0%}", "Lexicon estimate, analyzed sample")

    st.write("")
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.subheader("Largest complaint categories")
        category_counts = dataset["category"].value_counts().head(9).rename_axis("Category").to_frame("Tickets")
        st.bar_chart(category_counts, color="#83b8c6", horizontal=True)
    with right:
        st.subheader("What needs attention")
        urgent_rows = analysis_data.assign(
            urgency=sentiment_rows["urgency"].values,
            sentiment=sentiment_rows["sentiment"].values,
        )
        priority = urgent_rows[urgent_rows["urgency"].isin(["Urgent", "High"])].head(6)
        if priority.empty:
            st.info("No urgent signals found in the analyzed sample.")
        else:
            for _, row in priority.iterrows():
                st.markdown(f"**{row['urgency']}** · {row['category']}")
                st.caption(str(row["narrative"])[:180])
                st.divider()
    st.caption("Urgency and sentiment are heuristic signals; review before making customer-impacting decisions.")

elif view == "Live triage":
    page_heading("INTAKE / TRIAGE", "Turn a narrative into a route", "Classification, urgency, sentiment, and team suggestion in one pass.")
    sample_text = "My payment was deducted but the transaction is still pending."
    with st.form("triage_form"):
        ticket_text = st.text_area("Complaint narrative", value=sample_text, height=150, max_chars=10000)
        submitted = st.form_submit_button("Analyze complaint", type="primary", icon=":material/auto_awesome:")
    if submitted:
        if not ticket_text.strip():
            st.warning("Enter a complaint narrative to analyze.")
        else:
            category, score = classify_ticket(ticket_text, model)
            sentiment, sentiment_score = analyze_sentiment(ticket_text)
            urgency, signals = assess_urgency(ticket_text)
            department = suggest_department(category, ticket_text)
            st.markdown("### Suggested handling")
            first, second, third, fourth = st.columns(4)
            with first:
                metric("CATEGORY", category, f"Relative model score · {score:.0%}")
            with second:
                metric("URGENCY", urgency, ", ".join(signals) if signals else "No urgency rules matched")
            with third:
                metric("SENTIMENT", sentiment, f"VADER compound · {sentiment_score:+.2f}")
            with fourth:
                metric("SUGGESTED TEAM", department, "Keyword-based routing rule")
            st.info("Predictions are decision support, not a substitute for agent review. Model scores are not calibrated probabilities.")

    st.divider()
    st.subheader("Analyze a batch")
    batch_file = st.file_uploader("Upload a CSV with a narrative or Consumer complaint narrative column", type=["csv"], key="batch")
    if batch_file is not None:
        if batch_file.size > MAX_UPLOAD_SIZE_BYTES:
            st.error("This CSV exceeds the 25 MB upload limit. Export a smaller batch and try again.")
        else:
            try:
                batch_input = pd.read_csv(batch_file, nrows=10001)
                narrative_col = find_column(batch_input, "narrative")
                if narrative_col is None:
                    st.error("No complaint narrative column was found.")
                else:
                    if len(batch_input) > 10000:
                        st.warning("Only the first 10,000 rows are analyzed in this batch.")
                        batch_input = batch_input.iloc[:10000]
                    batch_texts = batch_input[narrative_col].fillna("").astype(str).tolist()
                    output = batch_analyze(batch_texts, model)
                    export_frame = pd.concat(
                        [batch_input.reset_index(drop=True), output], axis=1
                    )
                    st.dataframe(export_frame, width="stretch", hide_index=True)
                    safe_export = prepare_csv_export(export_frame)
                    st.download_button(
                        "Download analyzed CSV",
                        data=safe_export.to_csv(index=False).encode("utf-8"),
                        file_name="ticket_intelligence_results.csv",
                        mime="text/csv",
                        icon=":material/download:",
                    )
            except Exception:
                st.error("Could not analyze this batch. Confirm it is a valid CSV with a complaint narrative column.")

elif view == "Recurring issues":
    page_heading("TRENDS / TOPICS", "Find repeating complaint patterns", "TF-IDF features and K-Means surface clusters for analyst review.")
    topic_count = st.slider("Topic clusters", min_value=2, max_value=10, value=5)
    topics = discover_topics(analysis_data, topic_count)
    if topics.empty:
        st.warning("Add more complaint narratives to discover recurring issues.")
    else:
        chart_data = topics.set_index("keywords")[["count"]]
        left, right = st.columns([1, 1.2], gap="large")
        with left:
            st.subheader("Cluster volume")
            st.bar_chart(chart_data, color="#e77d61", horizontal=True)
        with right:
            st.subheader("Themes to investigate")
            for _, topic in topics.iterrows():
                st.markdown(f"**{topic['keywords']}** · {topic['count']:,} complaints ({topic['share']:.0%})")
                st.caption(f"Example: {topic['example'][:200]}")
                st.divider()
    st.caption("Clusters are unsupervised groupings, not verified root causes. Topic labels are the strongest TF-IDF terms.")

else:
    page_heading("MODEL / BASELINES", "Inspect the classification setup", "Transparent classical NLP baselines before any production deployment.")
    model_name = st.selectbox(
        "Classifier",
        model_options,
        index=model_options.index(model_name),
        key="model_name",
    )

    st.subheader("Held-out evaluation")
    evaluation = cached_evaluation(training_data, model_name)
    if evaluation["status"] == "ok":
        metric_columns = st.columns(4)
        with metric_columns[0]:
            st.metric(
                "Accuracy",
                f"{evaluation['accuracy']:.1%}",
                help="Share of held-out complaints assigned the correct category.",
            )
        with metric_columns[1]:
            st.metric(
                "Balanced accuracy",
                f"{evaluation['balanced_accuracy']:.1%}",
                help="Average recall across categories; useful when category counts differ.",
            )
        with metric_columns[2]:
            st.metric(
                "Macro F1",
                f"{evaluation['macro_f1']:.1%}",
                help="Unweighted average F1 across categories.",
            )
        with metric_columns[3]:
            st.metric("Holdout complaints", f"{evaluation['holdout_rows']:,}")

        split_description = (
            "stratified by category"
            if evaluation["stratified"]
            else "random because some categories are too small to stratify"
        )
        st.caption(
            f"Deterministic 80/20 split, {split_description}: "
            f"{evaluation['training_rows']:,} fit rows, "
            f"{evaluation['holdout_rows']:,} held-out rows, "
            f"{evaluation['category_count']:,} categories."
        )

        if dataset_source != "Bundled illustrative dataset":
            st.warning(
                "These scores describe this holdout sample only. Use a representative, "
                "time-separated test set before making deployment decisions."
            )

        log_key = f"{model_name}:{evaluation_fingerprint(training_data)}"
        evaluation_log = st.session_state.setdefault("model_evaluation_log", {})
        evaluation_log.setdefault(
            log_key,
            {
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "model": model_name,
                "data_source": dataset_source,
                "accuracy": evaluation["accuracy"],
                "balanced_accuracy": evaluation["balanced_accuracy"],
                "macro_f1": evaluation["macro_f1"],
                "weighted_f1": evaluation["weighted_f1"],
                "training_rows": evaluation["training_rows"],
                "holdout_rows": evaluation["holdout_rows"],
                "category_count": evaluation["category_count"],
                "split": "stratified" if evaluation["stratified"] else "random",
            },
        )

        st.subheader("Evaluation log")
        log_frame = pd.DataFrame(evaluation_log.values()).sort_values(
            ["evaluated_at_utc", "model"], ascending=[False, True]
        )
        st.dataframe(log_frame, width="stretch", hide_index=True)
        st.download_button(
            "Download evaluation log",
            data=log_frame.to_csv(index=False).encode("utf-8"),
            file_name="model_evaluation_log.csv",
            mime="text/csv",
            icon=":material/download:",
        )
        st.caption(
            "The log contains metrics and row counts, not complaint text. It lives in this "
            "browser session; download it before closing the app."
        )
    else:
        st.warning(evaluation["reason"])

    st.write("")
    model_left, model_right = st.columns([1, 1], gap="large")
    with model_left:
        st.subheader("Pipeline")
        st.markdown("1. Normalize complaint narratives and category labels")
        st.markdown("2. Convert text to unigram + bigram TF-IDF features")
        st.markdown("3. Reserve 20% of the sampled data for a deterministic holdout")
        st.markdown(f"4. Fit **{model_name}** on the remaining rows and score the holdout")
        st.info("The bundled corpus is illustrative and intentionally small. It is not the CFPB database and does not produce a meaningful benchmark.")
    with model_right:
        st.subheader("Training snapshot")
        snapshot = training_data["category"].value_counts().head(12).rename_axis("Category").reset_index(name="Examples")
        st.dataframe(snapshot, width="stretch", hide_index=True)
        st.caption(f"Training rows: {len(training_data):,} of {len(dataset):,}. Labels: {training_data['category'].nunique():,}.")
    st.subheader("Model notes")
    st.markdown("- **Logistic Regression:** interpretable linear baseline with class balancing.")
    st.markdown("- **Linear SVM:** strong fit for sparse, high-dimensional text.")
    st.markdown("- **Naive Bayes:** fast probabilistic baseline for word-count features.")
    st.markdown("- **Sentiment:** VADER lexicon score; useful for screening, not a clinical or calibrated measure.")
    st.markdown("- **Urgency and routing:** explicit keyword rules that should be tuned with support-team policy.")

st.markdown("<br><div class='eyebrow'>SIGNAL DESK · INTERNAL DECISION SUPPORT</div>", unsafe_allow_html=True)
