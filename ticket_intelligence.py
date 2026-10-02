from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


_SENTIMENT = SentimentIntensityAnalyzer()
_MODEL_TYPES = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "Linear SVM": LinearSVC(class_weight="balanced", random_state=42),
    "Naive Bayes": MultinomialNB(),
}
_COLUMN_ALIASES = {
    "narrative": (
        "consumer complaint narrative",
        "complaint narrative",
        "narrative",
        "text",
        "description",
    ),
    "date_received": ("date received", "date_received", "created_at", "date"),
    "company": ("company", "company name", "organization"),
    "product": ("product", "category", "issue", "sub-product"),
}


def _column_map(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column).strip().casefold(): column for column in frame.columns}


def find_column(frame: pd.DataFrame, field: str) -> str | None:
    """Find a CFPB or common equivalent column without assuming header casing."""
    columns = _column_map(frame)
    for alias in _COLUMN_ALIASES[field]:
        if alias.casefold() in columns:
            return columns[alias.casefold()]
    return None


def normalize_dataset(frame: pd.DataFrame, label_column: str | None = None) -> pd.DataFrame:
    """Convert CFPB-style CSV fields into the app's small canonical schema."""
    narrative_column = find_column(frame, "narrative")
    if narrative_column is None:
        raise ValueError(
            "Could not find complaint text. Expected a 'Consumer complaint narrative' or 'narrative' column."
        )

    if label_column is None:
        label_column = find_column(frame, "product")
    elif label_column not in frame.columns:
        label_column = _column_map(frame).get(str(label_column).casefold())
    if label_column is None:
        raise ValueError("Choose a category column, such as 'Product' or 'Issue'.")

    date_column = find_column(frame, "date_received")
    company_column = find_column(frame, "company")
    normalized = pd.DataFrame(
        {
            "narrative": frame[narrative_column].fillna("").astype(str).str.strip(),
            "category": frame[label_column].fillna("Uncategorized").astype(str).str.strip(),
            "product": frame[label_column].fillna("Uncategorized").astype(str).str.strip(),
            "date_received": (
                pd.to_datetime(frame[date_column], errors="coerce")
                if date_column is not None
                else pd.NaT
            ),
            "company": (
                frame[company_column].fillna("Unknown").astype(str).str.strip()
                if company_column is not None
                else "Unknown"
            ),
        }
    )
    normalized = normalized[
        normalized["narrative"].ne("") & normalized["category"].ne("")
    ].copy()
    if normalized.empty:
        raise ValueError("No rows contain both complaint text and a category label.")
    return normalized.reset_index(drop=True)


def train_classifier(
    frame: pd.DataFrame, model_name: str = "Logistic Regression"
) -> Pipeline | None:
    """Fit a TF-IDF baseline; return None if the data has only one usable label."""
    usable = frame.dropna(subset=["narrative", "category"])
    usable = usable[
        usable["narrative"].astype(str).str.strip().ne("")
        & usable["category"].astype(str).str.strip().ne("")
    ]
    if usable["category"].nunique() < 2:
        return None
    estimator = clone(_MODEL_TYPES.get(model_name, _MODEL_TYPES["Logistic Regression"]))
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=35000,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", estimator),
        ]
    )
    pipeline.fit(usable["narrative"].astype(str), usable["category"].astype(str))
    return pipeline


def evaluate_classifier(
    frame: pd.DataFrame, model_name: str = "Logistic Regression"
) -> dict[str, Any]:
    """Evaluate a classifier on a deterministic holdout not used during fitting."""
    usable = frame.dropna(subset=["narrative", "category"])
    usable = usable[
        usable["narrative"].astype(str).str.strip().ne("")
        & usable["category"].astype(str).str.strip().ne("")
    ]
    labels = usable["category"].astype(str)
    label_counts = labels.value_counts()

    if len(usable) < 10:
        return {
            "status": "unavailable",
            "reason": "At least 10 labeled complaints are required for a holdout evaluation.",
        }
    if len(label_counts) < 2:
        return {
            "status": "unavailable",
            "reason": "At least two distinct category labels are required for evaluation.",
        }

    holdout_size = int(np.ceil(len(usable) * 0.2))
    training_size = len(usable) - holdout_size
    stratify = None
    if (
        label_counts.min() >= 2
        and holdout_size >= len(label_counts)
        and training_size >= len(label_counts)
    ):
        stratify = labels

    training_text, holdout_text, training_labels, holdout_labels = train_test_split(
        usable["narrative"].astype(str),
        labels,
        test_size=holdout_size,
        random_state=42,
        stratify=stratify,
    )
    if holdout_labels.nunique() < 2:
        return {
            "status": "unavailable",
            "reason": "The holdout contains fewer than two categories; add more labeled examples.",
        }

    evaluation_data = pd.DataFrame(
        {"narrative": training_text, "category": training_labels}
    )
    evaluation_model = train_classifier(evaluation_data, model_name)
    if evaluation_model is None:
        return {
            "status": "unavailable",
            "reason": "The training split contains fewer than two categories; add more labeled examples.",
        }
    predictions = evaluation_model.predict(holdout_text)
    all_labels = sorted(labels.unique())

    return {
        "status": "ok",
        "accuracy": float(accuracy_score(holdout_labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(holdout_labels, predictions)
        ),
        "macro_f1": float(
            f1_score(
                holdout_labels,
                predictions,
                labels=all_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                holdout_labels,
                predictions,
                labels=all_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "training_rows": len(training_text),
        "holdout_rows": len(holdout_text),
        "category_count": len(all_labels),
        "stratified": stratify is not None,
    }


def classify_ticket(text: str, model: Pipeline | None) -> tuple[str, float]:
    """Return the predicted label and its relative model score (not calibrated)."""
    if model is None:
        return "Needs labeled data", 0.0
    prediction = str(model.predict([text])[0])
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        scores = classifier.predict_proba(model.named_steps["tfidf"].transform([text]))[0]
    else:
        decision = np.asarray(classifier.decision_function(model.named_steps["tfidf"].transform([text])))
        if decision.ndim == 1:
            positive = 1.0 / (1.0 + np.exp(-np.clip(decision[0], -40, 40)))
            scores = np.array([1.0 - positive, positive])
        else:
            shifted = decision[0] - np.max(decision[0])
            exp_scores = np.exp(shifted)
            scores = exp_scores / exp_scores.sum()
    label_index = list(classifier.classes_).index(prediction)
    return prediction, float(scores[label_index])


_URGENT_TERMS = (
    "unauthorized",
    "stolen",
    "identity theft",
    "fraud",
    "account takeover",
    "money missing",
    "locked out",
    "cannot access",
    "can't access",
    "eviction",
    "foreclosure",
    "court date",
)
_HIGH_TERMS = (
    "overdue",
    "late fee",
    "deadline",
    "charged twice",
    "still pending",
    "weeks",
    "days",
    "no response",
    "repeatedly",
)


def assess_urgency(text: str) -> tuple[str, list[str]]:
    normalized = text.casefold()
    urgent_matches = [term for term in _URGENT_TERMS if term in normalized]
    if urgent_matches:
        return "Urgent", urgent_matches
    high_matches = [term for term in _HIGH_TERMS if term in normalized]
    if high_matches:
        return "High", high_matches
    return "Normal", []


def analyze_sentiment(text: str) -> tuple[str, float]:
    compound = float(_SENTIMENT.polarity_scores(text)["compound"])
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return label, compound


def suggest_department(category: str, text: str) -> str:
    context = f"{category} {text}".casefold()
    routes = (
        (("payment", "transfer", "deposit", "refund", "transaction", "money"), "Payments Support"),
        (("fraud", "dispute", "identity theft", "unauthorized", "stolen"), "Fraud & Disputes"),
        (("card", "billing", "statement", "fee", "interest"), "Cards & Billing"),
        (("loan", "mortgage", "student", "lending", "collection", "debt"), "Lending & Servicing"),
        (("access", "login", "password", "sign in", "account"), "Account Services"),
    )
    for terms, department in routes:
        if any(term in context for term in terms):
            return department
    return "Customer Care"


def discover_topics(frame: pd.DataFrame, requested_topics: int = 5) -> pd.DataFrame:
    """Cluster narratives and label each cluster with its strongest TF-IDF terms."""
    texts = frame["narrative"].fillna("").astype(str).str.strip()
    texts = texts[texts.ne("")]
    if len(texts) < 3 or texts.nunique() < 2:
        return pd.DataFrame(columns=["topic", "count", "share", "keywords", "example"])

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=8000,
        min_df=1,
    )
    matrix = vectorizer.fit_transform(texts)
    topic_count = min(max(2, requested_topics), len(texts), matrix.shape[1])
    if topic_count < 2:
        return pd.DataFrame(columns=["topic", "count", "share", "keywords", "example"])

    clusters = KMeans(n_clusters=topic_count, n_init=10, random_state=42).fit_predict(matrix)
    terms = vectorizer.get_feature_names_out()
    rows: list[dict[str, Any]] = []
    for cluster_id in sorted(set(clusters)):
        positions = np.flatnonzero(clusters == cluster_id)
        mean_weights = np.asarray(matrix[positions].mean(axis=0)).ravel()
        top_terms = terms[np.argsort(mean_weights)[-3:][::-1]].tolist()
        rows.append(
            {
                "topic": f"Theme {len(rows) + 1}",
                "count": int(len(positions)),
                "share": float(len(positions) / len(texts)),
                "keywords": ", ".join(top_terms),
                "example": str(texts.iloc[positions[0]]),
            }
        )
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def batch_analyze(texts: list[str], model: Pipeline | None) -> pd.DataFrame:
    results = []
    for text in texts:
        category, score = classify_ticket(text, model)
        sentiment, sentiment_score = analyze_sentiment(text)
        urgency, signals = assess_urgency(text)
        results.append(
            {
                "category": category,
                "model_score": score,
                "sentiment": sentiment,
                "sentiment_score": sentiment_score,
                "urgency": urgency,
                "department": suggest_department(category, text),
                "urgency_signals": ", ".join(signals),
            }
        )
    return pd.DataFrame(results)
