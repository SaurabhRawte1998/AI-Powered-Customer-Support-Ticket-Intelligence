import pandas as pd
import pytest

from data import make_sample_data
from ticket_intelligence import (
    analyze_sentiment,
    assess_urgency,
    classify_ticket,
    discover_topics,
    evaluate_classifier,
    normalize_dataset,
    suggest_department,
    train_classifier,
)


@pytest.mark.parametrize("model_name", ["Logistic Regression", "Linear SVM", "Naive Bayes"])
def test_payment_example_gets_payment_category_and_department(model_name):
    data = make_sample_data()
    model = train_classifier(data, model_name)
    text = "My payment was deducted but the transaction is still pending."

    category, score = classify_ticket(text, model)

    assert category == "Payment issue"
    assert score > 0
    assert suggest_department(category, text) == "Payments Support"


def test_cfpb_columns_normalize_with_selected_category():
    source = pd.DataFrame(
        {
            "Consumer complaint narrative": ["A transfer never arrived."],
            "Product": ["Bank account"],
            "Issue": ["Problem with a payment"],
            "Date received": ["2025-02-03"],
            "Company": ["Example Bank"],
        }
    )

    normalized = normalize_dataset(source, label_column="Issue")

    assert normalized.loc[0, "narrative"] == "A transfer never arrived."
    assert normalized.loc[0, "category"] == "Problem with a payment"
    assert normalized.loc[0, "company"] == "Example Bank"
    assert normalized.loc[0, "date_received"] == pd.Timestamp("2025-02-03")


def test_fraud_and_account_lockout_are_urgent():
    assert assess_urgency("My card was stolen and unauthorized charges appeared.")[0] == "Urgent"
    assert assess_urgency("I am locked out and cannot access my account.")[0] == "Urgent"


def test_sentiment_returns_documented_label_and_score():
    label, score = analyze_sentiment("I am furious that the company ignored my complaint.")

    assert label == "Negative"
    assert -1 <= score <= 1


def test_topic_discovery_returns_requested_number_of_clusters():
    topics = discover_topics(make_sample_data(), requested_topics=4)

    assert len(topics) == 4
    assert topics["count"].sum() == len(make_sample_data())
    assert topics["keywords"].str.len().gt(0).all()


@pytest.mark.parametrize("model_name", ["Logistic Regression", "Linear SVM", "Naive Bayes"])
def test_classifier_evaluation_uses_reproducible_holdout_metrics(model_name):
    data = make_sample_data()

    first = evaluate_classifier(data, model_name)
    second = evaluate_classifier(data, model_name)

    assert first == second
    assert first["status"] == "ok"
    assert 0 <= first["accuracy"] <= 1
    assert 0 <= first["balanced_accuracy"] <= 1
    assert 0 <= first["macro_f1"] <= 1
    assert first["training_rows"] + first["holdout_rows"] == len(data)
    assert first["stratified"] is True


def test_training_a_second_pipeline_does_not_mutate_the_first():
    original_data = make_sample_data()
    first_model = train_classifier(original_data, "Logistic Regression")
    second_data = original_data.copy()
    second_data["narrative"] = [
        f"{text} uncommonfeature{index}"
        for index, text in enumerate(second_data["narrative"])
    ]
    train_classifier(second_data, "Logistic Regression")

    assert first_model.predict([original_data.loc[0, "narrative"]])[0] == "Payment issue"
