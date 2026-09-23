"""
Spam Filtering Project
----------------------
A from-scratch Bernoulli Naive Bayes spam classifier.

Input file:
    spam_project.xlsx

Required columns:
    Label -> "spam" or "ham"
    Mail  -> message text

Output:
    Results2.xlsx
"""

import math
import re
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_FILE = "spam_project.xlsx"
RESULT_FILE = "Results2.xlsx"
TEST_SIZE = 0.25
RANDOM_STATE = 42
THRESHOLD = 0.60


def tokenize(text):
    """Convert a message into normalized word tokens."""
    text = str(text).lower()
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)


def prepare_data(filename):
    """Load and validate the dataset."""
    data = pd.read_excel(filename)

    required = {"Label", "Mail"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = data[["Label", "Mail"]].dropna().copy()
    data["Label"] = data["Label"].astype(str).str.strip().str.lower()
    data["Mail"] = data["Mail"].astype(str)

    invalid = set(data["Label"]) - {"spam", "ham"}
    if invalid:
        raise ValueError(
            f"Label column must contain only 'spam' or 'ham'. "
            f"Unexpected labels: {sorted(invalid)}"
        )

    if len(data) < 2:
        raise ValueError("The dataset must contain at least two messages.")

    return data.reset_index(drop=True)


def split_data(data):
    """Split the dataset into training and testing sets."""
    train_data, test_data = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["Label"],
    )
    return train_data.reset_index(drop=True), test_data.reset_index(drop=True)


def train_model(train_data):
    """
    Train a Bernoulli Naive Bayes model.

    Each vocabulary word is represented by whether it occurs in a message,
    not by the number of times it occurs.
    """
    labels = train_data["Label"]

    spam_count = (labels == "spam").sum()
    ham_count = (labels == "ham").sum()
    total = len(train_data)

    spam_prior = spam_count / total
    ham_prior = ham_count / total

    # Build a vocabulary from the training messages only.
    vocabulary = set()
    for mail in train_data["Mail"]:
        vocabulary.update(tokenize(mail))

    # Ignore extremely short tokens and a common stop word.
    vocabulary = {
        word for word in vocabulary
        if len(word) >= 3 and word != "the"
    }

    spam_word_count = {word: 0 for word in vocabulary}
    ham_word_count = {word: 0 for word in vocabulary}

    # Count the number of messages containing each word.
    for _, row in train_data.iterrows():
        words = set(tokenize(row["Mail"]))

        if row["Label"] == "spam":
            for word in words & vocabulary:
                spam_word_count[word] += 1
        else:
            for word in words & vocabulary:
                ham_word_count[word] += 1

    # Laplace-smoothed conditional probabilities:
    # P(word present | spam) and P(word present | ham)
    prob_df = pd.DataFrame({
        "SpamCount": pd.Series(spam_word_count),
        "HamCount": pd.Series(ham_word_count),
    }).fillna(0)

    prob_df["Pr(W|S)"] = (
        (prob_df["SpamCount"] + 1) /
        (spam_count + 2)
    )

    prob_df["Pr(W|H)"] = (
        (prob_df["HamCount"] + 1) /
        (ham_count + 2)
    )

    return {
        "prob_df": prob_df,
        "spam_prior": spam_prior,
        "ham_prior": ham_prior,
        "spam_count": int(spam_count),
        "ham_count": int(ham_count),
        "vocabulary": vocabulary,
    }


def word_spam_probability(word, model):
    """
    Calculate P(Spam | word) using Bayes' theorem.
    """
    prob_df = model["prob_df"]
    spam_prior = model["spam_prior"]
    ham_prior = model["ham_prior"]

    if word not in prob_df.index:
        return None

    p_word_spam = prob_df.loc[word, "Pr(W|S)"]
    p_word_ham = prob_df.loc[word, "Pr(W|H)"]

    denominator = (
        p_word_spam * spam_prior +
        p_word_ham * ham_prior
    )

    if denominator == 0:
        return None

    return (p_word_spam * spam_prior) / denominator


def spam_value(message, model):
    """
    Combine word-level evidence into P(Spam | message).

    This is calculated in log-space to avoid numerical underflow.
    """
    words = set(tokenize(message))
    prob_df = model["prob_df"]

    # Use only known words.
    known_words = [word for word in words if word in prob_df.index]

    if not known_words:
        # With no known words, use the class prior.
        return model["spam_prior"]

    log_spam = math.log(model["spam_prior"])
    log_ham = math.log(model["ham_prior"])

    for word in known_words:
        p_ws = prob_df.loc[word, "Pr(W|S)"]
        p_wh = prob_df.loc[word, "Pr(W|H)"]

        # Probability that the word is absent.
        # This is the Bernoulli Naive Bayes contribution.
        log_spam += math.log(p_ws)
        log_ham += math.log(p_wh)

    # Stable conversion of two log probabilities into a probability.
    difference = log_ham - log_spam
    if difference > 700:
        return 0.0
    if difference < -700:
        return 1.0

    return 1 / (1 + math.exp(difference))


def classify(message, model, threshold=THRESHOLD):
    """Return spam/ham classification and the spam probability."""
    probability = spam_value(message, model)
    label = "spam" if probability >= threshold else "ham"
    return label, probability


def evaluate(test_data, model):
    """Evaluate the trained model on unseen test messages."""
    results = test_data.copy()
    predictions = []
    probabilities = []

    for message in results["Mail"]:
        label, probability = classify(message, model)
        predictions.append(label)
        probabilities.append(round(probability, 9))

    results["spam_value"] = probabilities
    results["results"] = predictions

    accuracy = (results["results"] == results["Label"]).mean() * 100
    return results, round(accuracy, 2)


def main():
    data = prepare_data(DATA_FILE)
    train_data, test_data = split_data(data)

    model = train_model(train_data)
    results, accuracy = evaluate(test_data, model)

    results.to_excel(RESULT_FILE, index=False)

    print("=" * 55)
    print("SPAM FILTERING PROJECT")
    print("=" * 55)
    print(f"Total messages : {len(data)}")
    print(f"Training data  : {len(train_data)}")
    print(f"Testing data   : {len(test_data)}")
    print(f"Spam messages  : {model['spam_count']}")
    print(f"Ham messages   : {model['ham_count']}")
    print(f"P(Spam)        : {model['spam_prior']:.4f}")
    print(f"P(Ham)         : {model['ham_prior']:.4f}")
    print(f"Threshold      : {THRESHOLD}")
    print(f"Model accuracy : {accuracy}%")
    print(f"Results saved  : {RESULT_FILE}")
    print("=" * 55)

    # Example user test:
    # message = input("Enter a message to classify: ")
    # label, probability = classify(message, model)
    # print(f"Prediction: {label} ({probability:.2%} spam probability)")


if __name__ == "__main__":
    main()
