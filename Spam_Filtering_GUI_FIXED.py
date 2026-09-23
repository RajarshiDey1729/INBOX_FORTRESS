"""
Spam Filtering GUI
------------------
Tkinter interface for the Bernoulli Naive Bayes spam classifier.

Input file:
    spam_project.xlsx

Required columns:
    Label -> "spam" or "ham"
    Mail  -> message text
"""

import math
import re
import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_FILE = "spam_project.xlsx"
RESULT_FILE = "Results_GUI.xlsx"
TEST_SIZE = 0.25
RANDOM_STATE = 42
THRESHOLD = 0.60


def tokenize(text):
    text = str(text).lower()
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)


def load_data():
    data = pd.read_excel(DATA_FILE)

    required = {"Label", "Mail"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = data[["Label", "Mail"]].dropna().copy()
    data["Label"] = data["Label"].astype(str).str.strip().str.lower()
    data["Mail"] = data["Mail"].astype(str)

    invalid = set(data["Label"]) - {"spam", "ham"}
    if invalid:
        raise ValueError(f"Unexpected labels: {sorted(invalid)}")

    return data.reset_index(drop=True)


def split_data(data):
    train_data, test_data = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["Label"],
    )
    return train_data.reset_index(drop=True), test_data.reset_index(drop=True)


def train_model(train_data):
    spam_count = int((train_data["Label"] == "spam").sum())
    ham_count = int((train_data["Label"] == "ham").sum())
    total = len(train_data)

    spam_prior = spam_count / total
    ham_prior = ham_count / total

    vocabulary = set()
    for mail in train_data["Mail"]:
        vocabulary.update(tokenize(mail))

    vocabulary = {
        word for word in vocabulary
        if len(word) >= 3 and word != "the"
    }

    spam_word_count = {word: 0 for word in vocabulary}
    ham_word_count = {word: 0 for word in vocabulary}

    for _, row in train_data.iterrows():
        words = set(tokenize(row["Mail"]))

        if row["Label"] == "spam":
            for word in words & vocabulary:
                spam_word_count[word] += 1
        else:
            for word in words & vocabulary:
                ham_word_count[word] += 1

    prob_df = pd.DataFrame({
        "SpamCount": pd.Series(spam_word_count),
        "HamCount": pd.Series(ham_word_count),
    }).fillna(0)

    prob_df["Pr(W|S)"] = (prob_df["SpamCount"] + 1) / (spam_count + 2)
    prob_df["Pr(W|H)"] = (prob_df["HamCount"] + 1) / (ham_count + 2)

    return {
        "prob_df": prob_df,
        "spam_prior": spam_prior,
        "ham_prior": ham_prior,
        "spam_count": spam_count,
        "ham_count": ham_count,
    }


def spam_value(message, model):
    words = set(tokenize(message))
    prob_df = model["prob_df"]
    known_words = [word for word in words if word in prob_df.index]

    if not known_words:
        return model["spam_prior"]

    log_spam = math.log(model["spam_prior"])
    log_ham = math.log(model["ham_prior"])

    for word in known_words:
        log_spam += math.log(prob_df.loc[word, "Pr(W|S)"])
        log_ham += math.log(prob_df.loc[word, "Pr(W|H)"])

    difference = log_ham - log_spam

    if difference > 700:
        return 0.0
    if difference < -700:
        return 1.0

    return 1 / (1 + math.exp(difference))


def classify(message, model):
    probability = spam_value(message, model)
    label = "SPAM" if probability >= THRESHOLD else "HAM"
    return label, probability


def evaluate(test_data, model):
    results = test_data.copy()
    predictions = []
    probabilities = []

    for message in results["Mail"]:
        label, probability = classify(message, model)
        predictions.append(label.lower())
        probabilities.append(round(probability, 9))

    results["spam_value"] = probabilities
    results["results"] = predictions

    accuracy = (results["results"] == results["Label"]).mean() * 100
    return results, round(accuracy, 2)


class SpamFilterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Spam Filtering Project")
        self.root.geometry("600x520")
        self.root.resizable(False, False)

        self.data = None
        self.train_data = None
        self.test_data = None
        self.model = None

        title = tk.Label(
            root,
            text="SPAM FILTERING PROJECT",
            font=("Arial", 18, "bold")
        )
        title.pack(pady=15)

        self.status_var = tk.StringVar(value="Load and split the dataset first.")

        frame = tk.LabelFrame(root, text="Model Information", padx=15, pady=15)
        frame.pack(fill="x", padx=20, pady=10)

        self.train_label = tk.Label(frame, text="Training data: -")
        self.test_label = tk.Label(frame, text="Testing data: -")
        self.prior_label = tk.Label(frame, text="Class probabilities: -")
        self.accuracy_label = tk.Label(frame, text="Accuracy: -")

        self.train_label.pack(anchor="w")
        self.test_label.pack(anchor="w")
        self.prior_label.pack(anchor="w")
        self.accuracy_label.pack(anchor="w")

        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        tk.Button(
            button_frame, text="Fit Data",
            command=self.fit, width=15
        ).grid(row=0, column=0, padx=5)

        tk.Button(
            button_frame, text="Train Model",
            command=self.train, width=15
        ).grid(row=0, column=1, padx=5)

        tk.Button(
            button_frame, text="Test Model",
            command=self.test, width=15
        ).grid(row=0, column=2, padx=5)

        message_frame = tk.LabelFrame(root, text="Try Your Own Message", padx=10, pady=10)
        message_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.message_box = tk.Text(message_frame, height=6, width=65)
        self.message_box.pack(pady=5)

        tk.Button(
            message_frame,
            text="Classify Message",
            command=self.classify_message,
            width=20
        ).pack(pady=5)

        self.result_label = tk.Label(
            message_frame,
            text="Prediction: -",
            font=("Arial", 12, "bold")
        )
        self.result_label.pack(pady=5)

        tk.Label(root, textvariable=self.status_var).pack(pady=5)

    def fit(self):
        try:
            self.data = load_data()
            self.train_data, self.test_data = split_data(self.data)

            self.train_label.config(
                text=f"Training data: {len(self.train_data)}"
            )
            self.test_label.config(
                text=f"Testing data: {len(self.test_data)}"
            )
            self.accuracy_label.config(text="Accuracy: -")
            self.prior_label.config(text="Class probabilities: -")

            self.status_var.set("Data fitted and split successfully.")
        except Exception as error:
            messagebox.showerror("Error", str(error))

    def train(self):
        if self.train_data is None:
            messagebox.showwarning("Warning", "Click 'Fit Data' first.")
            return

        try:
            self.model = train_model(self.train_data)

            self.prior_label.config(
                text=(
                    f"P(Spam) = {self.model['spam_prior']:.4f}    "
                    f"P(Ham) = {self.model['ham_prior']:.4f}"
                )
            )
            self.status_var.set("Model trained. Ready to test.")
        except Exception as error:
            messagebox.showerror("Error", str(error))

    def test(self):
        if self.model is None or self.test_data is None:
            messagebox.showwarning(
                "Warning",
                "Fit the data and train the model first."
            )
            return

        try:
            results, accuracy = evaluate(self.test_data, self.model)
            results.to_excel(RESULT_FILE, index=False)

            self.accuracy_label.config(
                text=f"Accuracy: {accuracy}%"
            )
            self.status_var.set(
                f"Testing completed. Results saved to {RESULT_FILE}"
            )
        except Exception as error:
            messagebox.showerror("Error", str(error))

    def classify_message(self):
        if self.model is None:
            messagebox.showwarning(
                "Warning",
                "Fit the data and train the model first."
            )
            return

        message = self.message_box.get("1.0", tk.END).strip()

        if not message:
            messagebox.showwarning(
                "Warning",
                "Please enter a message."
            )
            return

        label, probability = classify(message, self.model)

        self.result_label.config(
            text=f"Prediction: {label} | Spam probability: {probability:.2%}"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = SpamFilterGUI(root)
    root.mainloop()
