# Spam Filtering Project

A from-scratch **Bernoulli Naive Bayes** spam classifier with both a command-line implementation and a Tkinter graphical user interface.

## Team Members

- Rajarshi Dey
- Deepshi Saha
- Ayush Banerjee

## Project Overview

This project demonstrates a supervised machine-learning approach to classifying text messages as **spam** or **ham**.

The project contains:

- A Python command-line implementation of the classifier.
- A Tkinter GUI for interactive message classification.
- Excel-based input and output.
- A 75/25 stratified train-test split.
- Laplace smoothing for conditional probabilities.
- Log-space probability calculation to reduce numerical underflow.

## Machine Learning Concept

The project explores **Bernoulli Naive Bayes**, a probabilistic supervised classification algorithm.

The implementation treats a vocabulary word mainly as a binary feature: whether the word occurs in a message. For each word, the model estimates:

- `P(word present | spam)`
- `P(word present | ham)`

The model combines this evidence with the prior probabilities of the two classes.

## Dataset

The program expects:

```text
spam_project.xlsx
```

with the following columns:

| Column | Description |
|---|---|
| `Label` | `spam` or `ham` |
| `Mail` | Message text |

## Project Structure

```text
Spam-Filtering-Project/
│
├── Spam_Filtering_Project.py
├── Spam_Filtering_GUI.py
├── spam_project.xlsx
├── README.md
├── requirements.txt
├── .gitignore
│
└── Results/
    └── (generated result files)
```

## Installation

Make sure Python is installed, then open the project folder in VS Code.

Install the required packages:

```powershell
python -m pip install pandas scikit-learn openpyxl
```

Tkinter is normally included with standard Python installations on Windows.

## Running the Command-Line Version

```powershell
python Spam_Filtering_Project.py
```

The program:

1. Loads and validates the dataset.
2. Splits it into training and testing data.
3. Builds the vocabulary from the training data.
4. Calculates class priors and word probabilities.
5. Evaluates the model on unseen test messages.
6. Saves the predictions to `Results2.xlsx`.

## Running the GUI

```powershell
python Spam_Filtering_GUI.py
```

Use the GUI in this order:

1. Click **Fit Data**.
2. Click **Train Model**.
3. Click **Test Model**.
4. Enter a message in the text box.
5. Click **Classify Message**.

The GUI displays the predicted class and spam probability.

## Core Parameters

The current implementation uses:

```text
TEST_SIZE = 0.25
RANDOM_STATE = 42
THRESHOLD = 0.60
```

Therefore, 25% of the dataset is reserved for testing, the split is reproducible using random state 42, and a spam probability of at least 0.60 produces a spam prediction.

## Mathematical Approach

Class priors:

```text
P(Spam) = spam training messages / total training messages
P(Ham)  = ham training messages / total training messages
```

Laplace-smoothed conditional probabilities:

```text
P(W|S) = (SpamCount(W) + 1) / (SpamCount + 2)
P(W|H) = (HamCount(W) + 1) / (HamCount + 2)
```

The implementation combines the probabilities in log-space and converts the resulting log-probability difference into a spam probability.

## Evaluation

The current implementation reports **accuracy** on the held-out test set.

The generated result file contains:

- Original message
- Original label
- Spam probability
- Predicted result

## Important Notes

- The vocabulary is built from training messages only.
- Tokens shorter than three characters and the word `the` are ignored.
- Repeated occurrences of a word in one message do not increase its Bernoulli feature count.
- Laplace smoothing prevents zero conditional probabilities.
- Unknown words are ignored during prediction.
- If a message contains no known vocabulary words, the model falls back to the spam prior.

## Limitations

This is an educational machine-learning project rather than a production-grade email security system.

Current limitations include:

- Simple regular-expression tokenization.
- No stemming or lemmatization.
- Accuracy is the primary evaluation metric.
- No continuous/online model training.
- Performance depends strongly on the dataset.

## Possible Future Improvements
•	Add precision, recall, F1-score, confusion matrix, and ROC-AUC where appropriate.
•	Add stronger text preprocessing such as stop-word handling and normalization.
•	Compare Bernoulli Naive Bayes with Multinomial Naive Bayes or TF-IDF-based classifiers.
•	Add model persistence so a trained model can be reused without retraining.
•	Improve the GUI with clearer validation, visual statistics, and batch prediction.



## License

This repository is intended as an academic/project demonstration. Add a specific open-source license if the team decides to distribute the code under one.
