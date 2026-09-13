# Sentiment-Analysis-Using-NLP

## Project Overview

This project focuses on developing a **Sentiment Analysis system using Natural Language Processing (NLP) and Machine Learning**. The objective is to automatically classify textual reviews as either **Positive** or **Negative**.

The project uses the **IMDb Movie Reviews dataset**, applies text preprocessing techniques to clean the reviews, converts the text into numerical features using **TF-IDF**, and trains machine-learning models for sentiment classification.

The primary model used is **Logistic Regression**, with **Multinomial Naive Bayes** used as a comparison model.

---

## Objectives

The main objectives of this project are to:

* Select and prepare a suitable textual sentiment dataset.
* Perform text preprocessing and noise removal.
* Convert textual data into numerical representations.
* Train machine-learning models for sentiment classification.
* Evaluate model performance using standard classification metrics.
* Analyze the model's practical applications and limitations.
* Identify possible improvements for future development.

---

## Dataset

### IMDb Movie Reviews Dataset

The project uses the **IMDb Movie Reviews dataset**, which contains movie reviews labelled according to their sentiment.

### Dataset Features

| Feature     | Description           |
| ----------- | --------------------- |
| `review`    | Textual movie review  |
| `sentiment` | Sentiment label       |
| Positive    | Positive movie review |
| Negative    | Negative movie review |

The sentiment labels are converted into numerical values:

```text
Positive → 1
Negative → 0
```

Before model training, the dataset is checked for missing values and duplicate reviews.

---

# Project Workflow

```text
IMDb Movie Reviews
        ↓
Data Exploration
        ↓
Data Cleaning
        ↓
Text Preprocessing
        ↓
Train/Test Split
        ↓
TF-IDF Feature Extraction
        ↓
Machine Learning Models
        ↓
Predictions
        ↓
Model Evaluation
        ↓
Insights & Recommendations
```

---

# Step 1 — Dataset Selection

The IMDb Movie Reviews dataset was selected because it contains a large collection of labelled textual reviews and is suitable for binary sentiment classification.

The dataset was initially explored to:

* Understand its structure.
* Identify the available columns.
* Check the number of observations.
* Detect missing values.
* Identify duplicate reviews.
* Analyze positive and negative sentiment distribution.

Duplicate reviews and missing records were removed where necessary.

---

# Step 2 — Text Preprocessing

Raw text usually contains unnecessary information that can negatively affect machine-learning models. Therefore, several preprocessing techniques were applied.

### Preprocessing workflow

```text
Raw Review
    ↓
Lowercasing
    ↓
HTML Tag Removal
    ↓
URL Removal
    ↓
Noise/Punctuation Removal
    ↓
Tokenization
    ↓
Stopword Removal
    ↓
Lemmatization
    ↓
Clean Review
```

### Techniques Used

#### 1. Lowercasing

All text was converted to lowercase to ensure that words such as:

```text
Amazing
AMAZING
amazing
```

are treated consistently.

#### 2. HTML Removal

IMDb reviews can contain HTML tags such as:

```text
<br />
```

These tags were removed because they do not provide useful sentiment information.

#### 3. URL Removal

URLs were removed from the reviews because they generally do not contribute to the sentiment classification task.

#### 4. Noise Removal

Unnecessary punctuation, numbers, and special characters were removed.

#### 5. Tokenization

Reviews were divided into individual words or tokens.

For example:

```text
"This movie was amazing"
```

becomes:

```text
["this", "movie", "was", "amazing"]
```

#### 6. Stopword Removal

Common English words that provide limited information for classification were removed.

Examples include:

```text
the
is
was
and
a
```

#### 7. Lemmatization

Words were converted to their base form where appropriate.

For example:

```text
loved → love
movies → movie
```

---

# Step 3 — Feature Extraction

## TF-IDF

**Term Frequency-Inverse Document Frequency (TF-IDF)** was selected to convert textual reviews into numerical representations.

Machine-learning models cannot directly process raw text, so TF-IDF assigns numerical weights to words based on their importance within the dataset.

The project uses:

```python
TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)
```

### Configuration

| Parameter            | Purpose                                   |
| -------------------- | ----------------------------------------- |
| `max_features=10000` | Limits the vocabulary to 10,000 features  |
| `ngram_range=(1,2)`  | Uses unigrams and bigrams                 |
| `min_df=2`           | Removes extremely rare terms              |
| `max_df=0.95`        | Removes terms appearing in most documents |

### Why TF-IDF?

TF-IDF was selected because it:

* Is simple and efficient.
* Works well with traditional machine-learning algorithms.
* Captures important words and phrases.
* Produces interpretable features.
* Is suitable for large text datasets.

Using unigrams and bigrams also allows the model to capture phrases such as:

```text
very good
excellent movie
not good
waste time
highly recommended
```

---

# Step 4 — Model Development

Two machine-learning models were developed and compared.

## 1. Logistic Regression

Logistic Regression was selected as the primary classification model because it performs well for text classification and works efficiently with high-dimensional TF-IDF features.

The model was trained using the TF-IDF training features and corresponding sentiment labels.

```python
logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train_tfidf, y_train)
```

The trained model was then used to predict the sentiment of unseen test reviews.

---

## 2. Multinomial Naive Bayes

Multinomial Naive Bayes was selected as a comparison model because it is commonly used for text classification problems.

```python
nb_model = MultinomialNB()

nb_model.fit(X_train_tfidf, y_train)
```

The performance of both models was compared using the same testing dataset.

---

# Step 5 — Model Evaluation

The models were evaluated using four primary classification metrics:

* Accuracy
* Precision
* Recall
* F1-Score
