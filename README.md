# CodSoft-
Task 1: Movie Genre Classification 
# Movie Genre Classification using Machine Learning
# TF-IDF + Naive Bayes (LinkedIn Ready)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# -----------------------------
# Dataset (can be replaced by CSV)
# -----------------------------
data = {
    "plot": [
        "A young boy learns magic in a wizard school",
        "A hero fights enemies and saves the city",
        "Two people fall in love during college life",
        "Friends go on a funny road trip",
        "Aliens attack earth and humans fight back",
        "A detective investigates a murder mystery",
        "A soldier fights in a dangerous war",
        "A magical kingdom with dragons and powers"
    ],
    "genre": [
        "Fantasy", "Action", "Romance", "Comedy",
        "SciFi", "Thriller", "Action", "Fantasy"
    ]
}

df = pd.DataFrame(data)

X = df["plot"]
y = df["genre"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -----------------------------
# ML Pipeline
# -----------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("nb", MultinomialNB())
])

# -----------------------------
# Train Model
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# Accuracy
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# -----------------------------
# Predict New Movie
# -----------------------------
def predict_genre(plot):
    return model.predict([plot])[0]

sample_plot = "A hero saves people from dangerous enemies"
print("Predicted Genre:", predict_genre(sample_plot))
