import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Sample training data
data = {
    'text': ['I love this product', 'This is terrible', 'Absolutely amazing!', 'I hate it', 'Not bad', 'Could be better', 'Excellent quality', 'Worst ever'],
    'label': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'negative']
}
df = pd.DataFrame(data)

# Convert text to vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X, y)

# Save the model and vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model and vectorizer saved successfully.")
# train_model.py
