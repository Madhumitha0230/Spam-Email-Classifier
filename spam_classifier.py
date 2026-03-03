import pandas as pd
import numpy as np
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Sample dataset (No need to download anything!)
data = {
    'message': [
        'Congratulations you have won a lottery',
        'Please call me back',
        'Free entry in weekly competition',
        'Are we meeting tomorrow?',
        'Win cash prizes now',
        'Let us discuss the project',
        'Claim your free reward now',
        'Can you send the report?',
        'Exclusive offer just for you',
        'Project meeting at 5 pm',
        'Get free coupons now',
        'Submit assignment before deadline',
        'Earn money fast and easy',
        'Lunch at 1 pm?',
        'Limited time discount offer',
        'Team presentation tomorrow'
    ],
    'label': [
        'spam','ham','spam','ham',
        'spam','ham','spam','ham',
        'spam','ham','spam','ham',
        'spam','ham','spam','ham'
    ]
}


df = pd.DataFrame(data)

# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = text.split()
    text = [word for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

df['cleaned'] = df['message'].apply(clean_text)

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report, confusion_matrix

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Test with new message
message = ["Win money now"]
message_cleaned = [clean_text(message[0])]
message_vector = vectorizer.transform(message_cleaned)
prediction = model.predict(message_vector)

print("\nMessage:", message[0])
print("Prediction:", prediction[0])