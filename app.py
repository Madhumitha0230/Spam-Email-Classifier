import streamlit as st
import pandas as pd
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download stopwords only if not available
try:
    from nltk.corpus import stopwords
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords

# Sample dataset
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

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

df['cleaned'] = df['message'].apply(clean_text)

# Feature extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

# Train model
model = MultinomialNB()
model.fit(X, y)

# Streamlit UI
st.set_page_config(page_title="Spam Classifier", page_icon="📧")

st.title("📧 Spam Email Classifier")
st.write("Enter a message to check whether it is Spam or Not Spam.")

user_input = st.text_area("Enter your message here:")

if st.button("Check Message"):
    if user_input.strip() == "":
        st.warning("⚠ Please enter a message.")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == "spam":
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is NOT Spam.")

st.markdown("---")
st.caption("Built using Machine Learning & Streamlit")