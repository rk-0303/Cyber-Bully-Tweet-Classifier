import pandas as pd # type: ignore
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from textblob import TextBlob
import pickle

# Load the dataset
df = pd.read_csv("/content/synthetic_cyberbullying_dataset.csv")

# Preprocessing: Text cleaning
def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+|#", "", text)  # Remove mentions and hashtags
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

df['tweet_text'] = df['tweet_text'].apply(preprocess_text)

# Sentiment analysis
def get_sentiment(text):
    analysis = TextBlob(text).sentiment
    return analysis.polarity  # Use polarity as a sentiment score

df['sentiment'] = df['tweet_text'].apply(get_sentiment)

# Prepare features and labels
X_tfidf = TfidfVectorizer(max_features=5000).fit_transform(df['tweet_text']).toarray()
X = pd.DataFrame(X_tfidf)
X['sentiment'] = df['sentiment']  # Add sentiment as a feature

y = df['cyberbullying_type'].apply(lambda x: 1 if x == 'cyberbullying' else 0)  # Binary labels

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Ensure all column names are strings
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

# Train Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

# Output classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))



# Save the trained model and vectorizer
with open("cyberbullying_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

print("Model saved successfully.")
# Load the model and ensure feature consistency
def load_model_and_prepare_vectorizer():
    with open("cyberbullying_model.pkl", "rb") as model_file:
        loaded_model = pickle.load(model_file)
    vectorizer = TfidfVectorizer(max_features=5000)
    vectorizer.fit(df['tweet_text'])  # Ensure the vectorizer aligns with the dataset
    return loaded_model, vectorizer

# Function to preprocess and predict new input text
def predict_text(tweet):
    """Predict whether a tweet is cyberbullying or not."""
    # Preprocess the input text
    tweet_cleaned = preprocess_text(tweet)
    
    # Extract sentiment
    sentiment_score = get_sentiment(tweet_cleaned)
    
    # Transform text using TfidfVectorizer
    tweet_vectorized = vectorizer.transform([tweet_cleaned]).toarray()
    
    # Create a DataFrame for compatibility with training data
    tweet_features = pd.DataFrame(tweet_vectorized, columns=[str(i) for i in range(tweet_vectorized.shape[1])])
    tweet_features['sentiment'] = sentiment_score  # Add sentiment as a feature

    # Predict using the loaded model
    prediction = model.predict(tweet_features)
    return "Cyberbullying" if prediction[0] == 1 else "Not Cyberbullying"

# Load model and vectorizer
model, vectorizer = load_model_and_prepare_vectorizer()

# Test the prediction function
sample_tweets = [
    "I loved your works ",
    "You're the best, keep being awesome!",
    "Go away, nobody wants you here.",
    "your are not a kind person"
]

for tweet in sample_tweets:
    print(f"Tweet: '{tweet}' -> Prediction: {predict_text(tweet)}")