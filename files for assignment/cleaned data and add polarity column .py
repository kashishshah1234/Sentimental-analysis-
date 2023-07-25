import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

data = pd.read_excel("sentiment_data1.xlsx", engine="openpyxl")

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = text.replace(r'[^\w\s]', '')
    return text

data["Cleaned_Description"] = data["DESCRIPTION"].apply(preprocess_text)

def get_sentiment(text):
    if isinstance(text, str):
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        if sentiment > 0:
            return "Positive"
        elif sentiment < 0:
            return "Negative"   
        else:
            return "Neutral"
    else:
        return "Neutral" 
data["Polarity"] = data["Cleaned_Description"].apply(get_sentiment)

print(data)

sentiment_counts = data["Polarity"].value_counts()
sentiment_counts.plot(kind="bar", edgecolor='black')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Distribution')
plt.show()

# Save Results to a New Excel File
data.to_excel("sentiment_analysis_results.xlsx", index=False, engine="openpyxl")
