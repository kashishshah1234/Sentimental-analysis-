import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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
        sentiment = blob.sentiment
        return sentiment.polarity, sentiment.subjectivity
    else:
        return 0, 0


data["Polarity"], data["Subjectivity"] = zip(*data["Cleaned_Description"].apply(get_sentiment))

def get_sentiment_label(polarity):
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

data["Sentiment_Label"] = data["Polarity"].apply(get_sentiment_label)

# Step 7: Sentiment Subjectivity 
average_subjectivity = data["Subjectivity"].mean()
print("Average Subjectivity:", average_subjectivity)

if "Date" in data.columns:
    data_time_series = data.set_index("Date")
    sentiment_time_series = data_time_series.groupby(pd.Grouper(freq='D')).mean()
    sentiment_time_series.plot(y="Polarity", kind="line")
    plt.xlabel('Date')
    plt.ylabel('Average Polarity')
    plt.title('Sentiment Over Time')
    plt.show()

def generate_word_cloud(sentiment_label):
    text = " ".join(data[data["Sentiment_Label"] == sentiment_label]["Cleaned_Description"].fillna(''))
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud for {sentiment_label} Sentiment")
    plt.show()

generate_word_cloud("Positive")
generate_word_cloud("Negative")
generate_word_cloud("Neutral")
