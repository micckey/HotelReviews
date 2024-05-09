import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from scipy.special import softmax

# Load the DataFrame with comments
df = pd.read_csv('reviews.csv')

# Set no of entries
df = df.head(500)


# Function to perform sentiment analysis
def perform_sentiment_analysis(comments):
    # Initialize sentiment analysis model
    # MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    # tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)

    # Load the tokenizer and model from the saved directories
    tokenizer = AutoTokenizer.from_pretrained("/home/mickey/Downloads/SentimentModel/tokenizer")
    model = TFAutoModelForSequenceClassification.from_pretrained("/home/mickey/Downloads/SentimentModel/model")

    # Perform sentiment analysis
    sentiments = []
    for comment in comments:
        encoded_text = tokenizer(comment, return_tensors='tf', truncation=True, max_length=512)
        output = model(**encoded_text)
        scores = softmax(output[0], axis=1)[0]
        sentiment = {
            'roberta_neg': scores[0],
            'roberta_neu': scores[1],
            'roberta_pos': scores[2]
        }
        sentiments.append(sentiment)

    return sentiments


# Function to plot sentiment analysis results
def plot_sentiment_analysis(sentiments):
    # Convert sentiments to DataFrame
    sentiments_df = pd.DataFrame(sentiments)

    # # Plot scatter plot
    # plt.figure(figsize=(12, 6))
    # sns.scatterplot(data=sentiments_df, x='roberta_neg', y='roberta_pos', hue='roberta_neu', palette='coolwarm')
    # plt.title('Sentiment Analysis Scatter Plot')
    # plt.xlabel('Negative Sentiment Score')
    # plt.ylabel('Positive Sentiment Score')
    # plt.legend(title='Neutral Sentiment Score')
    # plt.show()

    # Plot pie chart
    labels = ['Negative', 'Neutral', 'Positive']
    sizes = sentiments_df.mean()
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    explode = (0.1, 0, 0)  # explode 1st slice
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('Sentiment Analysis Pie Chart')
    plt.axis('equal')
    plt.show()


# Main function
def main():
    # print(df['offering_id'])

    # Prompt user for offering ID
    offering_id = input("Enter the Offering ID: ")

    # Clean input (e.g., remove leading/trailing whitespace)
    offering_id = offering_id.strip()

    # Convert offering_id to the same data type as DataFrame column (if necessary)
    offering_id = int(offering_id)

    # Perform comparison
    print(df['offering_id'] == offering_id)

    # Filter comments based on offering ID
    comments = df[df['offering_id'] == offering_id]['text'].tolist()

    if not comments:
        print("No comments found for the specified offering ID.")
        return

    # Perform sentiment analysis
    sentiments = perform_sentiment_analysis(comments)

    # Plot sentiment analysis results
    plot_sentiment_analysis(sentiments)


if __name__ == "__main__":
    main()
