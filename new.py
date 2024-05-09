import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from scipy.special import softmax
from tqdm.notebook import tqdm

plt.style.use('ggplot')

df = pd.read_csv('reviews.csv')

# Set no of entries
df = df.head(20)


# Preprocess text
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


# MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)

# Load the tokenizer and model from the saved directories
tokenizer = AutoTokenizer.from_pretrained("/home/mickey/Downloads/SentimentModel/tokenizer")
model = TFAutoModelForSequenceClassification.from_pretrained("/home/mickey/Downloads/SentimentModel/model")


# Determine polarity for each review
def polarity_scores_roberta(example):
    # Tokenize the input text
    encoded_text = tokenizer(example, return_tensors='tf', truncation=True, max_length=512)

    # Pass the tokenized input to the model
    output = model(**encoded_text)

    # Softmax normalization
    scores = softmax(output[0], axis=1)

    # Extract sentiment scores
    scores_dict = {
        'roberta_neg': scores[0][0],
        'roberta_neu': scores[0][1],
        'roberta_pos': scores[0][2],
    }
    return scores_dict


# Process each row of the dataframe
result = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['text']
        myid = row['id']

        # Calculate sentiment scores
        roberta_result = polarity_scores_roberta(text)
        result[myid] = roberta_result
    except RuntimeError:
        print(f'Broke for id {myid}')

results_df = pd.DataFrame(result).T
results_df = results_df.reset_index().rename(columns={'index': 'id'})
results_df = results_df.merge(df, how='left')

# Define bins for sentiment scores
bins = [-1, -0.5, 0, 0.5, 1]
labels = ['Very Negative', 'Negative', 'Neutral', 'Positive']

# Convert sentiment scores to categorical variables
results_df['sentiment_category'] = pd.cut(results_df['roberta_pos'], bins=bins, labels=labels)

# Count occurrences of sentiment categories
sentiment_counts = results_df['sentiment_category'].value_counts().reset_index()

# Create scatter plot for count of sentiment categories
plt.figure(figsize=(8, 6))
sns.scatterplot(data=sentiment_counts, x='sentiment_category', y='count')
plt.title('Sentiment Category Counts')
plt.xlabel('Sentiment Category')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
