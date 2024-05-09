import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
from scipy.special import softmax
from tqdm.notebook import tqdm

plt.style.use('ggplot')

df = pd.read_csv('reviews.csv')

# Set no of entries
df = df.head(100)


# Preprocess text
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


# #Save model
# MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
#
# # Save the tokenizer and model locally
# tokenizer.save_pretrained("/home/mickey/Downloads/SentimentModel/tokenizer")
# model.save_pretrained("/home/mickey/Downloads/SentimentModel/model")

# # Load the tokenizer and model from the saved directories
tokenizer = AutoTokenizer.from_pretrained("/home/mickey/Downloads/SentimentModel/tokenizer")
model = TFAutoModelForSequenceClassification.from_pretrained("/home/mickey/Downloads/SentimentModel/model")


# example = df['text'][50]
# print(example)


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

ress = results_df.head()
print(ress, results_df.columns)

# sns.pairplot(data=results_df, vars=['roberta_neg', 'roberta_neu', 'roberta_pos'],
#              diag_kind='hist', diag_kws={'bins': 20},
#              plot_kws={'alpha': 0.5, 'linewidth': 0.5, 'edgecolor': 'w'})
# plt.show()

# Create subplots for scatter plots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Scatter plot for negative sentiment score
sns.scatterplot(data=results_df, x='roberta_neg', y='roberta_neg', ax=axs[0])
axs[0].set_title('Negative Sentiment Score')
axs[0].set_xlabel('Negative Sentiment Score')
axs[0].set_ylabel('Negative Sentiment Score')

# Scatter plot for neutral sentiment score
sns.scatterplot(data=results_df, x='roberta_neu', y='roberta_neu', ax=axs[1])
axs[1].set_title('Neutral Sentiment Score')
axs[1].set_xlabel('Neutral Sentiment Score')
axs[1].set_ylabel('Neutral Sentiment Score')

# Scatter plot for positive sentiment score
sns.scatterplot(data=results_df, x='roberta_pos', y='roberta_pos', ax=axs[2])
axs[2].set_title('Positive Sentiment Score')
axs[2].set_xlabel('Positive Sentiment Score')
axs[2].set_ylabel('Positive Sentiment Score')

plt.tight_layout()
plt.show()
