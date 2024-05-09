from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
from scipy.special import softmax

# MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)

# Load the tokenizer and model from the saved directories
tokenizer = AutoTokenizer.from_pretrained("/home/mickey/Downloads/SentimentModel/tokenizer")
model = TFAutoModelForSequenceClassification.from_pretrained("/home/mickey/Downloads/SentimentModel/model")


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


result = polarity_scores_roberta('It\'s such a wonderful day to be alive today')
print(result)
