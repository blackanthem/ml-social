from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import TweetTokenizer

import re
import string
import pickle


# clean data
stop_words = stopwords.words('english')
translator = str.maketrans('', '', string.punctuation)
tokenizer = TweetTokenizer()
lemmatizer = WordNetLemmatizer()
vectoriser = pickle.load(
    open("../models/tweet_feature_extractor.pickle", "rb"))


def preprocess_tweet(text: str):

    # lowercase string
    text = text.lower()

    # remove stop words
    text = " ".join([word for word in str(
        text).split() if word not in stop_words])

    # remove urls
    text = re.sub('((www.[^s]+)|(https?://[^s]+))', ' ', text)

    # remove punctuations
    text = text.translate(translator)

    # remove repeating characters
    text = re.sub(r'(.)1+', r'1', text)

    # remove numbers
    text = re.sub('[0-9]+', '', text)

    # tokenize text
    text: list[str] = tokenizer.tokenize(text)

    # normalize with lemmatizer
    tokens = []
    for token, tag in pos_tag(text):
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        token = lemmatizer.lemmatize(token, pos)
        tokens.append(token)

    text = tokens
    text = " ".join(text)

    # extract features
    text = vectoriser.transform([text])

    return text


classifier = pickle.load(
    open("../models/twitter_sentiment_model.pickle", "rb"))


def get_tweet_sentiment(tweet: str):
    processed_tweet = preprocess_tweet(tweet)
    result = classifier.predict(processed_tweet)[0]

    if result == 1:
        return "positive"
    elif result == -1:
        return "negative"
    else:
        return "neutral"


# print(get_tweet_sentiment(
#     "through our vote ensure govt need and deserve anupam kher responds modis appeal for the 2019 elections"))
