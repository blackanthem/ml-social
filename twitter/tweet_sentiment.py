from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import TweetTokenizer

import re
import string
import pickle


# clean data
__stop_words = stopwords.words('english')
__translator = str.maketrans('', '', string.punctuation)
__tokenizer = TweetTokenizer()
__lemmatizer = WordNetLemmatizer()
__vectoriser = pickle.load(
    open("models/tweet_feature_extractor.pickle", "rb"))


def preprocess_tweet(text: str):

    # lowercase string
    text = text.lower()

    # remove stop words
    text = " ".join([word for word in str(
        text).split() if word not in __stop_words])

    # remove urls
    text = re.sub('((www.[^s]+)|(https?://[^s]+))', ' ', text)

    # remove punctuations
    text = text.translate(__translator)

    # remove repeating characters
    text = re.sub(r'(.)1+', r'1', text)

    # remove numbers
    text = re.sub('[0-9]+', '', text)

    # tokenize text
    text: list[str] = __tokenizer.tokenize(text)

    # normalize with lemmatizer
    tokens = []
    for token, tag in pos_tag(text):
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        token = __lemmatizer.lemmatize(token, pos)
        tokens.append(token)

    text = tokens
    text = " ".join(text)

    # extract features
    text = __vectoriser.transform([text])

    return text


__classifier = pickle.load(
    open("models/twitter_sentiment_model.pickle", "rb"))


def classify(tweet: str):
    processed_tweet = preprocess_tweet(tweet)
    result = __classifier.predict(processed_tweet)[0]

    if result == 1:
        return "positive"
    elif result == -1:
        return "negative"
    else:
        return "neutral"


# print(get_tweet_sentiment(
#     "through our vote ensure govt need and deserve anupam kher responds modis appeal for the 2019 elections"))
