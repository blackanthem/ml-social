from dotenv import load_dotenv
from tweepy import Client, Paginator, Tweet

import os
import math
import tweet_sentiment
import json

load_dotenv()

client = Client(
    bearer_token=os.environ["BEARER_TOKEN"], wait_on_rate_limit=True)

ACA_client = Client(
    bearer_token=os.environ["ACADEMIC_BEARER_TOKEN"], wait_on_rate_limit=True)


# Get maximum allowable tweets per user
def get_all_tweets(id: int):
    all_tweets = []

    for response in Paginator(client.get_users_tweets, id, tweet_fields=["created_at", "text",
                                                                         "source", "public_metrics", "geo", "entities", "conversation_id", "possibly_sensitive", "referenced_tweets"],
                              max_results=100):
        if response.data is not None:
            all_tweets.extend(response.data)

    return all_tweets


# check if tweet contains urls, hashtags, mentions
def get_entities_count(tweet):
    entities = tweet.entities

    if entities is None:
        return {
            "hashtag_count": 0,
            "url_count": 0,
            "mention_count": 0,
        }

    return{
        "hashtag_count": len(entities.get("hashtags", "")),
        "url_count": len(entities.get("urls", "")),
        "mention_count": len(entities.get("mentions", "")),
    }


# check if tweet is retweet
def check_if_retweet(tweet):
    referenced_tweets = tweet.referenced_tweets

    if referenced_tweets is None:
        return 0

    for referenced_tweet in referenced_tweets:
        if referenced_tweet["type"] == "retweeted":
            return 1

    return 0


# calculate h-index
def calculate_h_index(array: list[int]) -> int:
    N = len(array)
    tmp = [0 for _ in range(N+1)]
    for i, v in enumerate(array):
        if v > N:
            tmp[N] += 1
        else:
            tmp[v] += 1

    total = 0
    for i in range(N, -1, -1):
        total += tmp[i]
        if total >= i:
            return i


def get_total_tweets_mentioning_user(username, account_creation_date):
    if username is None or account_creation_date is None:
        return None

    query = f'("@{username}") -from:{username}'

    total_tweets = 0
    for response in Paginator(ACA_client.get_all_tweets_count, query, granularity="day", start_time=account_creation_date):
        total_tweets += response.meta["total_tweet_count"]

    return total_tweets


def get_user_level_features(id: int):
    response = client.get_user(
        id=id, user_fields=["public_metrics", "created_at"])
    if response.data is None:
        return None

    public_metrics = response.data["public_metrics"]

    username = response.data["username"]
    user_id = response.data["id"]
    followers_count = public_metrics["followers_count"]
    following_count = public_metrics["following_count"]
    listed_count = public_metrics["listed_count"]
    tweet_count = public_metrics["tweet_count"]
    account_creation_date = response.data["created_at"]
    social_reputation_score = math.log(
        (1 + followers_count) * (1+followers_count), 10) + math.log(1 + tweet_count) - math.log((1+following_count))

    mentions = get_total_tweets_mentioning_user(
        username, account_creation_date)
    mention_ratio = mentions / tweet_count

    return {
        "username": username,
        "user_id": user_id,
        "tweet_count": tweet_count,
        "listed_count": listed_count,
        "social_reputation_score": social_reputation_score,
        "followers_count": followers_count,
        "following_count": following_count,
        "mention_ratio": mention_ratio,
        "mentions": mentions
    }


def get_tweet_level_features(tweets: list[Tweet]):

    total_likes = total_replies = total_retweets = total_tweets_w_url = total_tweets_w_hashtag = 0
    total_quotes = total_hashtags = total_urls = total_mentions = total_retweeted_posts = total_retweets_plus_quotes = 0
    total_negative_tweets = total_positive_tweets = total_neutral_tweets = 0
    total_tweets = len(tweets)
    retweet_list = []
    quote_list = []
    like_list = []
    reply_list = []
    retweets_plus_quotes_list = []

    for tweet in tweets:

        retweet_count = tweet.public_metrics["retweet_count"]
        total_retweets += retweet_count
        retweet_list.append(retweet_count)

        quote_count = tweet.public_metrics["quote_count"]
        total_quotes += quote_count
        quote_list.append(quote_count)

        retweets_plus_quotes_count = tweet.public_metrics["quote_count"] + \
            tweet.public_metrics["retweet_count"]
        total_retweets_plus_quotes += retweets_plus_quotes_count
        retweets_plus_quotes_list.append(retweets_plus_quotes_count)

        like_count = tweet.public_metrics["like_count"]
        total_likes += like_count
        like_list.append(like_count)

        reply_count = tweet.public_metrics["reply_count"]
        total_replies += reply_count
        reply_list.append(reply_count)

        entities_count = get_entities_count(tweet)

        total_hashtags += entities_count["hashtag_count"]
        if entities_count["hashtag_count"] > 0:
            total_tweets_w_hashtag += 1

        total_urls += entities_count["url_count"]
        if entities_count["url_count"] > 0:
            total_tweets_w_url += 1

        total_mentions += entities_count["mention_count"]

        total_retweeted_posts += check_if_retweet(tweet)

        sentiment = tweet_sentiment.classify(tweet.text)
        if sentiment == "positive":
            total_positive_tweets += 1
        elif sentiment == "negative":
            total_negative_tweets += 1
        else:
            total_neutral_tweets += 1

    # tweet credibility components
    original_content_ratio = (
        total_tweets - total_retweeted_posts) / total_tweets
    quote_ratio = total_quotes / total_tweets
    like_ratio = total_likes / total_tweets
    retweet_ratio = total_retweets / total_tweets
    reply_ratio = total_replies / total_tweets
    url_ratio = total_tweets_w_url / total_tweets
    hashtag_ratio = total_tweets_w_hashtag / total_tweets

    # calculate tweet credibility
    tweet_credibility = ((retweet_ratio + like_ratio + quote_ratio +
                         hashtag_ratio + url_ratio) / 5) * original_content_ratio

    # index score
    retweet_h_index = calculate_h_index(retweet_list)
    like_h_index = calculate_h_index(like_list)
    quote_h_index = calculate_h_index(quote_list)
    reply_h_index = calculate_h_index(reply_list)
    retweets_plus_quotes_h_index = calculate_h_index(retweets_plus_quotes_list)

    sentiment_score = (total_positive_tweets + total_neutral_tweets) / \
        (total_negative_tweets+total_positive_tweets + total_neutral_tweets)

    return {
        "retweet_h_index": retweet_h_index,
        "like_h_index": like_h_index,
        "quote_h_index": quote_h_index,
        "reply_h_index": reply_h_index,
        "retweets_plus_quotes_h_index": retweets_plus_quotes_h_index,
        "tweet_credibility": tweet_credibility,
        "original_content_ratio": original_content_ratio,
        "total_tweets_w_url": total_tweets_w_url,
        "retweet_ratio": retweet_ratio,
        "like_ratio": like_ratio,
        "reply_ratio": reply_ratio,
        "quote_ratio": quote_ratio,
        "url_ratio": url_ratio,
        "hashtag_ratio": hashtag_ratio,
        "sentiment_score": sentiment_score,
        "total_fetched_tweets": total_tweets,
        "base_tweet_features": {
            "total_likes": total_likes,
            "total_replies": total_replies,
            "total_retweets": total_retweets,
            "total_quotes": total_quotes,
            "total_hashtags": total_hashtags,
            "total_tweets_w_hashtag": total_tweets_w_hashtag,
            "total_urls": total_urls,
            "total_tweets_w_url": total_tweets_w_url,
            "total_mentions": total_mentions,
            "total_retweeted_posts": total_retweeted_posts,
            "total_retweets_plus_quotes": total_retweets_plus_quotes,
            "total_negative_tweets": total_negative_tweets,
            "total_positive_tweets": total_positive_tweets,
            "total_neutral_tweets": total_neutral_tweets,
        }
    }


def get_influence_score(features: dict):
    sentiment_score = features["sentiment_score"]
    tweet_credibility = features["tweet_credibility"]
    social_reputation_score = features["social_reputation_score"]
    retweet_h_index = features["retweet_h_index"]
    like_h_index = features["like_h_index"]
    quote_h_index = features["quote_h_index"]

    influence_score = sum([tweet_credibility, social_reputation_score,
                          quote_h_index, sentiment_score, retweet_h_index, like_h_index]) / 6

    features["influence_score"] = influence_score

    return features


def extract(id: int):
    tweets = get_all_tweets(id)
    tweet_level_features = get_tweet_level_features(tweets)

    user_level_features = get_user_level_features(id)

    user_level_features.update(tweet_level_features)

    result = get_influence_score(user_level_features)
    return result


if __name__ == "__main__":
    print(extract(1145137598379282432))
