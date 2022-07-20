from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path


import os
import tweepy
import time
import pandas as pd

load_dotenv()


def clear(): return os.system("cls")


def throttle(wait_time):
    """
    Decorator that will throttle a function so that it is called only once every wait_time seconds
    If it is called multiple times, will run only the first time.
    See the test_throttle.py file for examples
    """

    def decorator(function):
        def throttled(*args, **kwargs):
            def call_function():
                return function(*args, **kwargs)

            if time.time() - throttled._last_time_called >= wait_time:
                call_function()
                throttled._last_time_called = time.time()

        throttled._last_time_called = 0
        return throttled

    return decorator


class TwitterStream(tweepy.StreamingClient):
    user_ids = set()

    @throttle(2)
    def print_ids_collected(self):
        clear()
        print(f"Collected {len(self.user_ids)} IDs")

    def save_to_csv(self):
        if(len(self.user_ids) == 0):
            return
        user_ids = list(self.user_ids)
        date = datetime.today().strftime('%Y-%m-%d %H-%M-%S')

        pd.DataFrame({"id": user_ids}).to_csv(
            f"datasets/twitter_ids/{date}.csv", index=False)
        self.user_ids = set()

    def on_connect(self):
        Path("datasets/twitter_ids").mkdir(parents=True, exist_ok=True)
        clear()
        print("Connected")

    def on_tweet(self, tweet):
        if tweet["lang"] == "en":
            self.user_ids.add(tweet["author_id"])

    def on_response(self, response):
        self.print_ids_collected()

    def on_errors(self, errors):
        self.save_to_csv()

    def on_closed(self, response):
        self.save_to_csv()

    def on_disconnect(self):
        self.save_to_csv()

    def on_exception(self):
        self.save_to_csv()


twitter_stream = TwitterStream(
    bearer_token=os.environ["BEARER_TOKEN"], wait_on_rate_limit=True)

twitter_stream.sample(expansions=["author_id"],
                      tweet_fields=["author_id", "lang"])
