import user_features
import pandas as pd
import pymongo
import os
import random

from dotenv import load_dotenv

load_dotenv()


try:
  mongo_client = pymongo.MongoClient(os.environ["MONGODB_URL"])
except:
  print("Could not connect to DB")
  exit()


db = mongo_client["ml-social"]
twitter_collection = db["twitter"]
print("Connected to DB")


def clear(): return os.system("cls")


clear()

ids = pd.read_csv("datasets/twitter_ids.csv")["id"].to_list()
random.shuffle(ids)
print("Loaded Twitter ID dataset")

count = 0

for id in ids:
    user = twitter_collection.find_one({"user_id": id})

    if user is not None:
        continue

    features = user_features.extract(id)
    x = twitter_collection.insert_one(features)

    count += 1
    clear()
    print(f"{count} done so far")
