import queue
import user_features
import pandas as pd
import pymongo
import os
import random
import logging

from dotenv import load_dotenv
from queue import Queue
from threading import Thread
from datetime import datetime

load_dotenv()


count = 0


def clear(): return os.system("cls")


clear()
try:
    mongo_client = pymongo.MongoClient(os.environ["MONGODB_URL"])
except:
    print("Could not connect to DB")
    exit()


db = mongo_client["ml-social"]
twitter_collection = db["twitter"]
print("Connected to DB")


class DownloadWorker(Thread):
    def __init__(self, queue: Queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            id = self.queue.get()
            try:
                user = twitter_collection.find_one({"user_id": id})
                if user is not None:
                    return

                features = user_features.extract(id)
                twitter_collection.insert_one(features)
                print(id)
            except:
                date = datetime.today().strftime('%H:%M:%S')
                print(f"{date} error")
            finally:
                self.queue.task_done()


ids = pd.read_csv("datasets/twitter_ids.csv")["id"].to_list()
random.shuffle(ids)
print("Loaded Twitter ID dataset")


queue = Queue()

for x in range(4):
    worker = DownloadWorker(queue)
    worker.daemon = True
    worker.start()

for id in ids:
    queue.put(id)

queue.join()
