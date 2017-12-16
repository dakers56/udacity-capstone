import re
import os
import joblib
import dateutil.parser
import datetime

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


def alphanumerical(text):
    return re.sub(pattern="^[a-zA-Z0-9\s]", repl="", string=text)


def build_corpus():
    corpus = []
    for folder in data_folders():
        for file in os.listdir(folder):
            file = open("%s/%s" % (folder, file))
            corpus.append(alphanumerical("".join(file.readlines())))
            file.close()
    return corpus


def data_folders():
    folders = []
    base_dir = 'data_backup/seeking_alpha'
    for folder in os.listdir(base_dir):
        folders.append("%s/%s" % (base_dir, folder))
    return folders


def write(cnt_vec, folder='./'):
    now = str(dateutil.parser.parse(str(datetime.datetime.now()))).replace(" ", "_")
    joblib.dump(cnt_vec, "%svocab_%s.pkl" % (folder, now))


stopwords = stopwords.words('english')
test_file = open('/Users/devonakers/udacity/capstone/data_backup/seeking_alpha/A/A_August 15, 2017 04:30 PM ET', 'r')
test_transcript = alphanumerical("".join(test_file.readlines()))
corpus = build_corpus()

print("Fitting CountVectorizer to determine vocabulary.")
cnt_vec = CountVectorizer(input=test_transcript, stop_words=set(stopwords)).fit(corpus)
cnt_vec = '{}'
print("Final vocabulary:")
for v in cnt_vec.vocabulary_:
    print(str(v))

write(cnt_vec)
