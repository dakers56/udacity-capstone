import re
import os
import joblib
import dateutil.parser
import datetime
import pandas as pd
import time
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np


def alphanumerical(text):
    return re.sub(pattern=r'[\\]x[0-9a-eA-Z]{1,2}', repl="",
                  string=re.sub(pattern="[^a-zA-Z0-9\s]", repl="", string=text)).replace("\n", "")


def stem_file(file):
    print("Stemming file: %s" % file)
    lines = [alphanumerical(line) for line in file.readlines()]
    words = []
    for line in lines:
        for word in line.split(" "):
            words.append(word)
    return [PorterStemmer().stem(word) for word in words if word not in set(stopwords.words('english'))]


def unpack_elements(nested_list):
    unpacked = []
    for i in range(len(nested_list)):
        for element in nested_list[i]:
            unpacked.append(element)
    return unpacked


def build_corpus(data=None):
    corpus = []
    for folder in data if data else data_folders():
        for file in os.listdir(folder):
            file = open("%s/%s" % (folder, file))
            for word in stem_file(file):
                corpus.append(word)
            file.close()
    return corpus


def data_folders():
    folders = []
    base_dir = 'data_backup/seeking_alpha'
    for folder in os.listdir(base_dir):
        folders.append("%s/%s" % (base_dir, folder))
    return folders


def symbols():
    symbols = set()
    for path in data_folders():
        symbols.add(path.split("/")[-1])
    return symbols


def path_to_symbol(symbol):
    return 'data_backup/seeking_alpha/%s' % symbol


def get_samples(cnt_vec):
    X_train = []
    y_train = []
    for symbol in symbols():
        path = path_to_symbol(symbol)
        for file in os.listdir(path):
            X_train.append(vectorize_file(cnt_vec, "%s/%s" % (path, file)))
            y_train.append(symbol)

    return train_test_split(X_train, np.array(y_train))


def write_vectorizer(cnt_vec, folder='./', testing=True):
    now = str(dateutil.parser.parse(str(datetime.datetime.now()))).replace(" ", "_")
    if not testing:
        joblib.dump(cnt_vec, "%svocab_%s.pkl" % (folder, now))
        return
    joblib.dump(cnt_vec, "vocab.pkl")


def get_vectorizer(corpus):
    return CountVectorizer(max_features=1000).fit(corpus)


def transform(cnt_vec, string):
    return cnt_vec.transform(string)


def vectorize_file(cnt_vec, file):
    file = open(file, 'r')
    X = transform(cnt_vec, file.readlines())
    file.close()
    return X.sum(axis=0, dtype=np.int64).getA()[0]


def read_fund(file):
    return pd.read_csv(file)

def all_funds():
    all = None
    base_dir = "fundamentals"
    for file in os.listdir(base_dir):
        path = "%s/%s" % (base_dir, file)
        df = read_fund(path)
        if all is None:
            all = df
        else:
            all = all.append(df)
    return all

def get_date(file):
    file = open(file, 'r')
    for line in file.readlines():
        pat = ['F[0-9]Q[0-9]{,4}']
        for p in pat:
            m = re.search(p, line)
            if m:
                return m.group(0).replace(" ", "_")
    return "no_date_found"



if __name__ == '__main__':
    test = ['data_backup/seeking_alpha/A/A__August_14,_2007_4:30_pm_ET_', 'data_backup/seeking_alpha/A/A__February_15,_2007_4:30_pm_ET_', 'data_backup/seeking_alpha/A/A__May_15,_2007_4:30_pm_ET_']
    for t in test:
        print(get_date(t))



    # print("Training model for capstone project")
    # now = time.clock()
    # cnt_vec = None
    # vocab_path = 'vocab.pkl'
    #
    # if not os.path.exists(vocab_path):
    #     corpus = build_corpus()
    #     cnt_vec = get_vectorizer(corpus)
    #     write_vectorizer(cnt_vec)
    # else:
    #     cnt_vec = joblib.load(vocab_path)
    #
    # X_train, X_test, y_train, y_test = get_samples(cnt_vec)
    #
    # print("Training classifier")
    # clf = GaussianNB().fit(X_train, y_train)
    # print("Training accuracy: %s" % clf.score(X_train, y_train))
    # print("Testing accuracy: %s" % clf.score(X_test, y_test))
    #
    # now = time.clock() - now
    # print("Process took %s miliseconds." )