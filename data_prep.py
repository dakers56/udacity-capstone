import re
import os
import joblib
import dateutil.parser
import datetime

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def alphanumerical(text):
    return re.sub(pattern=r'[\\]x[0-9a-eA-Z]{1,2}', repl="",string=re.sub(pattern="[^a-zA-Z0-9\s]",repl="", string=text)).replace("\n","")


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


def write_vectorizer(cnt_vec, folder='./'):
    now = str(dateutil.parser.parse(str(datetime.datetime.now()))).replace(" ", "_")
    joblib.dump(cnt_vec, "%svocab_%s.pkl" % (folder, now))


def get_vectorizer(corpus):
    return CountVectorizer(max_features=1000).fit(corpus)

def transform(cnt_vec, string):
    return cnt_vec.transform(string)

def vectorize_file(cnt_vec, file):
    file = open(file, 'r')
    X = transform(cnt_vec, file.readlines())
    file.close()
    return X.sum(axis=0, dtype=np.int8)

corpus = build_corpus(data=['/Users/devonakers/udacity/capstone/data_backup/seeking_alpha/A'])

cnt_vec = get_vectorizer(corpus)
X = vectorize_file(cnt_vec, '/Users/devonakers/udacity/capstone/data_backup/seeking_alpha/WRK/WRK_April_29,_2016_10:00_am_ET')
print(X)
print ("X.shape: %s" % str(X.shape))