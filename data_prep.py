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
    paths = []
    for symbol in symbols():
        path = path_to_symbol(symbol)
        for file in os.listdir(path):
            paths.append("%s/%s" % (path, file))
            X_train.append(vectorize_file(cnt_vec, "%s/%s" % (path, file)))
            y_train.append(symbol)
    if (len(X_train)) != (len(y_train)):
        print("Number of training examples does not match number of files")
    return np.array(X_train), np.array(y_train), paths


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
    return X.sum(axis=0, dtype=np.int64)


def target_labels():
    targets = []
    # for symbol in data_folders()


print("symbols: %s" % symbols())


cnt_vec = None
vocab_path = 'vocab.pkl'

if not os.path.exists(vocab_path):
    corpus = build_corpus()
    cnt_vec = get_vectorizer(corpus)
    write_vectorizer(cnt_vec)
else:
    cnt_vec = joblib.load(vocab_path)

X, y, paths = get_samples(cnt_vec)
print(X)
print(y)
print(paths)
print("X.shape: %s" % str(X.shape))
print("y.shape: %s" % str(y.shape))
print("paths.shape: %s" % str(y.shape))
