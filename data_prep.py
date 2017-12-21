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


def all_transcript_samples(cnt_vec):
    """

    :param cnt_vec: count vectorizer
    :return: X_train as shuffled dataset, files they come from. Files will be used to pick out dates.
    """
    X_train = []
    files = []
    for symbol in symbols():
        path = path_to_symbol(symbol)
        for file in os.listdir(path):
            X_train.append(vectorize_transcript(cnt_vec, "%s/%s" % (path, file)))
            files.append(symbol)

    return train_test_split(X_train, files)


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


def vectorize_transcript(cnt_vec, file):
    file = open(file, 'r')
    X = transform(cnt_vec, file.readlines())
    file.close()
    return X.sum(axis=0, dtype=np.int64).getA()[0]


def vectorize_funds(file):
    return pd.read_csv(file).drop('symbol', axis=1).drop('end_date', axis=1).drop('amend', axis=1).drop('doc_type',
                                                                                                        axis=1)


def cat_vectors(transcript, funds):
    print("transcript.shape: %s" % transcript.shape)
    print("funds.shape: %s" % str(funds.shape))
    return np.concatenate((transcript, funds))

def funds_exist(symbol):
    return os.path.exists("fundamentals/%s" % symbol)

def get_input_data(cnt_vec, base_dir='data_backup/seeking_alpha'):
    X_train = []
    for symbol in os.listdir(base_dir):
        transcript_path = "%s/%s" % (base_dir, symbol)
        funds = vectorize_funds('fundamentals/%s' % symbol)
        if funds_exist(symbol):
            for file in os.listdir(transcript_path):
                quarter, year = get_date(file)
    return X_train

def all_funds():
    all = None
    base_dir = "fundamentals"
    for file in os.listdir(base_dir):
        path = "%s/%s" % (base_dir, file)
        df = vectorize_funds(path)
        if all is None:
            all = df
        else:
            all = all.append(df)
    return all


def get_matching_funds(funds, quarter, year):
    print("Type(funds.period_focus): %s" % type(funds.period_focus))
    print("Type(funds.fiscal_year): %s" % type(funds.fiscal_year))
    print("Type(funds.fiscal_year[0]): %s" % type(funds.period_focus[0]))
    print("Type(funds.fiscal_year[0]): %s" % type(funds.fiscal_year[0]))
    return funds[(funds.period_focus == quarter) & (funds.fiscal_year == year)]

def swap_str(string, ch1, ch2):
    ch1_index = string.index(ch1)
    ch2_index = string.index(ch2)
    as_list = list(string)
    as_list[ch1_index] = ch2
    as_list[ch2_index] = ch1
    return str(as_list)


def swap_at_ind(string, i1, i2):
    string = list(string)
    temp = string[i1]
    string[i1] = string[i2]
    string[i2] = temp
    return "".join(string)


def swap_NQ(nq_or_qn):
    return swap_at_ind(nq_or_qn, 0, 1)


def get_NQYYYY(string):
    is_nqyy = re.search("[0-4]Q[0-9]{2,4}", string)
    if is_nqyy:
        match = is_nqyy.group(0)
        return match[:2], match[2:]
    is_nqyy = re.search("Q[0-4][0-9]{2,4}", string)
    if is_nqyy:
        match = swap_NQ(is_nqyy.group(0))
        return match[:2], match[2:]
    is_nqyy = re.search(r'Q[0-4]\s[0-9]{2,4}', string)
    if is_nqyy:
        match = swap_NQ(is_nqyy.group(0))
        return match[:2], match[2:]
    is_nqyy = re.search('F[1-4]Q\\s[0-9]{2,4}', string)
    if is_nqyy:
        match = swap_NQ(is_nqyy.group(0))
        return match[1:3], match[-2:]
    is_nqyy = re.search(r'F[0-4]Q\s[0-9]{4}', string)
    if is_nqyy:
        match = swap_NQ(is_nqyy.group(0))
        return match[1:3], "20".join(match[-2:])


def get_N(qn_or_nq):
    return re.search("[0-4]", qn_or_nq).group(0)


def get_date(file):
    file = open(file, 'r')
    for line in file.readlines():
        pat1 = [r'F[0-4]Q\s[0-9]{4}', 'F[0-4]Q[0-9]{,4}', r'Q[0-4]\s[0-9]{,4}', 'Q[0-4] [0-9]{2,4}',
                'F[1-4]Q\\s[0-9]{2,4}']
        # pat1 = ['F[0-4]Q[0-9]{,4}', r'Q[0-4]\s[0-9]{,4}', 'Q[0-4] [0-9]{2,4}', 'F[1-4]Q\\s[0-9]{2,4}']

        for p in pat1:
            m = re.search(p, line)
            if m:
                return get_NQYYYY(m.group(0))
    return "no_date_found"


def feature_vector(cnt_vec, transcript_file, quarter, year):
    fv = vectorize_transcript(cnt_vec, transcript_file)
    fv = np.append(fv, quarter)
    fv = np.append(fv, year)
    return fv


if __name__ == '__main__':

    # base = 'data_backup/seeking_alpha/A'
    # total_correct = 0
    # total = 0
    # for base in data_folders():
    #     for t in os.listdir(base):
    #         date = get_date(base + '/' + str(t))
    #         total += 1
    #         if date != 'no_date_found' and date is not None:
    #             total_correct += 1
    #         print('%s: %s' % (t, str(date)))
    # print('total correct: %s' % total_correct)
    # print('total: %s' % total)

    # print("Training model for capstone project")
    # now = time.clock()
    # cnt_vec = None
    # vocab_path = 'vocab.pkl'
    # #
    # if not os.path.exists(vocab_path):
    #     corpus = build_corpus()
    #     cnt_vec = get_vectorizer(corpus)
    #     write_vectorizer(cnt_vec)
    # else:
    #     cnt_vec = joblib.load(vocab_path)

    X_train = pd.read_csv('fundamentals/AAP')
    # print('X_train: %s' % X_train['period_focus'])
    quarter, year = 'Q1', '2014'
    # print(X_train)
    funds = get_matching_funds(funds=X_train, quarter='Q1', year=int('2014'))
    print("Matching funds:\n%s" % funds)

    # X_train = get_input_data(cnt_vec)
    #
    # print("Sample from input data:\n%s" % X_train[0])


    #
    # print("Training classifier")
    # clf = GaussianNB().fit(X_train, y_train)
    # print("Training accuracy: %s" % clf.score(X_train, y_train))
    # print("Testing accuracy: %s" % clf.score(X_test, y_test))
    #
    # now = time.clock() - now
    # print("Process took %s miliseconds." )
