# Trying to parallelize stemming
import datetime
import multiprocessing as mp
import os
import time
import joblib
import pandas as pd
import numpy as np
import sys
from time import sleep
from sklearn.feature_extraction.text import CountVectorizer

import dateutil
from nltk.stem import PorterStemmer

import data_prep

home = "/home/ubuntu"


class Output:
    def __init__(self, X_train, all_eps, all_diluted_eps):
        self.X_train = X_train
        self.all_eps = all_eps
        self.all_diluted_eps = all_diluted_eps


class Results(Output):
    def __init__(self, X_train, all_eps, all_diluted_eps, not_processed):
        super().__init__(X_train=X_train, all_eps=all_eps, all_diluted_eps=all_diluted_eps)
        self.not_processed = not_processed


def read_file(fn):
    corpus = []
    file = open(fn, 'r')
    lines = [l.split(" ") for l in file.readlines()]
    file.close()

    for l in lines:
        for w in l:
            corpus.append(data_prep.alphanumerical(w))

    return set(corpus)


def files_for_symbol(symbol, base_dir='data'):
    symb_dir = base_dir + '/' + symbol + '/'
    return [symb_dir + f for f in os.listdir(base_dir + '/' + symbol)]


def all_transcript_files(base_dir='data', with_symbol=False):
    if with_symbol:
        all_files = {}
    else:
        all_files = []
    for symbol in os.listdir(base_dir):
        print("Getting files for symbol %s" % symbol)
        symbol_files = files_for_symbol(symbol, base_dir=base_dir)
        for sf in symbol_files:
            if with_symbol:
                all_files[sf] = symbol
            else:
                all_files.append(sf)
    return all_files


def stem_word(stemmer, word):
    return stemmer.stem(word)


def stem_words(words):
    stemmer = PorterStemmer()
    return stem_word(stemmer, words)


def save_corpus(corpus, stemmed=False, loc='corpora'):
    if not os.path.exists(loc):
        print("Directory '%s' did not exist. Creating now." % loc)
        os.mkdir(loc)
    if stemmed:
        stemmed = 'stemmed'
    else:
        stemmed = 'unstemmed'
    if type(corpus) is type(set):
        corpus = list(corpus)
    now = str(dateutil.parser.parse(str(datetime.datetime.now()))).replace(" ", "_")
    fn = "%s/corpus-%s-%s" % (loc, stemmed, now)
    joblib.dump(corpus, fn)
    # f.write(dumps(corpus))
    # f.close()


class FileSymbolAndCountVect:
    def __init__(self, file, symbol, cnt_vec, df_file=None):
        self.file = file
        self.symbol = symbol
        self.cnt_vec = cnt_vec
        self.df_file = df_file


def __match_funds(fscv_obj, df_file=None):
    print("File: %s" % fscv_obj.file)
    print("Symbol: %s" % fscv_obj.symbol)
    print("cnt_vec: %s" % fscv_obj.cnt_vec)
    test = None

    try:
        test = data_prep.match_funds(file=fscv_obj.file, symbol=fscv_obj.symbol, cnt_vec=fscv_obj.cnt_vec,
                                     df_file=df_file)
    except pd.io.common.EmptyDataError as e:
        print("Caught exception while matching transcript with fundamentals: " + str(e))
    return test


def __to_fscv_list(fs_dict, cnt_vec, df_file=None):
    return [FileSymbolAndCountVect(file=f, symbol=s, cnt_vec=cnt_vec, df_file=df_file) for f, s in fs_dict.items()]


def symbol_map(X_train):
    df = pd.DataFrame(X_train)
    symbols = set(df[:][0].unique())
    i = 0
    sym_map = {}
    for s in symbols:
        sym_map[s] = np.float64(i)
        i += 1
    return sym_map


def symbol_counts(X_train):
    df = pd.DataFrame(X_train)
    sym_cnt = {}
    for sym in set(df[:][0].unique()):
        sym_cnt[sym] = np.float64(df[df[0] == sym].count()[0])
    return sym_cnt


def split_mro_list(mro_list):
    split_list = []
    for i in range(0, len(mro_list), 2):
        if i < (len(mro_list) - 1):
            split_list.append([mro_list[i], mro_list[i + 1]])
        else:
            split_list.append([mro_list[i]])
    return split_list


if __name__ == '__main__':

    _start = time.time()

    print("Generating vectorized data for udacity capstone project model.")
    print("Arguments provided:")
    for a in sys.argv[1:]:
        print(a)

    ld_unstemmed = False
    unstemmed_file = None
    if len(sys.argv) > 1:
        unstemmed_file = sys.argv[1]
        ld_unstemmed = True

    ld_stemmed = False
    stemmed_file = None
    if len(sys.argv) > 2:
        stemmed_file = sys.argv[2]
        ld_stemmed = True

    ld_BoW = False
    BoW_transform = None
    if len(sys.argv) > 3:
        BoW_transform = sys.argv[3]
        ld_BoW = True

    print("Unstemmed file arg: %s" % unstemmed_file)
    print("Stemmed file arg: %s" % stemmed_file)
    print("Stemmed file arg: %s" % stemmed_file)

    corpus = set()
    _stemmed = set()

    if ld_stemmed:
        print("Loading stemmed corpus from %s" % stemmed_file)
        corpus = joblib.load(stemmed_file)
    else:
        if ld_unstemmed:
            if not ld_stemmed:
                print("Unstemmed file provided and no stemmed file provided. Loading unstemmed corpus.")
                corpus = joblib.load(unstemmed_file)
                print("Done loading unstemmed corpus.")
        else:
            print("No corpus file provided, generating unstemmed corpus.")
            with mp.Pool(processes=mp.cpu_count()) as pool:
                start = time.time()
                res = pool.map(read_file, all_transcript_files())
                for r in res:
                    for s in r:
                        corpus.add(s)
                print("size of corpus is %s" % len(corpus))
                stop = time.time()
                print("Time to process corpus: %s" % str((stop - start)))
                save_corpus(list(corpus), stemmed=False)

                print("No stemmed corpus file provided, generating stemmed corpus.")
                print("Stemming corpus.")
                start = time.time()
                _stemmed = pool.map(stem_words, corpus)
                print("Done stemming corpus.")
                stop = time.time()
                print("Time to stem corpus: %s" % str((stop - start)))
                print("Saving stemmed corpus.")
                for s in _stemmed:
                    corpus.add(s)
                save_corpus(list(corpus), stemmed=True)

    if not ld_BoW:
        print("Fitting BoW transform")
        start = time.time()
        BoW_transform = CountVectorizer(analyzer='word').fit(corpus)
        stop = time.time()
        print("Done training BoW transform - %s s" % str(stop - start))
        if not os.path.exists("BoW_transform"):
            print("Directory BoW_transform did not exist. Creating now")
        os.mkdir("BoW_transform")
        joblib.dump(BoW_transform,
                    "BoW_transform/%s-vocab_size=%s" % (
                        str(dateutil.parser.parse(str(datetime.datetime.now()))).replace(" ", "_"),
                        BoW_transform.max_features))
        print("Wrote BoW_transform to file.")

    trans_dict = all_transcript_files(with_symbol=True)
    df_file = "/Users/mve526/udacity/udacity-capstone/russell_full.csv"
    fscv_list = __to_fscv_list(trans_dict, BoW_transform, df_file=df_file)

    model_input = None
    with mp.Pool(processes=mp.cpu_count()) as pool:
        model_input = pool.map(__match_funds, fscv_list)


    model_input = [m for m in model_input if m.is_valid()]
    print("Combining inputs into single matrix.")
    print("Time: %s" % str(datetime.datetime.now()))
    start = time.time()
    # model_input = data_prep.join_output(model_input)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        as_list = split_mro_list(model_input)
        while len(as_list) > 1:
            as_list = pool.map(data_prep.join_output, as_list)
            as_list = split_mro_list(as_list)

    as_list[0][0].join(as_list[0][1])
    model_input = as_list[0][0]
    stop = time.time()
    print("Done - %ss" % str(stop - start))
    now = str(datetime.datetime.now()).replace(" ", "_")
    start = time.time()
    print("Mapping symbols to integers.")
    sym_map = symbol_map(model_input.X_train)
    stop = time.time()
    print("Done mapping symbols to integers - %ss" % str(stop - start))

    print("Counting number of instances of each stock symbol.")
    start = time.time()
    sym_cnt = symbol_counts(model_input.X_train)
    stop = time.time()

    print("Dumping symbol map to file")
    if not os.path.exists("symbol_map"):
        print("Directory symbol_map did not exist. Creating now")
        os.mkdir("symbol_map")
    start = time.time()
    joblib.dump(sym_map, "symbol_map/symbol-map-%s" % now)
    stop = time.time()
    print("Done - %ss" % str(stop - start))

    print("Dumping symbol count to file")
    if not os.path.exists("symbol_map"):
        print("Directory symbol_count did not exist. Creating now")
        os.mkdir("symbol_count")
    start = time.time()
    joblib.dump(sym_cnt, "symbol_count/symbol-count-%s" % now)
    stop = time.time()
    print("Done - %ss" % str(stop - start))

    print("Converting symbols to integers.")
    start = time.time()
    for i in range(model_input.X_train.shape[0]):
        model_input.X_train[i][0] = sym_map[model_input.X_train[i][0]]
    stop = time.time()
    print("Done - %ss" % str(stop - start))

    model_input.X_train = model_input.X_train

    print("Saving final vectors")
    if not os.path.exists("model_input"):
        print("Directory model_input did not exist. Creating now")
        os.mkdir("model_input")
    start = time.time()
    joblib.dump(model_input,
                ("model_input/model_input-%s-vocab_size=%s" % (now, BoW_transform.max_features)))
    stop = time.time()
    print("Done - %ss" % str(stop - start))

    print("Total time taken: %s" % (str(time.time() - _start)))
    print("Total time taken(process time): %s" % str(time.process_time()))
