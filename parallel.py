# Trying to parallelize stemming
import datetime
import multiprocessing as mp
import os
import time
from json import load, dump
import joblib
from time import sleep
from sklearn.feature_extraction.text import CountVectorizer

import dateutil
from nltk.stem import PorterStemmer

import data_prep

home = "/home/ubuntu"

def put_all(q, iter):
    for obj in iter:
        print("putting obj: %s" % obj)
        q.put(obj)
    sleep(1)


def get_all(q):
    all = []
    while not q.empty():
        el = q.get(timeout=1)
        print("el : %s" % str(el))
        all.append(el)
    print("inside get_all: %s" % all)
    return all


class Output:
    def __init__(self, X_train, all_eps, all_diluted_eps):
        self.X_train = X_train
        self.all_eps = all_eps
        self.all_diluted_eps = all_diluted_eps


class Results(Output):
    def __init__(self, X_train, all_eps, all_diluted_eps, not_processed):
        super().__init__(X_train=X_train, all_eps=all_eps, all_diluted_eps=all_diluted_eps)
        self.not_processed = not_processed


def __print(statement, lock):
    lock.acquire()
    try:
        print(statement)
    finally:
        lock.release()


def read_file(fn):
    corpus = []
    file = open(fn, 'r')
    lines = [l.split(" ") for l in file.readlines()]
    file.close()

    for l in lines:
        for w in l:
            corpus.append(data_prep.alphanumerical(w))

    return set(corpus)


def files_for_symbol(symbol, base_dir='data_backup/seeking_alpha'):
    symb_dir = base_dir + '/' + symbol + '/'
    return [symb_dir + f for f in os.listdir(base_dir + '/' + symbol)]


def all_transcript_files(base_dir='data_backup/seeking_alpha', with_symbol=False):
    if with_symbol:
        all_files={}
    else:
        all_files = []
    for symbol in os.listdir(base_dir):
        print("Getting files for symbol %s" % symbol)
        symbol_files = files_for_symbol(symbol, base_dir=base_dir)
        for sf in symbol_files:
            if with_symbol:
                all_files[sf]=symbol
            else:
                all_files.append(sf)
    return all_files


def stem_word(stemmer, word):
    return stemmer.stem(word)


def stem_words(words):
    stemmer = PorterStemmer()
    return stem_word(stemmer, words)


def save_corpus(corpus, stemmed=False, loc='corpora'):
    if stemmed:
        stemmed = 'stemmed'
    else:
        stemmed = 'unstemmed'
    if type(corpus) is type(set):
        corpus = list(corpus)
    now = str(dateutil.parser.parse(str(datetime.datetime.now()))).replace(" ", "_")
    f = open("%s/corpus-%s-%s" % (loc, stemmed, now), 'w')
    dump(corpus, f)
    # f.write(dumps(corpus))
    # f.close()


class FileSymbolAndCountVect:
    def __init__(self, file, symbol, cnt_vec):
        self.file=file
        self.symbol=symbol
        self.cnt_vec=cnt_vec

def __match_funds(fscv_obj):
    print("File: %s" % fscv_obj.file)
    print("Symbol: %s" % fscv_obj.symbol)
    print("cnt_vec: %s" % fscv_obj.cnt_vec)
    data_prep.match_funds(file=fscv_obj.file, symbol=fscv_obj.symbol, cnt_vec=fscv_obj.cnt_vec)

def __to_fscv_list(fs_dict, cnt_vec):
    return [FileSymbolAndCountVect(file=f, symbol=s, cnt_vec=cnt_vec) for f,s in fs_dict.items()]

if __name__ == '__main__':
    start = time.clock()
    corpus = set()

    with mp.Pool(processes=mp.cpu_count()) as pool:
        res = pool.map(read_file, all_transcript_files())
        for r in res:
            for s in r:
                corpus.add(s)
        print("size of corpus is %s" % len(corpus))
        stop = time.clock()

        print("Time to process corpus: %s" % str(stop - start))

        save_corpus(list(corpus), stemmed=False)

        file = open(home + '/udacity/udacity-capstone/corpus-unstemmed-2018-04-02_13:32:53.084386', 'r')
        corpus = load(file)
        file.close()

        print("Corpus contents:")
        for w in corpus:
            print(w)

        print("Stemming corpus")
        start = time.clock()
        _stemmed = pool.map(stem_words, corpus)
        print("type(_stemmed): %s" % type(_stemmed))
        print("len(_stemmed): %s" % len(_stemmed))
        print("_stemmed: %s" % _stemmed)
        stop = time.clock()
        print("Time to stem corpus: %s" % str(stop - start))

        stemmed = set()
        for s in _stemmed:
            stemmed.add(s)

        print("Done stemming corpus")
        print("Size of corpus: %s" % len(corpus))
        print("Stemmed corpus:")
        for w in stemmed:
            print(w)
        save_corpus(list(stemmed), stemmed=True)
        corpus=stemmed
    print("Fitting BoW transform")
    start=time.clock()
    BoW_transform = CountVectorizer(analyzer='word').fit(stemmed)
    stop=time.clock()
    print("Done training- %s" % str(stop-start))
    print("Type of BoW_transform: %s" % type(BoW_transform))
    joblib.dump(BoW_transform, "BoW_transform/%s" % str(dateutil.parser.parse(str(datetime.datetime.now()))).replace(" ", "_"))
    print("Wrote BoW_transform to file")

    trans_dict = all_transcript_files(with_symbol=True)
    fscv_list=__to_fscv_list(trans_dict, BoW_transform)

    model_input=None
    with mp.Pool(processes=mp.cpu_count()) as pool:
        model_input=pool.map(__match_funds, fscv_list)

    print("type(model_input)= %s" % type(model_input))
    print("Saving final vectors")
    joblib.dump(model_input, "model_input/model_input-%s" % str(dateutil.parser.parse(str(datetime.datetime.now()))).replace(" ", "_"))

