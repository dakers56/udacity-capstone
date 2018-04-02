# Trying to parallelize stemming
import datetime
import multiprocessing as mp
import os
import time
from json import load, dump
from time import sleep

import dateutil
from nltk.stem import PorterStemmer

import data_prep


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


def all_transcript_files(base_dir='data_backup/seeking_alpha'):
    all_files = []
    for symbol in os.listdir(base_dir):
        print("Getting files for symbol %s" % symbol)
        symbol_files = files_for_symbol(symbol, base_dir=base_dir)
        for sf in symbol_files:
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

        file = open('/Users/mve526/udacity/udacity-capstone/corpus-unstemmed-2018-04-02_13:32:53.084386', 'r')
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
