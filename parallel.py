# Trying to parallelize stemming
import multiprocessing as mp
import os
import time
from time import sleep

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
    print("Final corpus (unstemmed):")

    for w in corpus:
        print(w)
