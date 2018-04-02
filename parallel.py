# Trying to parallelize stemming
import os
import time
import copy
import queue
import data_prep
import multiprocessing as mp
import sys
import joblib
from nltk.stem import PorterStemmer
import numpy as np
# from multiprocessing import Pool, Process, Queue, Lock, cpu_count, set_start_method
from time import sleep
import data_prep

file_and_symbol = None
num_cores = 2


# ps = [PorterStemmer() for i in range(num_cores)]




def test(i):
    print("Running test thread %s" % str(i))
    print("Exiting test thread %s" % str(i))
    return i


def partition(data):
    l = len(data)
    print('l: %s' % l)
    n_parts = int(l / num_cores)
    print('n_parts: %s' % n_parts)
    size_part = num_cores
    print('size_part: %s' % size_part)
    parts = [data[i * size_part: (i + 1) * size_part] for i in range(n_parts)]
    if l % size_part > 0:
        parts.append(data[n_parts * size_part:])
    return parts


def parallelize(func, iter):
    pool = Pool()
    return pool.map(func, partition(iter))


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


# create a supervisor process for each processor. will obtain next file from queue, then begin processing. upon completion
# will append results to a queue.
def process(file_and_symbol, output, cnt_vec, not_processed_queue, print_lock, base_dir='data_backup/seeking_alpha'):
    X_train, all_eps, all_diluted_eps = None, None, None
    not_processed = data_prep.UnprocessedFileList()

    while not file_and_symbol.empty():
        file, symbol = file_and_symbol.get()
        __print("Next file is %s" % str(file), print_lock)
        __print("Next symbol is %s" % str(symbol), print_lock)

        funds = data_prep.vectorize_funds('fundamentals/%s' % symbol)
        if funds is None:
            continue

        quarter, year = data_prep.get_date(file)
        if quarter is None:
            not_processed.no_q.append(file)
            continue
        if quarter is "no_date_found":
            __print("No date found for file '%s'" % file)
            not_processed.no_diluted_eps.append(file)
            continue

        fv, eps, diluted_eps = data_prep.feature_vector(cnt_vec, file, funds, quarter, year, not_processed)

        if fv is None:
            not_processed.no_date_found.append(file)
            continue

        if eps is None:
            not_processed.no_eps.append(eps)
            continue
        if diluted_eps is None:
            not_processed.diluted_eps.append(diluted_eps)
            continue
        if not data_prep.are_all_finite(np.array([fv])) or not data_prep.are_all_finite(
                np.array([eps])) or not data_prep.are_all_finite(np.array([diluted_eps])):
            continue
        if X_train is None:
            X_train = np.array([fv])
        else:
            X_train = np.append(X_train, np.array([fv]), axis=0)
        if all_eps is None:
            all_eps = np.array([eps])
        else:
            all_eps = np.append(all_eps, np.array([eps]), axis=0)
        if all_diluted_eps is None:
            all_diluted_eps = np.array([diluted_eps])
        else:
            all_diluted_eps = np.append(all_diluted_eps, np.array([diluted_eps]), axis=0)

        __print("putting %s into output queue" % str((file, symbol)))
        not_processed_queue.put(not_processed)
        output.put(Output(X_train=X_train, all_eps=all_eps, all_diluted_eps=all_diluted_eps))


def pool_process(file_and_symbol, cnt_vec, print_lock, base_dir='data_backup/seeking_alpha'):
    X_train, all_eps, all_diluted_eps = None, None, None
    not_processed = data_prep.UnprocessedFileList()

    while not file_and_symbol.empty():
        file, symbol = file_and_symbol.get()
        __print("Next file is %s" % str(file), print_lock)
        __print("Next symbol is %s" % str(symbol), print_lock)

        funds = data_prep.vectorize_funds('fundamentals/%s' % symbol)
        if funds is None:
            continue

        quarter, year = data_prep.get_date(file)
        if quarter is None:
            not_processed.no_q.append(file)
            continue
        if quarter is "no_date_found":
            __print("No date found for file '%s'" % file)
            not_processed.no_diluted_eps.append(file)
            continue

        fv, eps, diluted_eps = data_prep.feature_vector(cnt_vec, file, funds, quarter, year, not_processed)

        if fv is None:
            not_processed.no_date_found.append(file)
            continue

        if eps is None:
            not_processed.no_eps.append(eps)
            continue
        if diluted_eps is None:
            not_processed.diluted_eps.append(diluted_eps)
            continue
        if not data_prep.are_all_finite(np.array([fv])) or not data_prep.are_all_finite(
                np.array([eps])) or not data_prep.are_all_finite(np.array([diluted_eps])):
            continue
        if X_train is None:
            X_train = np.array([fv])
        else:
            X_train = np.append(X_train, np.array([fv]), axis=0)
        if all_eps is None:
            all_eps = np.array([eps])
        else:
            all_eps = np.append(all_eps, np.array([eps]), axis=0)
        if all_diluted_eps is None:
            all_diluted_eps = np.array([diluted_eps])
        else:
            all_diluted_eps = np.append(all_diluted_eps, np.array([diluted_eps]), axis=0)

        return Results(X_train=X_train, all_eps=all_eps, all_diluted_eps=all_diluted_eps, not_processed=not_processed)


def read_all_transcript_files(base_dir='data_backup/seeking_alpha'):
    all_transcripts = []
    for symbol in os.listdir(base_dir):
        all_transcripts += data_prep.read_all_files_for_symbol(symbol, base_dir)
    return all_transcripts


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
