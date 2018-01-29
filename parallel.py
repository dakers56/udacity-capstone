# Trying to parallelize stemming
import datetime
import os
import copy
import sys
import joblib
from nltk.stem import PorterStemmer
import numpy as np
import multiprocessing as mp
from time import sleep
import data_prep

file_and_symbol = None
num_cores = mp.cpu_count()


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
    pool = mp.Pool()
    return pool.map(func, partition(iter))


def put_all(q, iter):
    for obj in iter:
        q.put(obj)
    return q


def get_all(q):
    all = []
    while not q.empty():
        el = q.get(timeout=1)
        all.append(el)
    return all


class Output:
    def __init__(self, X_train, all_eps, all_diluted_eps):
        self.X_train = X_train
        self.all_eps = all_eps
        self.all_diluted_eps = all_diluted_eps


def __print(statement, lock):
    lock.acquire()
    try:
        print("%s - %s" % (datetime.datetime.now(), statement))
    finally:
        lock.release()


def process(file_and_symbol, output, cnt_vec, not_processed_queue, print_lock, proc_num,
            base_dir='data_backup/seeking_alpha'):
    print("processing data")
    print("--------------")
    X_train, all_eps, all_diluted_eps = None, None, None
    not_processed = data_prep.UnprocessedFileList()

    while not file_and_symbol.empty():
        __print("mp.Process number %s running." % proc_num, print_lock)
        file, symbol = file_and_symbol.get()
        __print("Next file to process is %s" % str(file), print_lock)
        __print("Next symbol to process is %s" % str(symbol), print_lock)
        __print("Items remaining in queue to process: %s" % str(symbol), print_lock)

        funds = data_prep.vectorize_funds('fundamentals/%s' % symbol)
        if funds is None:
            file_and_symbol.task_done()
            continue

        quarter, year = data_prep.get_date(file)
        if quarter is None:
            not_processed.no_q.append(file)
            file_and_symbol.task_done()
            continue
        if quarter is "no_date_found":
            __print("No date found for file '%s'" % file, print_lock)
            not_processed.no_diluted_eps.append(file)
            file_and_symbol.task_done()
            continue

        fv, eps, diluted_eps = data_prep.feature_vector(cnt_vec, file, funds, quarter, year, not_processed)

        if fv is None:
            not_processed.no_date_found.append(file)
            file_and_symbol.task_done()
            continue

        file_and_symbol.task_done()

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

        __print("putting %s into output queue" % str((file, symbol)), print_lock)
        not_processed_queue.put(not_processed)
        output.put(Output(X_train=X_train, all_eps=all_eps, all_diluted_eps=all_diluted_eps))


def write_output(output):
    joblib.dump(output.X_train, "X_train.pkl")
    joblib.dump(output.all_eps, "all_eps.pkl")
    joblib.dump(output.all_diluted_eps, "all_diluted_eps.pkl")


def cat_output(output_q):
    output = output_q.get()
    X_train = output.X_train
    all_eps = output.all_eps
    all_diluted_eps = output.all_diluted_eps
    while not output_q.empty():
        output = output_q.get()
        X_train = np.append(X_train, output.X_train)
        all_eps = np.append(all_eps, output.all_eps)
        all_diluted_eps = np.append(all_diluted_eps, output.all_diluted_eps)
    return Output(X_train=X_train, all_eps=all_eps, all_diluted_eps=all_diluted_eps)


def read_all_transcript_files(base_dir='data_backup/seeking_alpha'):
    all_transcripts = []
    for symbol in os.listdir(base_dir):
        all_transcripts += data_prep.read_all_files_for_symbol(symbol, base_dir)
    return all_transcripts


def make_corpus_q(files_q, stemmed_q, print_lock):
    sleep(.5)
    __print("Selecting files for corpus", print_lock)
    stemmer = PorterStemmer()
    while not files_q.empty():
        file = files_q.get()
        fn = str(file)
        q_size = files_q.qsize()
        __print("Files remaining in queue for corpus:  %s" % str(q_size), print_lock)
        file = open(file, 'r')
        __print("Next file for corpus is %s" % fn, print_lock)
        for word in data_prep.stem_file(stemmer, file):
            stemmed_q.put(word)
        file.close()
        files_q.task_done()
        __print("Closed file '%s'" % fn, print_lock)


def corpus_q_as_set(corpus_q):
    as_set = set()
    while not corpus_q.empty():
        q = corpus_q.get()
        print("Adding %s to corpus" % str(q))
        as_set.add(q)
        corpus_q.task_done()
    print("Done converting corpus queue to set")
    return as_set


if __name__ == '__main__':
    print("inside main")
    train_new = sys.argv[1] == "true"

    transcript_tuples = read_all_transcript_files()
    all_symbols = [x[0] for x in transcript_tuples]
    all_transcript_files = [x[1] for x in transcript_tuples]
    file_and_symbol = mp.Queue()
    put_all(file_and_symbol, (all_symbols, all_transcript_files))

    file_only = mp.JoinableQueue()
    file_only = put_all(file_only, all_transcript_files)

    output = mp.Queue()
    not_processed = mp.Queue()
    print_lock = mp.Lock()

    cnt_vec = None
    vocab_path = 'vocab.pkl'

    if train_new:
        stemmed_q = mp.Queue()
        stem_proc = []
        for i in range(num_cores):
            p = mp.Process(target=make_corpus_q, args=(file_only, stemmed_q, print_lock))
            stem_proc.append(p)
            p.start()
            print("Started process %s for stemming" % i)
        print("Done starting processes")
        while not file_only.empty():
            print("Still have active children: %s" % len(mp.active_children()))
            sleep(.5)
        print("Waiting until all files are consumed to create corpus queue")
        file_only.join()
        for p in stem_proc:
            print("Performing final join of process")
            p.join()
        print("Done making corpus queue")
        corpus = corpus_q_as_set(stemmed_q)
        print("Done converting corpus queue to set")
        print("Creating count vectorizer")
        cnt_vec = data_prep.get_vectorizer(corpus)
        print("Done creating count vectorizer")
        print("Writing count vectorizer to disk")
        data_prep.write_vectorizer(cnt_vec)
        print("Done writing vectorizer to disk")
    else:
        print("Reading vocab from disk")
        cnt_vec = joblib.load(vocab_path)

    if train_new:
        print("Creating new model.")
        cnt_vec = [copy.copy(cnt_vec) for i in range(num_cores)]
        cnt_vec_proc = []
        for i in range(num_cores):
            print("------")
            print("Creating process %s" % i)
            p = mp.Process(target=process, args=(file_and_symbol, output, cnt_vec[i], not_processed, print_lock, i))
            cnt_vec_proc.append(p)
            p.start()

        while not file_and_symbol.empty():
            print("Still have active children: %s" % len(mp.active_children()))
            sleep(.5)
        print("Waiting to consume all of file_and_symbol queue")
        file_and_symbol.join()
        print("Done creating model.")

        for p in cnt_vec_proc:
            print("Terminating process for cnt_vec")
            p.join()

        print("Writing output to file.")
        write_output(cat_output(output))
        print("Done writing output to file.")
