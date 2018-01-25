# Trying to parallelize stemming
import os
from nltk.stem import PorterStemmer
import numpy as np
from multiprocessing import Pool, Process, Queue, Lock
from time import sleep
import data_prep

files = None
num_cores = 64
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
    def __init__(self, X_train, all_eps, all_diluted_eps, not_processed):
        self.X_train = X_train
        self.all_eps = all_eps
        self.all_diluted_eps = all_diluted_eps
        self.not_processed = not_processed
def __print(statement, lock):
    lock.acquire()
    try:
        print(statement)
    finally:
        lock.release()

#create a supervisor process for each processor. will obtain next file from queue, then begin processing. upon completion
# will append results to a queue.
def supervisor(file_and_symbol, output, print_lock, base_dir='data_backup/seeking_alpha'):
    X_train, all_eps, all_diluted_eps = None, None, None
    next_file = None
    not_processed = data_prep.UnprocessedFileList()

    while not file_and_symbol.empty():
        file, symbol = file_and_symbol.get()
        __print("Next file is %s" % str(next_file), print_lock)
        __print("Next symbol is %s" % str(symbol), print_lock)

        funds = data_prep.vectorize_funds('fundamentals/%s' % symbol)
        if funds is None:
            continue

        quarter, year = data_prep.get_date(file)
        if quarter is None:
            not_processed.no_q.append(file)
            continue
        if quarter is "no_date_found":
            print("No date found for file '%s'" % file)
            not_processed.no_diluted_eps.append(file)
            continue

        __print("putting %s into output queue" % str((file, symbol)))
        output.put((file, symbol))

def read_all_transcript_files(base_dir='data_backup/seeking_alpha'):
    all_transcripts = []
    for symbol in os.listdir(base_dir):
        all_transcripts += data_prep.read_all_files_for_symbol(symbol ,base_dir)
    return all_transcripts



if __name__ == '__main__':
   # files = Queue()
   # output = Queue()
   #
   # print_lock = Lock()
   #
   # files.put((1,2))
   # files.put((3,4))
   # files.put((4,5))
   # files.put((5,6))
   #
   # p1 = Process(target=supervisor, args=(files, output))
   # p2 = Process(target=supervisor, args=(files, output))
   #
   # sleep(1)
   # p1.start()
   # p2.start()
   #
   # p1.join()
   # p2.join()
   #
   # print("Output: %s" % str(get_all(output)))
   t = read_all_transcript_files()
   for u in t:
       if type(u) is not type((1,2)):
           raise RuntimeError("'%s' was not a tuple")
   print(t)




