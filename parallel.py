# Trying to parallelize stemming
import threading
from nltk.stem import PorterStemmer
from multiprocessing import Pool

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


if __name__ == '__main__':
   data = ["thread_%s" % i for i in range(512)]
   print(partition(data))
