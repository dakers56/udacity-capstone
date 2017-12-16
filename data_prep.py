import re
import os

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


def alphanumerical(text):
    return re.sub(pattern="^[a-zA-Z0-9\s]", repl="", string=text)


def build_corpus(folders):
    corpus = []
    for folder in folders:
        print("Folder: %s" % folder)
        for file in os.listdir(folder):
            print("File: %s" % file)
            file = open("%s/%s" % (folder, file))
            corpus.append(alphanumerical("".join(file.readlines())))
            file.close()
    return corpus


stopwords = stopwords.words('english')
test_file = open('/Users/devonakers/udacity/capstone/data_backup/seeking_alpha/A/A_August 15, 2017 04:30 PM ET', 'r')
test_transcript = alphanumerical("".join(test_file.readlines()))
corpus = build_corpus(['/Users/devonakers/udacity/capstone/data_backup/seeking_alpha/A'])

print("Fitting CountVectorizer to determine vocabulary.")
print("Corpus: %s" % corpus)
cnt_vec = CountVectorizer(input=test_transcript, stop_words=set(stopwords)).fit(corpus)
print("Final vocabulary:")
for v in cnt_vec.vocabulary_:
    print(str(v))
