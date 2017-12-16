import re

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


def alphanumerical(text):
    return re.sub(pattern="^[a-zA-Z0-9\s]", repl="", string=text)


stopwords = stopwords.words('english')
test_file = open('/Users/devonakers/udacity/capstone/data_backup/seeking_alpha/A/A_August 15, 2017 04:30 PM ET', 'r')
test_transcript = alphanumerical("".join(test_file.readlines()))
corpus = [test_transcript]

print("Fitting CountVectorizer to determine vocabulary.")
print("Corpus: %s" % corpus)
cnt_vec = CountVectorizer(input=test_transcript, stop_words=set(stopwords)).fit(corpus)
print("Final vocabulary:")
for v in cnt_vec.vocabulary_:
    print(str(v))
