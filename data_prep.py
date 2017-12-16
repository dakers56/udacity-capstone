import re
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

def alphanumerical(text):
    return re.sub(pattern="^[a-zA-Z0-9\s]", repl="", string=text)


stopwords = stopwords.words('english')
test_file = open('/Users/devonakers/udacity/capstone/data_backup/seeking_alpha/A/A_August 15, 2017 04:30 PM ET', 'r')
test_transcript = alphanumerical("".join(test_file.readlines()))
print("type(test_transcript): %s" % type(test_transcript))
print("type(stopwords): %s" % type(stopwords))


cnt_vec = CountVectorizer(input=test_transcript, stop_words=set(stopwords)).fit([test_transcript])
print("Final vocabulary:")
for v in cnt_vec.vocabulary_:
    print(str(v))
