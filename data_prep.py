import re
import os
import math
import joblib
import dateutil.parser
import datetime
import pandas as pd
import time
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np


def alphanumerical(text):
    return re.sub(pattern=r'[\\]x[0-9a-eA-Z]{1,2}', repl="",
                  string=re.sub(pattern="[^a-zA-Z0-9\s]", repl="", string=text)).replace("\n", "")


def stem_file(file):
    print("Stemming file: %s" % file)
    lines = [alphanumerical(line) for line in file.readlines()]
    words = []
    for line in lines:
        for word in line.split(" "):
            words.append(word)
    return [PorterStemmer().stem(word) for word in words if word not in set(stopwords.words('english'))]


def unpack_elements(nested_list):
    unpacked = []
    for i in range(len(nested_list)):
        for element in nested_list[i]:
            unpacked.append(element)
    return unpacked


def build_corpus(data=None):
    corpus = []
    for folder in data if data else data_folders():
        for file in os.listdir(folder):
            file = open("%s/%s" % (folder, file))
            for word in stem_file(file):
                corpus.append(word)
            file.close()
    return corpus


def data_folders():
    folders = []
    base_dir = 'data_backup/seeking_alpha'
    for folder in os.listdir(base_dir):
        folders.append("%s/%s" % (base_dir, folder))
    return folders


def symbols():
    symbols = set()
    for path in data_folders():
        symbols.add(path.split("/")[-1])
    return symbols


def path_to_symbol(symbol):
    return 'data_backup/seeking_alpha/%s' % symbol


def all_transcript_samples(cnt_vec):
    """

    :param cnt_vec: count vectorizer
    :return: X_train as shuffled dataset, files they come from. Files will be used to pick out dates.
    """
    X_train = []
    files = []
    for symbol in symbols():
        path = path_to_symbol(symbol)
        for file in os.listdir(path):
            X_train.append(vectorize_transcript(cnt_vec, "%s/%s" % (path, file)))
            files.append(symbol)

    return train_test_split(X_train, files)


def write_vectorizer(cnt_vec, folder='./', testing=True):
    now = str(dateutil.parser.parse(str(datetime.datetime.now()))).replace(" ", "_")
    if not testing:
        joblib.dump(cnt_vec, "%svocab_%s.pkl" % (folder, now))
        return
    joblib.dump(cnt_vec, "vocab.pkl")


def get_vectorizer(corpus):
    return CountVectorizer(max_features=1000).fit(corpus)


def transform(cnt_vec, string):
    return cnt_vec.transform(string)


def vectorize_transcript(cnt_vec, file):
    file = open(file, 'r')
    X = transform(cnt_vec, file.readlines())
    file.close()
    return X.sum(axis=0, dtype=np.float64).getA()[0]


def vectorize_funds(file):
    if not os.path.exists(file):
        print("Path to fundamentals '%s' did not exist." % file)
        return None
    funds = pd.read_csv(file).drop('symbol', axis=1).drop('end_date', axis=1).drop('amend', axis=1).drop('doc_type',
                                                                                                         axis=1)
    # eps_key, diluted_eps_key = 'eps_basic', 'eps_diluted'
    # eps, diluted_eps = funds['eps_basic'], funds['eps_diluted']
    return funds


def cat_vectors(transcript, funds):
    print("transcript.shape: %s" % transcript.shape)
    # print("transcript: %s" % transcript)
    if funds is None:
        print('Found None as funds')
        return None
    print("funds.shape: %s" % str(funds.shape))
    print("np.array(funds).shape: %s" % str(np.array(funds)[0].shape))
    print(funds)
    if funds.empty:
        return None
    # print('type(funds): %s' % type(funds))
    # print('type(transcript): %s' % type(transcript))
    return np.concatenate((transcript, np.array(funds)[0]))


def funds_exist(symbol):
    return os.path.exists("fundamentals/%s" % symbol)


class UnprocessedFileList:
    def __init__(self):
        self.no_q = []
        self.no_y = []
        self.no_funds = []
        self.no_date_found = []

    def len(self):
        return len(self.no_q) + len(self.no_y) + len(self.no_funds) + len(self.no_date_found)


def get_input_data(cnt_vec, base_dir='data_backup/seeking_alpha'):
    X_train_ = []
    all_eps_ = []
    all_diluted_eps_ = []
    not_processed = UnprocessedFileList()
    print("Type X_train: %s" % type(X_train_))
    for symbol in os.listdir(base_dir):
        print('symbol: %s' % symbol)
        transcript_path = "%s/%s" % (base_dir, symbol)
        funds = vectorize_funds('fundamentals/%s' % symbol)
        if funds is None:
            print("Fund was None")
            continue
        if funds_exist(symbol):

            for file in os.listdir(transcript_path):
                file = "%s/%s/%s" % (base_dir, symbol, file)
                print("File: %s" % file)
                quarter, year = get_date(file)
                if quarter is None:
                    not_processed.no_q.append(file)
                    continue
                these_funds, eps, diluted_eps = get_matching_funds(funds, quarter, year)
                if these_funds is None:
                    not_processed.no_funds.append(file)
                    continue
                fv = feature_vector(cnt_vec, file, these_funds)
                if fv is None:
                    not_processed.no_date_found.append(file)
                    continue
                print("Appending to X_train, which is type %s" % type(X_train_))
                # X_train_.append(fv)
                # all_eps_.append(eps)
                # all_diluted_eps_.append(diluted_eps)
                try:
                    print("X_train: %s" % X_train_)
                    print("Shape of X_train: %s" % X_train_.shape)
                    print("fv: %s" % fv)
                    print("Shape of fv: %s" % fv.shape)
                    X_train_.append(fv)
                except AttributeError:
                    np.concatenate((X_train_, fv), axis=0)
                try:
                    print("all_eps: %s" % all_eps_)
                    print("eps: %s" % eps)
                    all_eps_.append(eps)
                except AttributeError:
                    print("Shape of all_eps: %s" % str(all_eps_.shape))
                    print("Shape of np.array(eps): %s" % str(np.array([eps]).shape))
                    all_eps_ = np.concatenate((all_eps_, np.array([eps])), axis=0)
                try:
                    print("all_diluted_eps: %s" % all_diluted_eps_)
                    print("diluted_eps: %s" % diluted_eps)
                    all_diluted_eps_.append(diluted_eps)
                except AttributeError:
                    print("Shape of all_diluted_eps: %s" % str(all_diluted_eps_.shape))
                    print('dluted_eps: %s' % str(diluted_eps))
                    print("Shape of diluted_eps: %s" % str(np.array([diluted_eps]).shape))
                    all_diluted_eps_= np.concatenate(((all_diluted_eps_, np.array([diluted_eps]))), axis=0)
                for x in X_train_:
                    print("x: %s" % x)
                    print("type(x): %s" % type(x))
                print('X_train: %s' % X_train_)
                X_train_ = np.array(X_train_, dtype=np.float64)
                print("X_train: %s" % X_train_)
                print("all_eps: %s" % all_eps_)
                all_eps_ = np.array(all_eps_)
                all_diluted_eps_ = np.array(all_diluted_eps_)
        else:
            print("Fundamentals did not exist for %s" % symbol)
    return np.array(X_train_), np.array(all_eps_), np.array(all_diluted_eps_), not_processed


def all_funds():
    all = None
    base_dir = "fundamentals"
    for file in os.listdir(base_dir):
        path = "%s/%s" % (base_dir, file)
        df = vectorize_funds(path)
        if all is None:
            all = df
        else:
            all = all.append(df)
    return all


def get_matching_funds(funds, quarter, year, drop_and_replace=False):
    if year is 'no_year_found':
        print("No year found")
        return None, None, None
    print('quarter: %s' % quarter)
    print('type(quarter): %s' % type(quarter))
    print('year: %s' % year)
    print('type(year): %s' % type(year))
    funds1 = funds
    print("Fundamentals df:\n" % funds)
    match = funds[(funds.period_focus == quarter) & (funds.fiscal_year == year)]
    if match.empty or match is None:
        print("Did not find quarter and year in datafram:\n%s" % match)
        return None, None, None
    eps, diluted_eps = None, None
    try:
        eps, diluted_eps = match['eps_basic'].iloc[0], match['eps_diluted'].iloc[0]
    except KeyError as e:
        print(funds)
        raise RuntimeError("Caught key error", e)
    print('len(match): %s' % len(match))
    print('type(match): %s' % type(match))

    print('type(match[\'period_focus\']: %s' % type(match['period_focus']))
    print('type(match[\'period_focus\'.iloc(0)]: %s' % type(match['period_focus'].iloc[0]))
    if drop_and_replace:
        m = match['period_focus']
        print("m : %s" % m)
        m.iloc[0] = int(funds['period_focus'].iloc[0].replace('Q', ''))
        funds = funds.drop(match.index, axis=0)
    # funds['fiscal_year']
    print("Returning matching fundamentals:\n%s" % str(match))
    return match, eps, diluted_eps
    # return match.copy().drop('eps_basic', axis=1).drop('eps_diluted', axis=1), eps, diluted_eps


def swap_str(string, ch1, ch2):
    ch1_index = string.index(ch1)
    ch2_index = string.index(ch2)
    as_list = list(string)
    as_list[ch1_index] = ch2
    as_list[ch2_index] = ch1
    return str(as_list)


def swap_at_ind(string, i1, i2):
    string = list(string)
    temp = string[i1]
    string[i1] = string[i2]
    string[i2] = temp
    return "".join(string)


def swap_NQ(nq_or_qn):
    return swap_at_ind(nq_or_qn, 0, 1)


def get_NQYYYY(string):
    is_nqyy = re.search("[0-4]Q[0-9]{2,4}", string)
    if is_nqyy:
        match = swap_NQ(is_nqyy.group(0))
        return match[:2], get_Y(match[2:])
    is_nqyy = re.search("Q[0-4][0-9]{2,4}", string)
    if is_nqyy:
        match = is_nqyy.group(0)
        return match[:2], get_Y(match[2:])
    is_nqyy = re.search(r'Q[0-4]\s[0-9]{2,4}', string)
    if is_nqyy:
        match = is_nqyy.group(0)
        return match[:2], get_Y(match[2:])
    is_nqyy = re.search('F[1-4]Q\\s[0-9]{2,4}', string)
    if is_nqyy:
        match = is_nqyy.group(0)
        return match[1:3], get_Y(match[-2:])
    is_nqyy = re.search(r'F[0-4]Q\s[0-9]{4}', string)
    if is_nqyy:
        match = swap_NQ(is_nqyy.group(0))
        return match[1:3], get_Y(match[-2:])
    print("Did not find matching date pattern. String: '%s'" % string)
    return None, None


def get_N(qn_or_nq):
    return re.search("[0-4]", qn_or_nq).group(0)


def get_Y(string, debug=True):
    string = string.strip()
    if len(string) > 4:
        raise RuntimeError("Got unexpected year format: more than four digits in '%s'" % string)
    if len(string) == 4:
        return int(string)
    if len(string) == 3:
        if (debug):
            print("3 digit string: %s" % string)
        print("Returning string '%s'" % "2".join(string))
        return int("2".join(string))
    return int("20".join(string))


def get_date(file):
    file = open(file, 'r')
    for line in file.readlines():
        pat1 = [r'F[0-4]Q\s[0-9]{4}', 'F[0-4]Q[0-9]{,4}', r'Q[0-4]\s[0-9]{,4}', 'Q[0-4] [0-9]{2,4}',
                'F[1-4]Q\\s[0-9]{2,4}']
        # pat1 = ['F[0-4]Q[0-9]{,4}', r'Q[0-4]\s[0-9]{,4}', 'Q[0-4] [0-9]{2,4}', 'F[1-4]Q\\s[0-9]{2,4}']

        for p in pat1:
            m = re.search(p, line)
            if m:
                try:
                    return get_NQYYYY(m.group(0))
                except RuntimeError:
                    break
    return "no_date_found", "no_year_found"


def feature_vector(cnt_vec, transcript_file, funds):
    quarter, year = get_date(transcript_file)
    if quarter is "no_date_found":
        print("No date found for file '%s'" % transcript_file)
        return None
    funds_vec, _, _ = get_matching_funds(funds, quarter, year, drop_and_replace=True)
    transcript_vec = vectorize_transcript(cnt_vec, transcript_file)
    if funds_vec is None:
        print("Funds was 'None'")
        return None
    return cat_vectors(transcript_vec, funds_vec)


if __name__ == '__main__':
    print("Training model for capstone project")
    now = time.clock()
    cnt_vec = None
    vocab_path = 'vocab.pkl'

    if not os.path.exists(vocab_path):
        corpus = build_corpus()
        cnt_vec = get_vectorizer(corpus)
        write_vectorizer(cnt_vec)
    else:
        cnt_vec = joblib.load(vocab_path)

    all_input, all_eps, all_diluted_eps, not_processed = get_input_data(cnt_vec)
    print("Number of processed vectors: %s" % all_input.size)
    print("Number of unprocessed vectors: %s" % str(not_processed.len()))
    print('Reasons for not processing:')
    print('No year: %s' % not_processed.no_y)
    print('No quarter: %s' % not_processed.no_q)
    print('No fundamentals file: %s' % not_processed.no_funds)
    print('Could not parse data from file: %s' % not_processed.no_date_found)

    print("Performing linear regression of transcripts and fundamentals vs basic eps.")
    # all_eps = all_eps.apply(math.log)
    # all_eps = pd.Series(all_eps)
    print("type all_eps: %s" % all_eps)
    print("type all_input: %s" % all_input)
    X_train, X_test, y_train, y_test = train_test_split(all_input, all_eps)
    # print("X_train type: %s" % type(X_train))
    # print("X_test type: %s" % type(X_test))
    # print("y_train: %s" % type(y_train))
    # print("y_test: %s" % type(y_test))
    print("X_train: %s" % X_train)
    print("X_test: %s" % X_test)
    # print("y_train: %s" % y_train)
    # print("y_test: %s" % y_test)
    # print("Elements of arrays:")
    # print("el 0: %s" % X_train[0])
    # print("type(el 0): %s" % type(X_train[0]))
    # print("type el -1: %s" % type(X_train[-1]))
    # print("el -1: %s" % X_train[-1])
    lin_reg = LinearRegression()
    print("Fitting data")
    print("Performance on test data:")
    print('Shape of X_train: %s' % X_train.shape)
    print('Shape of y_train: %s' % y_train.shape)
    lin_reg.fit(X_train, y_train)
    # lin_reg.fit(X_train, X_train)
    # lin_reg.fit(X_train, X_train)
    # lin_reg.fit(y_test, y_test)
    # lin_reg.fit(y_train, y_train)
    print(lin_reg.score(X_test, y_test))
    now = time.clock() - now
    print("Process took %s seconds." % (now / 1000))
