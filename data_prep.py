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
from sklearn.utils.validation import _assert_all_finite
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
    try:
        X = X.sum(axis=0, dtype=np.float64).getA()[0]
        _assert_all_finite(X)
        return X
    except ValueError:
        print('Transcript contained invalid value' % X)
        return None


def vectorize_funds(file, clean_data=False):
    if not os.path.exists(file):
        print("Path to fundamentals '%s' did not exist." % file)
        return None
    funds = None
    if clean_data:
        return pd.read_csv(file)
    funds = pd.read_csv(file).drop('symbol', axis=1).drop('end_date', axis=1).drop('amend', axis=1).drop('doc_type',
                                                                                                         axis=1)
    # eps_key, diluted_eps_key = 'eps_basic', 'eps_diluted'
    # eps, diluted_eps = funds['eps_basic'], funds['eps_diluted']
    return funds

def df_str(df):
    s = "[ " + str(df[0])
    for d in df[1:]:
        s += "," + str(d)
    s += " ]"
    return s

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
    print('type(funds): %s' % type(funds))
    print('type(transcript): %s' % type(transcript))
    #return assert_finite(np.concatenate((transcript, np.array(funds)[0])))
    w = np.array(funds)[0]
       #return np.concatenate((transcript, np.array(funds)[0]))
    return np.array(funds)[0]

def is_finite(v):
    v1 = np.array([v])
    try:
        _assert_all_finite(v1)
        return True
    except ValueError:
        print("Value %s was not finite" % v)
        return False

def are_all_finite(v):
    for u in v:
        if not is_finite(u):
            return False
    return True


def assert_finite(v):
    try:
        _assert_all_finite(v)
        return v
    except ValueError:
        print('Concatenated vectors contained invalid value: %s' % v)
        return None

def funds_exist(symbol):
    return os.path.exists("fundamentals/%s" % symbol)

class UnprocessedFileList:
    def __init__(self):
        self.no_q = []
        self.no_y = []
        self.no_funds = []
        self.no_date_found = []
        self.no_eps = []
        self.no_diluted_eps = []

    def len(self):
        return len(self.no_q) + len(self.no_y) + len(self.no_funds) + len(self.no_date_found)


def get_input_data(cnt_vec, base_dir='data_backup/seeking_alpha'):
    X_train_ = None
    all_eps_ = None
    all_diluted_eps_ = None
    not_processed = UnprocessedFileList()
    for symbol in os.listdir(base_dir):
        print('symbol: %s' % symbol)
        transcript_path = "%s/%s" % (base_dir, symbol)
        funds = vectorize_funds('fundamentals/%s' % symbol, clean_data=True)
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
                if quarter is "no_date_found":
                    print("No date found for file '%s'" % file)
                    not_processed.no_diluted_eps.append(file)
                    continue
                fv, eps, diluted_eps  = feature_vector(cnt_vec, file, funds, quarter, year, not_processed)
                if fv is None:
                    not_processed.no_date_found.append(file)
                    continue
                if eps is None:
                    not_processed.no_eps.append(eps)
                    continue
                if diluted_eps is None:
                    not_processed.diluted_eps.append(diluted_eps)
                    continue
                if X_train_ is None:
                    #print("Shape fv: %s" % fv.shape)
                    X_train_ = np.array([fv])
                else:
                    #print("Shape fv: %s" % fv.shape)
                    #print('X_train: %s' % X_train_)
                    X_train_ = np.append(X_train_, np.array([fv]), axis=0)
                    #print("Shape X_train_: %s" % str(X_train_.shape))
                if all_eps_ is None:
                    all_eps_ = np.array([eps])
                else:
                    all_eps_ = np.append(all_eps_, np.array([eps]), axis=0)
                if all_diluted_eps_ is None:
                    all_diluted_eps_ = np.array([diluted_eps])
                else:
                    all_diluted_eps_ = np.append(all_diluted_eps_, np.array([diluted_eps]), axis=0)
        else:
            print("Fundamentals did not exist for %s" % symbol)
    return X_train_, all_eps_, all_diluted_eps_, not_processed


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


def get_matching_funds(funds, quarter, year, not_processed):
    print("Getting matching fundamentals.")
    print("Funds:\n%s" % str(funds))
    if year is 'no_year_found':
        print("No year found")
        return None, None, None
    print('quarter: %s' % quarter)
    print('year: %s' % year)
    match = funds[(funds.period_focus == quarter) & (funds.fiscal_year == year)]
    print('match:\n%s' % str(match))
    if match.empty or match is None:
        print("Did not find quarter and year in dataframe:\n%s" % match)
        return None, None, None
    if not are_all_finite(match):
        out = open('bad_tr_ex.out', 'a')
        out.write(df_str(w) + "\n")
        out.close()
        return None, None, None

    eps, diluted_eps = None, None
    try:
        print("eps match: %s" % match['eps_basic'])
        print("eps match iloc: %s" % match['eps_basic'].iloc[0])
        print("shape of eps match: %s" % str(match['eps_basic'].iloc[0].shape))
        eps, diluted_eps = match['eps_basic'].iloc[0].reshape(-1,1), match['eps_diluted'].iloc[0].reshape(-1,1)
        _assert_all_finite(eps)
    except KeyError as e:
        print("Caught key error: %s" % str(e))
        print("match: \n%s" % match)
        not_processed.no_eps.append(funds)
        return None, None, None
    except ValueError:
        not_processed.no_eps.append(match)
        print('eps value was not finite: %s' % eps)
        return None, None, None
    print('len(match): %s' % len(match))
    print('type(match): %s' % type(match))
    print('type(match[\'period_focus\']: %s' % type(match['period_focus']))
    print('type(match[\'period_focus\'.iloc(0)]: %s' % type(match['period_focus'].iloc[0]))
    m = match['period_focus']
    print("shape of m: %s" % m.shape)
    print("m : %s" % m)
    replacement = funds['period_focus'].iloc[0].replace('Q', '').replace('FY','')
    print("after replacement: %s" % replacement)
    try:
        match.period_focus.replace(to_replace=re.compile(".*"), value=replacement, inplace=True)
    except ValueError as e:
        print("Caught value error while attempting to determine quarter: %s" % str(e))
        return None, None, None
    funds = funds.drop(match.index, axis=0)
    print("Returning matching fundamentals:\n%s" % str(match))
    match = drop_label(match, 'eps_basic')
    match = drop_label(match, 'eps_diluted')
    match = drop_label(match, 'period_focus')
    match = drop_label(match, 'fiscal_year')
    #return match.drop('eps_basic', axis=1).drop('eps_diluted', axis=1).drop('period_focus',axis=1).drop('fiscal_year', axis=1), eps, diluted_eps)
    return match, eps, diluted_eps
def drop_label(df, label):
    print("Dropping column '%s' from dataframe:\n%s" % (label, str(df)))
    print("Shape before drop operation: %s" % str(df.shape))
    df = df.drop(label, axis=1)
    print("Shape after drop operation: %s" % str(df.shape))
    return df

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


def feature_vector(cnt_vec, transcript_file, funds, quarter, year, not_processed):
    print("-------------")
    print("Creating feature vector") 
    funds_vec, eps, diluted_eps = get_matching_funds(funds, quarter, year, not_processed)
    print("funds_vec: %s" % str(funds_vec))
    transcript_vec = vectorize_transcript(cnt_vec, transcript_file)  
    print("-------------")
    return cat_vectors(transcript_vec, funds_vec), eps, diluted_eps

def bad_rows(X, is_X=True):
    bad = []
    from sklearn.utils.validation import assert_all_finite
    for i in range(X.shape[0]):
        try:
            assert_all_finite(X[i])
        except ValueError:
            print("Index %s was not finite" % i)
            bad.append(i)
            print_bad(X[i], i, is_X)
    return bad


def print_bad(X_train, k, is_X=True):
    print('Dataset shape: %s' % X_train.shape)
    finite = np.isfinite(X_train)
    print("Bad entries in row %s" % k)
    for i in range(finite.shape[0]):
        if is_X:
            for j in range(X_train.shape[1]):
                if not finite[i][j]:
                    print("(%s,%s)  -> %s" % (i, j, X_train[i][j]))
        else:
            if not finite[i]:
                print("(%s,)  -> %s" % (i, X_train[i]))


def check_dim(X, y):
    x1 = X.shape[0]
    y1 = y.shape[0]
    if not (x1 == y1):
        print("X and y did not have same first dimension X: %s; y: %s." % (X, y))


def check_all_finite(X_train, y_train):
    bad = bad_rows(X_train)
    bad.append(bad_rows(y_train, is_X=False))
    print("Removing rows %s" % str(bad))
    if bad:
        X_train = np.delete(X_train, bad, axis=0)
        y_train = np.delete(y_train, bad, axis=0)
    return X_train, y_train


def check_finite(X):
    if not np.isfinite(X):
        return False
    return True


def write_to_file(X, file, ds_name):
    file = open(file, 'w')
    file.write(ds_name + ':\n')
    for x in X:
        file.write(str(x) + '\n')
    file.close()


def write_to_error(err_file, tr_ex, tr_lbl, i):
    err_file = open(err_file, 'a')
    err_file.write(str(i) + "|"   + str(tr_ex) + ' | ' + str(tr_lbl) + "\n")
    err_file.close()


def validate_data(tr_ex, tr_lbl, i):
    clf = LinearRegression()
    try:
        clf.fit(tr_ex, tr_lbl.reshape(-1,1))
    except ValueError as e:
        print("Encountered ValueError while training data: %s" % str(e))
        print("Training example: %s" % tr_ex)
        print("Training label: %s" % tr_lbl)
        write_to_error('bad_data.out', tr_ex, tr_lbl, i)


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

    for i in range(all_input.shape[0]):
        validate_data(all_input[i], all_eps[i], i)    

    print("Number of processed vectors: %s" % all_input.size)
    print("Number of unprocessed vectors: %s" % str(not_processed.len()))
    print('Reasons for not processing:')
    print('No year: %s' % not_processed.no_y)
    print('No quarter: %s' % not_processed.no_q)
    print('No fundamentals file: %s' % not_processed.no_funds)
    print('Could not parse data from file: %s' % not_processed.no_date_found)

    print("Performing linear regression of transcripts and fundamentals vs basic eps.")
    print("shape of all_eps: %s" % str(all_eps.shape))
    print("shape of all_input: %s" % str(all_input.shape))
    print("All input:\n%s" % all_input)
    X_train, X_test, y_train, y_test = train_test_split(all_input, all_eps)
    lin_reg = LinearRegression()
    print("Fitting data")
    print("Performance on test data:")
    lin_reg.fit(X_train, y_train)
    print(lin_reg.score(X_test, y_test))
    now = time.clock() - now
    print("Process took %s seconds." % (now / 1000))
    #
    # print("Before removing bad rows:")
    # print("X_train shape: %s" % str(X_train.shape))
    # write_to_file(X_train, 'x_train.out', 'x_train')
    # print("y_train shape: %s" % str(y_train.shape))
    # write_to_file(y_train, 'y_train.out', 'y_train')
    # print("X_test shape: %s" % str(X_test.shape))
    # write_to_file(X_train, 'x_test.out', 'x_test')
    # print("y_test shape: %s" % str(y_test.shape))
    # write_to_file(X_train, 'y_test.out', 'y_test')
    # check_dim(X_train, y_train)
    # check_dim(X_test, y_test)


    # print("X_train shape: %s" % str(X_train.shape))
    # print("X_test shape: %s" % str(X_test.shape))
    # print('Shape of y_train: %s' % str(y_train.shape))
    # print('Shape of y_test: %s' % str(y_test.shape))

