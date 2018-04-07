import sys
import joblib

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    print("Training sample neural net")
    in_file = sys.argv[1]
    print("Input file: %s" % in_file)

    data = joblib.load(in_file)
    X_, y_ = data.X_train, data.all_eps

    X_train, X_test, y_train, y_test = train_test_split(X_, y_)
    model = Sequential()
    model.add(Dense(1000, input_shape=((X_train.shape[1],))))
    model.add(Activation('relu'))
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dense(1))

    print("Compiling model.")
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    print("Done compiling model.")
    model.fit(X_train, y_train, epochs=128)

    score = model.evaluate(X_test, y_test)

    print("Score: %s" % score)

