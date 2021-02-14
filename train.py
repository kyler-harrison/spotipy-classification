import pandas as pd
import numpy as np
import pickle
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# TODO separate some validation data
# TODO try more data (different genres - label input csv file names better better)
# TODO balance dataset (over/under sample for a more equal distribution)
# load data into x_train, x_test, y_train, y_test arrays
# inputs:
#        data_path: path to csv file of features and artist labels
#        start_col: starting feature column name of the csv
#        end_col: ending feature column name of the csv
#        label_col: name of the class label column (the artist name)
#        scaler_path: path to save scaler to (if you want to model on new data, will need to scale it)
#        neural_net: option to reshape data for a simple nn input, set to False if doing any other sklearn classifier
def load_data(data_path, start_col, end_col, label_col, scaler_path, neural_net=False):
    df = pd.read_csv(data_path)  # NOTE it is assumed the data file has a column name header
    x = np.array(df.loc[:, start_col:end_col])  # all rows, from the start col to the end col
    y = np.array(df[label_col])  # column of artist name labels

    # encode string artist labels into integer indexes
    labelEncoder = LabelEncoder()
    labelEncoder.fit(y)
    y = labelEncoder.transform(y) 

    # standardize data (makes a normal distribution)
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    # save scaler to file for reuse
    pickle.dump(scaler, open(scaler_path, "wb"))

    # split data into train/test sets
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

    # neural network input needs a different shape
    if neural_net:
        x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
    
    return x_train, x_test, y_train, y_test


# save a trained classifier to a pickle file, if it's a tensorflow model save the tensorflow way
def save_sklearn_model(model, model_path):
    pickle.dump(model, open(model_path, "wb"))


# reload in a saved sklearn model and its corresponding scaler, can use the returned model and scaler as they are used in any other function
def load_sklearn_model(model_path, scaler_path):
    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
    return model, scaler


# naive bayes classification model implementation
def naive_bayes_classifier(x_train, x_test, y_train, y_test):
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"naive bayes accuracy = {accuracy}")
    return gnb


# decision tree classification model implementation
def decision_tree_classifier(x_train, x_test, y_train, y_test):
    dt_cls = DecisionTreeClassifier()
    dt_cls.fit(x_train, y_train)
    y_pred = dt_cls.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"decision tree accuracy = {accuracy}")
    return dt_cls


# k nearest neighbors classification model implementation
def knn_classifier(x_train, x_test, y_train, y_test, neighbors):
    knn_cls = KNeighborsClassifier(n_neighbors=neighbors)
    knn_cls.fit(x_train, y_train)
    accuracy = knn_cls.score(x_test, y_test)
    print(f"knn accuracy = {accuracy}")
    return knn_cls


# support vector machine classification model implementation
# kernel should be "linear", "rbf", or "poly"
def svm_classifier(x_train, x_test, y_train, y_test, kernel):
    svm_cls = svm.SVC(kernel=kernel, C=2)
    svm_cls.fit(x_train, y_train)
    y_pred = svm_cls.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"svm with {kernel} kernel accuracy = {accuracy}")
    return svm_cls


# simple tensorflow neural network for classification
# could probably tweak hyperparameters and mess with number of layers and their types for better results
def train_nn(x_train, x_test, y_train, y_test, num_epochs, model_path):
    # initialize a sequential network
    model = Sequential()

    # input layer, shape is the shape of one feature vector
    model.add(Flatten(input_shape=x_train.shape[1:]))

    # add a dense layer with 16 nodes, relu activation, and 20% dropout
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.2))

    # get number of classes in data, it is assumed that all classes show up in y_train
    num_classes = len(set(y_train))  

    # last layer with softmax probability output on however many classes in the dataset
    model.add(Dense(num_classes, activation="softmax"))

    # adam optimizer, consider messing with hyperparams here
    opt = Adam(lr=1e-3, decay=1e-5)

    # set up model
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # train model 
    model.fit(x_train, y_train, epochs=num_epochs, validation_data=(x_test, y_test))

    # see how model does - should have probably put some validation aside at the start
    loss, acc = model.evaluate(x_test, y_test)
    print(f"loss = {loss}, accuracy = {acc}")

    # save model to file for reuse
    model.save(model_path)


# main that runs all classifiers
def main():
    # vars for the data load
    data_path = "data/tracks.csv"
    start_col = "danceability"
    end_col = "time_signature"
    label_col = "artist"
    scaler_path = "scalers/scaler.pkl"
    x_train, x_test, y_train, y_test = load_data(data_path, start_col, end_col, label_col, scaler_path)

    # example runs of each model
    gnb_cls = naive_bayes_classifier(x_train, x_test, y_train, y_test)
    dt_cls = decision_tree_classifier(x_train, x_test, y_train, y_test)
    knn_cls = knn_classifier(x_train, x_test, y_train, y_test, 9)
    svm_linear_cls = svm_classifier(x_train, x_test, y_train, y_test, "linear")
    svm_poly_cls = svm_classifier(x_train, x_test, y_train, y_test, "poly")
    svm_rbf_cls = svm_classifier(x_train, x_test, y_train, y_test, "rbf")

    # example save of a model
    save_sklearn_model(dt_cls, "models/dt_cls.pkl")
    
    # example reload and usage of saved model (scaler not used)
    reloaded_dt_cls, reloaded_sclaer = load_sklearn_model("models/dt_cls.pkl", "scalers/scaler.pkl")
    reloaded_predictions = reloaded_dt_cls.predict(x_test)
    reloaded_accuracy = metrics.accuracy_score(y_test, reloaded_predictions)
    # this accuracy should be the same as the accuracy of the original model
    print(f"accuracy of reloaded decision tree classifier = {reloaded_accuracy}")

    # reload data with nn reshape (a wee bit inefficient but whatever)
    x_train, x_test, y_train, y_test = load_data(data_path, start_col, end_col, label_col, scaler_path, neural_net=True)
    num_epochs = 100
    train_nn(x_train, x_test, y_train, y_test, num_epochs, "models/simple_nn.model")


if __name__ == "__main__":
    main()


