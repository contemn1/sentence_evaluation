from IOUtil import load_numpy_arraies
import numpy as np
from binary import BinaryClassifierEval

DATA_PATH = "/home/zxj/Downloads/data/models/"


def generate_dataset(positive_path, negative_path, limit=1000):
    data = {}
    positive_vectors = load_numpy_arraies(positive_path)
    negative_vectors = load_numpy_arraies(negative_path)
    data["X"] = np.vstack((positive_vectors, negative_vectors))
    data["y"] = np.array([0]*positive_vectors.shape[0] + [1] * negative_vectors.shape[0])
    return data


def classfication(params):
    train_positive = params["train_positive"]
    train_negative = params["train_negative"]
    test_positive = params["test_positive"]
    test_negative = params["test_negative"]
    train_data = generate_dataset(train_positive, train_negative)
    test_data = generate_dataset(test_positive, test_negative)
    binary_classifier = BinaryClassifierEval(train=train_data, test=test_data)
    res = binary_classifier.run(params)
    print(res)

if __name__ == '__main__':
    params = {}
    params["train_positive"] = DATA_PATH + "all_positive_samples.npy"
    params["train_negative"] = DATA_PATH + "all_negative_samples.npy"
    params["test_positive"] = DATA_PATH + "all_positive_samples_test.npy"
    params["test_negative"] = DATA_PATH + "all_negative_samples_test.npy"
    params["usepytorch"] = True
    params["classifier"] = "MLP"
    params["kfold"] = 5
    classfication(params)
