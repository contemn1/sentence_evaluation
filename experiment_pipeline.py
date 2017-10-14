from IOUtil import load_numpy_arraies
import numpy as np
from binary import BinaryClassifierEval
import torch
from torch.utils.data import TensorDataset, DataLoader


DATA_PATH = "/home/zxj/Downloads/data/models/"


def read_data(positive_path, negative_path, sampler=None):
    sample = True if sampler is not None else False
    data = {"sample": sample}
    positive_vectors = load_numpy_arraies(positive_path)
    negative_vectors = load_numpy_arraies(negative_path)
    x = np.vstack((positive_vectors, negative_vectors))
    y = np.array([0]*positive_vectors.shape[0] + [1] * negative_vectors.shape[0])
    data["first"] = (x, y)
    if sample:
        train_positive, train_negative = sampler(positive_vectors)
        dev_positive, dev_negative = sampler(negative_vectors)
        train_x = np.vstack((train_positive, train_negative))
        train_y = np.array([0]*train_positive.shape[0] + [1] * train_negative.shape[0])
        dev_x = np.vstack((dev_positive, dev_negative))
        dev_y = np.array([0]*dev_positive.shape[0] + [1] * dev_negative.shape[0])
        data["first"] = (train_x, train_y)
        data["second"] = (dev_x, dev_y)

    return data


def random_sample(percentage):
    def sample_vectors(vectors):
        length = vectors.shape[0]
        indices = np.random.permutation(length)
        train_length = length*percentage
        return vectors[indices[:train_length]], vectors[indices[train_length:]]

    return sample_vectors


def dataset_generation(input_data):
    X, y = input_data
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    return TensorDataset(X_tensor, y_tensor)


def classfication(params):
    train_positive = params["train_positive"]
    train_negative = params["train_negative"]
    test_positive = params["test_positive"]
    test_negative = params["test_negative"]
    random_sampler = random_sample(0.8)
    all_data = read_data(train_positive, train_negative, random_sampler)
    train_data = dataset_generation(all_data["first"])
    params["dimension"] = all_data["first"][0].shape[1]

    dev_data = dataset_generation(all_data["second"])
    test_data = dataset_generation(read_data(test_positive, test_negative)["first"])
    train_loader = DataLoader(dataset=train_data, batch_size=128,
                              shuffle=False, num_workers=4, pin_memory=True)

    dev_loader = DataLoader(dataset=dev_data, batch_size=128,
                            shuffle=False, num_workers=4, pin_memory=True)

    test_loader = DataLoader(dataset=test_data, batch_size=128,
                             shuffle=False, num_workers=4, pin_memory=True)

    classifier = BinaryClassifierEval(train_loader, dev_loader, test_loader)
    res = classifier.run(params)
    print(res)

if __name__ == '__main__':
    params = {}
    params["train_positive"] = DATA_PATH + "all_positive_samples.npy"
    params["train_negative"] = DATA_PATH + "all_negative_samples.npy"
    params["test_positive"] = DATA_PATH + "all_positive_samples_test.npy"
    params["test_negative"] = DATA_PATH + "all_negative_samples_test.npy"
    params["usepytorch"] = True
    params["classifier"] = "MLP"
    classfication(params)
