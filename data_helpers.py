import numpy as np
import re
import itertools
from collections import Counter
import os
import pickle


def load_data_and_labels():

    with open('x_text_file.txt', 'rb') as f1:
        x_text = pickle.load(f1)

    f1.close()

    with open('y_file.txt', 'rb') as f2:
        y = pickle.load(f2)

    f2.close()

    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
