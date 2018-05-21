import numpy as np
import re
import itertools
from collections import Counter
import pandas as pd


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def get_coefs(word, *arr): 
    return word, np.asarray(arr, dtype='float32')


def load_data_and_labels():
    # Split by words
    x_train = pd.read_csv('./data/train.csv').comment_text
    y_train = pd.read_csv('./data/train.csv', usecols=['toxic','severe_toxic','obscene','threat','insult','identity_hate']).fillna("list").values
    x_train = x_train.apply(clean_str).fillna('fillna')
    x_test = pd.read_csv('./data/test.csv').comment_text
    x_test = x_test.apply(clean_str).fillna('fillna')
    y_test = pd.read_csv('./data/test_labels.csv', usecols=['toxic','severe_toxic','obscene','threat','insult','identity_hate']).fillna("list").values
    

    
    return [x_train, y_train, x_test, y_test]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
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


def blocks(data, block_size):
    """
    For Test data. Can use generate_batch function.
    """
    data = np.array(data)
    data_size = len(data)
    print(data_size)
    nums = int((data_size-1)/block_size) + 1
    for block_num in range(nums):
        if block_num == 0:
            print("prediction start!")
        start_index = block_num * block_size
        end_index = min((block_num + 1) * block_size, data_size)
        yield data[start_index:end_index]
