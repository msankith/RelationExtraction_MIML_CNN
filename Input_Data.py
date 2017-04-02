'''
Created on 05-Nov-2016

@author: suvadeep

Modified version of input_data.py from https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist
'''

import numpy as np
import random
import os
import json

class Data_Set(object):
    def __init__(self, words, labels):
        self._words = words
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = words.shape[0]

    @property
    def words(self):
        return self._words

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._words = self._words[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._words[start:end], self._labels[start:end]


class Data_Sets(object):
    def __init__(self):
        self.train = []
        self.validation = []
        self.test = []

    def read_text(self, words_file):
        with open(words_file) as f:
            words = [w.strip() for w in f.readlines()]
        
        char_set = sorted(set(''.join(words)))
        char_mapping = {ch: i for i, ch in enumerate(char_set)}
        
        data = np.zeros((len(words), len(words[0]), len(char_set)), np.int8)
        for i, w in enumerate(words):
            for j, ch in enumerate(w):
                data[i, j, char_mapping[ch]] = 1 
                
        return data
        
    def read_labels(self, labels_file):
        with open(labels_file) as f:
            labels = [int(l) for l in f.readlines()]
        labels = np.array(labels, np.int8)
        return labels

    
    
    def read_data_sets(self, params):
        data_dir = params["data_dir"]
        num_train_examples = params["num_train_examples"]
        num_validation_examples = params["num_validation_examples"]
        num_test_examples = params["num_test_examples"]
        word_length = params["word_length"]
        num_chars = params["num_chars"]
        
        # Check whether the parameters match or not
        data_set_params = json.loads(open(os.path.join(data_dir, 'params.txt')).read())
        assert params["num_chars"] == data_set_params["num_chars"], \
                                                    'Dataset parameter mismatch'
        assert params["word_length"] == data_set_params["word_length"], \
                                                    'Dataset parameter mismatch'
        assert params["num_class"] == data_set_params["num_class"], \
                                                    'Dataset parameter mismatch'
        assert params["num_input_symbols"] == data_set_params["num_input_symbols"], \
                                                    'Dataset parameter mismatch'

        
        words = self.read_text(os.path.join(data_dir, 'words.txt'))
        labels = self.read_labels(os.path.join(data_dir, 'labels.txt'))

        assert words.shape[2] == num_chars, 'Inconsistent datasets'
        assert words.shape[0] == labels.shape[0], 'Inconsistent datasets'
        assert word_length == words.shape[1], 'Inconsistent datasets'
        assert labels.shape[0] >= (num_train_examples+num_validation_examples+ \
                        num_test_examples), 'Not enough examples in the file'
                        
        # Perform an initial random permutation
        perm = np.arange(labels.shape[0])
        np.random.shuffle(perm)
        words = words[perm]
        labels = labels[perm]
            
        self.train = Data_Set(words[:num_train_examples], labels[:num_train_examples])
        self.validation = Data_Set(words[num_train_examples: num_train_examples+num_validation_examples], 
                                labels[num_train_examples: num_train_examples+num_validation_examples])
        self.test = Data_Set(
                    words[num_train_examples+num_validation_examples: num_train_examples+num_validation_examples + num_test_examples], \
                    labels[num_train_examples+num_validation_examples: num_train_examples+num_validation_examples + num_test_examples])
        
