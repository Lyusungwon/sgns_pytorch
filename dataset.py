import sys, os
import numpy as np
import pickle as pkl
import torch
from torch.utils.data import Dataset, ConcatDataset
import re
import random
import operator
from functools import reduce
from nltk.corpus import stopwords
import bz2
import _pickle as cPickle
from multiprocessing import Process, Pool
import multiprocessing as mp
from collections import Counter
from functools import wraps
import time
import csv
from torch.utils.data import DataLoader
import torch.nn as nn
from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize

def timefn(fn):
    def wrap(*args):
        t1 = time.time()
        result = fn(*args)
        t2 = time.time()
        print("@timefn:{} took {} seconds".format(fn.__name__, t2-t1))
        return result
    return wrap


class TextDataset(Dataset):
    def __init__(self, data_dir, dataset, window_size, ns_size, remove_th, subsam_th):
        self.dataset_dir = os.path.join(data_dir, dataset)
        data_config_list = [dataset, window_size, ns_size]
        self.data_file = '_'.join(map(str, data_config_list)) + '.pkl'
        self.file_dir = os.path.join(data_dir, self.data_file)
        self.window_size = window_size
        self.ns_size = ns_size
        self.rm_th = remove_th
        self.subsam_th = subsam_th
        self.stopwords = set(stopwords.words('english'))
        if not self.is_data_exist():
            self.make_dataset()

        with open(self.file_dir, 'rb') as f:
            self.word_pairs, self.neg_samples, self.vocabs, self.word2idx, self.idx2word = pkl.load(f)

    def is_data_exist(self):
        if os.path.isfile(self.file_dir):
            print("Data {} exist".format(self.data_file))
            return True
        else:
            print("Data {} does not exist".format(self.data_file))
            return False

    @timefn
    def preprocess_counter(self, tokenized):
        tokenized = reduce(operator.concat, tokenized)
        cnt = Counter(tokenized)
        #remove small words
        small_words = [key for key, value in cnt.items() if value < self.rm_th]
        for word in small_words:
            cnt.pop(word, None)
        #subsample
        prob = 1 - (self.subsam_th / (np.array(list(cnt.values())) / sum(cnt.values())))**0.5
        rm_idx = (np.random.random((len(prob),)) < prob)
        rm_words = np.array(list(cnt.keys()))[rm_idx]
        for word in rm_words:
            cnt.pop(word, None)
        # vocabs
        self.vocabs = list(cnt.keys())
        # calculate distribution
        power = 3 / 4
        probs = (np.array(list(cnt.values())) / sum(cnt.values()))**power
        self.probs = probs / probs.sum()
        self.stopwords = self.stopwords|set(small_words)|set(rm_words)
        return cnt

    def make_dataset(self):
        text = open(self.dataset_dir, encoding="utf-8").read().lower().strip()
        print("Start to make data")
        # words = word_tokenize(text)
        tokenized_text = self.tokenize(text)
        print("compelete tokenize")
        self.cnt = self.preprocess_counter(tokenized_text)
        print("Start to make data again")
        tokenized_text = self.tokenize(text)
        print("compelete tokenize again")
        word2idx, idx2word = self.map_word_idx(self.vocabs)
        word_pairs = []
        for sentence in tokenized_text:
            for i, word in enumerate(sentence):
                sentence = np.asarray(sentence)
                for w in range(-self.window_size, self.window_size + 1):
                    context_word_idx = i + w
                    if context_word_idx < 0 or context_word_idx >= len(sentence) or context_word_idx == i:
                        continue
                    word_pairs.append((word2idx[word], word2idx[sentence[context_word_idx]]))
        neg_samples = self.negative_sampling(len(word_pairs))
        neg_samples = [[word2idx[word] for word in neg_sample] for neg_sample in neg_samples]
        saves = word_pairs, neg_samples, self.vocabs, word2idx, idx2word
        with open(self.file_dir, 'wb') as f:
            cPickle.dump(saves, f, protocol=2)
            print("Data saved in {}".format(self.data_file))
    
    @timefn
    def tokenize(self, text):
        ps = PorterStemmer()
        text = re.sub('<.*>'," ", text)
        text = re.sub('[^A-Za-z.]+', " ", text)
        text = text.split(".")
        tokens_list = []
        for sen in text:
            tokens = []
            for word in sen.split():
                word = ps.stem(word)
                if word not in self.stopwords:
                    tokens.append(word)
            tokens_list.append(tokens)
        return tokens_list

    def map_word_idx(self, vocabs):
        word2idx = {}
        idx2word = {}
        for n, word in enumerate(vocabs):
            word2idx[word] = n
            idx2word[n] = word
        return word2idx, idx2word

    # @timefn
    def negative_sampling(self, batch):
        # neg_sample = np.random.choice(self.vocabs, self.ns_size, replace=True, p=probs)
        neg_sample = np.random.choice(self.vocabs, size=self.ns_size * batch, replace=True, p=self.probs)
        neg_sample = neg_sample.reshape(batch, self.ns_size)
        return neg_sample

    def _unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')

    def __getitem__(self, idx):
        center_idx, context_idx = self.word_pairs[idx]
        negs_idx = self.neg_samples[idx]
        center_idx, context_idx, negs_idx = torch.tensor(center_idx), torch.tensor(context_idx), torch.tensor(negs_idx)
        return center_idx, context_idx, negs_idx

    def __len__(self):
        return len(self.word_pairs)

if __name__ == '__main__':
    text_dataset = TextDataset('/home/sungwonlyu/data/corpus', 'the_lord_of_the_rings.txt', 5, 7, 5, 1e-04)
    # t2 = time.time()
    # print(t2-t1)
    # index = 1
    # print(text_dataset.word_pairs[index])
    # p = Pool(10)
    # p.map(trial, range(0,100))
    # p.close()
    # p.join()
