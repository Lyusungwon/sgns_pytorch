import sys, os
import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from torch.utils.data import DataLoader
from dataset import TextDataset
from random import Random
from torch import distributed, nn
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
def collate_text(list_inputs):
    batch = len(list_inputs)
    center_list = [len(list_inputs[i][0]) for i in range(batch)]
    max_len_center = max(center_list)
    padded_center = torch.zeros(batch, max_len_center, dtype=torch.long)
    context_list = [len(list_inputs[i][1]) for i in range(batch)]
    max_len_context = max(context_list)
    padded_context = torch.zeros(batch, max_len_context, dtype=torch.long)
    for i in range(batch):
        padded_center[i,:center_list[i]] = list_inputs[i][0]
        padded_context[i, :context_list[i]] = list_inputs[i][1]
    neg = []
    neg_size = len(list_inputs[0][2])
    for k in range(neg_size):
        neg_len = [len(list_inputs[i][2][k]) for i in range(batch)]
        max_len_neg = max(neg_len)
        padded_neg = torch.zeros(batch, max_len_neg, dtype=torch.long)
        for i in range(batch):
            padded_neg[i,:neg_len[i]] = list_inputs[i][2][k]
        neg.append((padded_neg, neg_len))
    return (padded_center, center_list), (padded_context, context_list), neg

def collate_word(words):
    words.sort(key=lambda x: len(x), reverse=True)
    length_list = [len(words[i]) for i in range(len(words))]
    padded_words = pad_sequence(words).transpose(0,1)
    return padded_words, length_list

def collate_pretrained(list_inputs):
    batch = len(list_inputs)
    words = [list_input[0] for list_input in list_inputs]
    words_len = [len(word) for word in words]
    embeddings = torch.cat([list_input[1] for list_input in list_inputs], 0)
    max_len = max(words_len)
    padded_words = torch.zeros(batch, max_len, dtype=torch.long)
    for i in range(batch):
        padded_words[i, :words_len[i]] = words[i]
    return padded_words, words_len, embeddings

class Partition(object):
    def __init__(self, data, index):
        self.data = data
        self.index = index
         
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataParitioner(object):
    def __init__(self, data, sizes, seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]
    
    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

class TextDataLoader(DataLoader):
    def __init__(self, batch_size, multinode, num_workers, data_dir, dataset, window_size, ns_size, remove_th, subsample_th):
        self.dataset = TextDataset(data_dir, dataset, window_size, ns_size, remove_th, subsample_th)
        self.vocabs = self.dataset.vocabs
        self.word2idx = self.dataset.word2idx
        if multinode:
            size = distributed.get_world_size()
            batch_size = int(batch_size / float(size))
            partition_sizes = [1.0 / size for _ in range(size)]
            partition = DataParitioner(self.dataset, partition_sizes)
            self.dataset = partition.use(distributed.get_rank())
        super(TextDataLoader, self).__init__(self.dataset, batch_size, num_workers=num_workers, shuffle=True)


if __name__ == '__main__':
    text_loader = TextDataLoader(32, False, 1, '/home/sungwonlyu/data/corpus', 'the_lord_of_the_rings.txt', 5, 7, 5, 1e-04)
    for i, (center_word, context_word, ns_words) in enumerate(text_loader):
        print(center_word)
        print(context_word)
        print(ns_words)
        if i > 10:
            break
