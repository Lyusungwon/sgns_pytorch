from torch.utils.data import DataLoader
from dataset import TextDataset
from random import Random
from torch import distributed, nn


class Partition(object):
    def __init__(self, data, index):
        self.data = data
        self.vocabs = data.vocabs
        self.index = index

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataParitioner(object):
    def __init__(self, data, sizes, seed=2):
        self.data = data
        self.partitions = []
        self.ground_truth = data.ground_truth
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
    def __init__(self, batch_size, multinode, num_workers, data_dir, dataset, window_size, ns_size, remove_th, subsample_th, embedding_size, seed):
        self.multinode = multinode
        self.dataset = TextDataset(data_dir, dataset, window_size, ns_size, remove_th, subsample_th, embedding_size, seed)
        self.vocabs = self.dataset.vocabs
        self.word2idx = self.dataset.word2idx
        if self.multinode:
            size = distributed.get_world_size()
            batch_size = int(batch_size / float(size))
            partition_sizes = [1.0 / size for _ in range(size)]
            partition = DataParitioner(self.dataset, partition_sizes)
            self.dataset = partition.use(distributed.get_rank())
        super(TextDataLoader, self).__init__(self.dataset, batch_size, num_workers=num_workers, shuffle=True)

    def resample(self):
        if self.multinode:
            self.dataset.data.negative_sampling()
        else:
            self.dataset.negative_sampling()


if __name__ == '__main__':
    text_loader = TextDataLoader(32, False, 1, './data', 'harry_potter.txt', 5, 10, 5, 2e-3, 300)
    text_loader.resample()
    for i, (center_word, context_word, ns_words) in enumerate(text_loader):
        print(center_word)
        print(context_word)
        print(ns_words)
        if i > 10:
            break
