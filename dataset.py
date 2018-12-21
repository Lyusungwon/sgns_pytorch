import os
import numpy as np
import pickle as pkl
import torch
from torch.utils.data import Dataset, ConcatDataset
import re
from nltk.corpus import stopwords
import _pickle as cPickle
from collections import Counter
import time
from collections import defaultdict
import nltk
# nltk.download('wordnet')

def timefn(fn):
    def wrap(*args):
        t1 = time.time()
        result = fn(*args)
        t2 = time.time()
        print("@timefn:{} took {} seconds".format(fn.__name__, t2-t1))
        return result
    return wrap

class TextDataset(Dataset):
    def __init__(self, data_dir, dataset, window_size, ns_size, remove_th, subsam_th, embedding_size, is_character, seed):
        np.random.seed(seed)
        self.dataset_dir = os.path.join(data_dir, dataset)
        data_config_list = [dataset, window_size, ns_size, remove_th, subsam_th, embedding_size, is_character]
        self.data_file = '_'.join(map(str, data_config_list)) + '.pkl'
        self.file_dir = os.path.join(data_dir, self.data_file)
        self.window_size = window_size
        self.ns_size = ns_size
        self.rm_th = remove_th
        self.subsam_th = subsam_th
        self.embedding_size = embedding_size
        self.is_character = is_character
        self.stopwords = set(stopwords.words('english'))
        if not self.is_data_exist():
            self.make_dataset()

        with open(self.file_dir, 'rb') as f:
            if is_character:
                # self.word_pairs, self.vocabs, self.char2idx, self.idx2char = joblib.load(f)
                self.word_pairs, self.vocabs, self.char2idx, self.idx2char, self.probs, self.ground_truth = pkl.load(f)
            # self.word_pairs, self.neg_samples, self.vocabs, self.word2idx, self.idx2word = pkl.load(f)
            else:
                self.word_pairs, self.vocabs, self.word2idx, self.idx2word, self.probs, self.ground_truth = pkl.load(f)

    def is_data_exist(self):
        if os.path.isfile(self.file_dir):
            print("Data {} exist".format(self.data_file))
            return True
        else:
            print("Data {} does not exist".format(self.data_file))
            return False

    @timefn
    def preprocess_counter(self, tokenized):
        # tokenized = reduce(operator.concat, tokenized)
        cnt = Counter(tokenized)
        print(len(cnt), cnt)
        #subsample
        prob = 1 - (self.subsam_th / (np.array(list(cnt.values())) / sum(cnt.values())))**0.5
        rm_idx = (np.random.random((len(prob),)) < prob)
        rm_words = np.array(list(cnt.keys()))[rm_idx]
        for word in rm_words:
            cnt.pop(word, None)
        #remove small words
        small_words = [key for key, value in cnt.items() if value < self.rm_th]
        for word in small_words:
            cnt.pop(word, None)
        # vocabs
        self.vocabs = list(cnt.keys())
        # calculate distribution
        power = 3 / 4
        probs = (np.array(list(cnt.values())) / sum(cnt.values()))**power
        self.probs = probs / probs.sum()
        self.stopwords = self.stopwords|set(small_words)|set(rm_words)
        print(len(cnt), cnt)
        return cnt

    def make_dataset(self):
        text = open(self.dataset_dir, encoding="utf-8").read().lower().strip()
        print("Start to make data")
        # words = word_tokenize(text)
        tokenized_text, word_text = self.tokenize(text)
        print("complete tokenize")
        self.cnt = self.preprocess_counter(word_text)
        print("Start to make data again")
        tokenized_text, word_text = self.tokenize(text)
        print("complete tokenize again")
        if self.is_character:
            self.char2idx, self.idx2char = self.map_char_idx()
        else:
            word2idx, idx2word = self.map_word_idx(self.vocabs)
        word_pairs = []
        pmi = defaultdict(lambda : defaultdict(int))
        for sentence in tokenized_text:
            sentence = np.asarray(sentence)
            for i, word in enumerate(sentence):
                for w in range(-self.window_size, self.window_size + 1):
                    context_word_idx = i + w
                    if context_word_idx < 0 or context_word_idx >= len(sentence) or context_word_idx == i:
                        continue
                    pmi[word][sentence[context_word_idx]] += 1
                    if self.is_character:
                        word_pairs.append(self.make_chars((word, sentence[context_word_idx])))
                    else:
                        word_pairs.append((word2idx[word], word2idx[sentence[context_word_idx]]))
        # neg_samples = self.negative_sampling(len(word_pairs))
        # neg_samples = [[word2idx[word] for word in neg_sample] for neg_sample in neg_samples]
        print(len(word_pairs))
        pmi_matrix = self.pmi(pmi, len(word_text))
        # embedding = self.factorization(pmi_matrix)
        # print(embedding.shape)
        if self.is_character:
            saves = word_pairs, self.vocabs, self.char2idx, self.idx2char, self.probs, pmi_matrix
        else:
            saves = word_pairs, self.vocabs, word2idx, idx2word, self.probs, pmi_matrix
        with open(self.file_dir, 'wb') as f:
            cPickle.dump(saves, f, protocol=2)
            print("Data saved in {}".format(self.data_file))

    def map_char_idx(self):
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        char2idx = {}
        idx2char = {}
        for i in range(len(alphabet)):
            char2idx[list(alphabet)[i]] = i+1
            idx2char[i+1] = list(alphabet)[i]
        return char2idx, idx2char

    def make_chars(self, pairs):
        center, context = pairs
        center_idx = [self.char2idx[char] for char in list(center)]
        context_idx = [self.char2idx[char] for char in list(context)]
        return center_idx, context_idx
    
    def make_char(self, neg_samples):
        negs_idx = []
        for i in range(self.ns_size):
            negs_idx.append([self.char2idx[char] for char in list(neg_samples[i])])
        return negs_idx

    @timefn
    def pmi(self, pmi, D):
        ns = np.log(self.ns_size)
        matrix = []
        for word1 in self.vocabs:
            row = []
            for word2 in self.vocabs:
                wc = pmi[word1][word2]
                if wc != 0:
                    pmiwc = np.log((wc * D) / (self.cnt[word1] * self.cnt[word2]))
                    sppmi = max(0, pmiwc - ns)
                    row.append(sppmi)
                else:
                    row.append(0)
            matrix.append(row)
        return np.array(matrix)

    @timefn
    def factorization(self, matrix):
        u, s, vh = np.linalg.svd(matrix)
        u = u[:, :self.embedding_size]
        s = np.sqrt(s)[:self.embedding_size]
        embedding = u * s
        return embedding

    @timefn
    def tokenize(self, text):
        lemma = nltk.wordnet.WordNetLemmatizer()
        text = re.sub('<.*>'," ", text)
        text = re.sub('[^A-Za-z.]+', " ", text)
        text = text.split(".")
        tokens_list = []
        words_list = []
        for sen in text:
            tokens = []
            for word in sen.split():
                word = lemma.lemmatize(word)
                if word not in self.stopwords and len(word) > 1:
                    tokens.append(word)
                    words_list.append(word)
            tokens_list.append(tokens)
        return tokens_list, words_list

    def map_word_idx(self, vocabs):
        word2idx = {}
        idx2word = {}
        for n, word in enumerate(vocabs):
            word2idx[word] = n
            idx2word[n] = word
        return word2idx, idx2word

    @timefn
    def negative_sampling(self):
        if self.is_character:
            neg_samples = np.random.choice(self.vocabs, size=len(self.word_pairs) * self.ns_size, replace=True, p=self.probs)
            neg_samples = neg_samples.reshape(len(self.word_pairs), self.ns_size)
            self.neg_samples = [self.make_char(neg_sample) for neg_sample in neg_samples]
        else:
            neg_samples = np.random.choice(range(len(self.vocabs)), size=len(self.word_pairs) * self.ns_size, replace=True, p=self.probs)
            self.neg_samples = neg_samples.reshape(len(self.word_pairs), self.ns_size)

    def _unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')

    def __getitem__(self, idx):
        if self.is_character:
            center_idx, context_idx = self.word_pairs[idx]
            negs_idx = self.neg_samples[idx]
            center_idx, context_idx = torch.tensor(center_idx), torch.tensor(context_idx)
            negs_tensor_idx = []
            for i in range(self.ns_size):
                negs_tensor_idx.append(torch.tensor(negs_idx[i]))
            return center_idx, context_idx, negs_tensor_idx
        else:    
            center_idx, context_idx = self.word_pairs[idx]
            negs_idx = self.neg_samples[idx]
            center_idx, context_idx, negs_idx = torch.tensor(center_idx), torch.tensor(context_idx), torch.tensor(negs_idx)
        return center_idx, context_idx, negs_idx

    def __len__(self):
        return len(self.word_pairs)


class EvalDataset(Dataset):
    def __init__(self, data_dir):
        self.file_dir = os.path.join(data_dir, 'harry_potter.txt_5_10_5_0.002_200_True.pkl')
        with open(self.file_dir, 'rb') as f:
            self.word_pairs, self.vocabs, self.char2idx, self.idx2char, self.probs, self.ground_truth = pkl.load(f)

    def make_char_for_eval(self):
        wordslist = list(self.vocabs)
        word2char_idx=[]
        for word in wordslist:
            word2char_idx.append([self.char2idx[char] for char in list(word)])
        return word2char_idx

    def __getitem__(self, idx):
        word2char_idx = self.make_char_for_eval()
        char_idx = word2char_idx[idx]
        char_idx = torch.tensor(char_idx)
        return char_idx

    def __len__(self):
        return len(self.vocabs)

if __name__ == '__main__':
    text_dataset = TextDataset('./data', 'harry_potter.txt', 5, 10, 5, 2e-3, 300)
