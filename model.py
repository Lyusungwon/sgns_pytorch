import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
from dataloader import *
import time 

class SGNS(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SGNS, self).__init__()
        self.center_embedding = nn.Embedding(vocab_size, embed_size)
        self.context_embedding = nn.Embedding(vocab_size, embed_size)

    def pos_loss(self, center, context):
        score_target = torch.bmm(center.unsqueeze(1), context.unsqueeze(2))
        loss = F.logsigmoid(score_target).sum()
        return loss

    def neg_loss(self, center, ns):
        score_neg = torch.bmm(center.unsqueeze(1), ns.transpose(1, 2))
        loss = F.logsigmoid(-score_neg).sum()
        return loss

    def forward(self, center, context, ns):
        center = self.center_embedding(center)
        context = self.context_embedding(context)
        ns = self.context_embedding(ns) #[32,5,128]
        return - (self.pos_loss(center, context) + self.neg_loss(center, ns))

    def get_embedding(self, center):
        return self.center_embedding(center)

class generator(nn.Module):
    def __init__(self, char_num, gen_embed_dim, hidden_size, num_layer, dropout, bidirectional, device):
        super(generator, self).__init__()
        self.bidirectional = bidirectional
        self.device = device
        self.embedding = nn.Embedding(char_num, gen_embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(gen_embed_dim, hidden_size, num_layers=num_layer,
                    dropout=dropout, batch_first = True, bidirectional=bidirectional)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            else: 
                nn.init.xavier_normal_(param)

    def sorting(self, x, x_len):
        x_ordered = np.sort(x_len)[::-1]
        sort_idx = np.argsort(x_len)[::-1]
        unsort_idx = np.argsort(sort_idx)[::-1]
        x_ordered = torch.from_numpy(x_ordered.copy()).to(self.device)
        sort_idx= torch.from_numpy(sort_idx.copy()).to(self.device)
        # sort_idx= torch.from_numpy(sort_idx.copy())
        unsort_idx = torch.from_numpy(unsort_idx.copy()).to(self.device)
        x = x.index_select(0, sort_idx)
        return x, unsort_idx, x_ordered

    def forward(self, x, x_len):
        x, unsort_idx, x_ordered = self.sorting(x, x_len)
        embedded = self.embedding(x)
        embedded = pack_padded_sequence(embedded, x_ordered, batch_first = True)
        output, (h,_) = self.lstm(embedded)
        if self.bidirectional:
            ordered_hidden_1 = h[-1].index_select(0, unsort_idx)
            ordered_hidden_2 = h[-2].index_select(0, unsort_idx)
            ordered_hidden = torch.cat((ordered_hidden_1,ordered_hidden_2), dim=1)
            output_padded, _ = pad_packed_sequence(output, batch_first=True)
            ordered_output = output_padded.index_select(0, unsort_idx)
        else:
            ordered_hidden = h[-1].index_select(0, unsort_idx)
            output_padded, _ = pad_packed_sequence(output, batch_first=True)
            ordered_output = output_padded.index_select(0, unsort_idx)
        return ordered_hidden, ordered_output

class word_embed_ng(nn.Module):
    def __init__(self, char_num, gen_embed_dim, hidden_size, num_layer, 
                dropout, fc_hidden, embed_size, k, bidirectional, device):
        super(word_embed_ng, self).__init__()
        self.center_generator = generator(char_num, gen_embed_dim, hidden_size, num_layer, dropout, bidirectional, device)
        self.context_generator = generator(char_num, gen_embed_dim, hidden_size, num_layer, dropout, bidirectional, device)
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        if self.bidirectional:
            self.hidden_size = hidden_size*2 
        self.k = k
        self.cen_add_fc= nn.Sequential(
            nn.Linear(self.hidden_size, embed_size)
        )
        self.con_add_fc= nn.Sequential(
            nn.Linear(self.hidden_size, embed_size)
        )

    def cal_loss(self, x, y, neg):
        score_target = torch.bmm(x.unsqueeze(1),y.unsqueeze(2))
        score_neg = torch.bmm(x.unsqueeze(1), neg.transpose(0,1).transpose(1,2))
        loss = -F.logsigmoid(score_target).sum() + -F.logsigmoid(-score_neg).sum()
        return loss

    def forward(self, x, x_len, y, y_len, neg):
        embedded_cen, _ = self.center_generator(x, x_len)
        embedded_con, _ = self.context_generator(y, y_len)
        prediction = self.cen_add_fc(embedded_cen)
        target = self.con_add_fc(embedded_con)
        neg_outputs =[]
        for i in range(self.k):
            embedded_neg, _= self.context_generator(neg[i][0], neg[i][1])
            neg_outputs.append(self.con_add_fc(embedded_neg))
        neg_output_tensor = torch.stack(neg_outputs)
        loss = self.cal_loss(prediction, target, neg_output_tensor)
        return loss
    
    def get_center_embedding(self, center, center_len):
        embedded_cen, _ = self.center_generator(center, center_len)
        embedding = self.cen_add_fc(embedded_cen)
        return embedding

if __name__=='__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    text_loader = TextDataLoader('./data', 'toy/merge.txt', 8, 5, 5, True, 0, 5, 1e-04)

    model = SGNS(26, 10)
    model = model.to(device)
    
    # text_loader = TextDataLoader('./data', batch_size = 2, window_size = 5, k=5)

    for i, (center, context, neg) in enumerate(text_loader):
        center, center_len = center
        context, context_len = context
        center = center.to(device)
        context = context.to(device)
        n =[]
        for k in range(5):
            padded_neg, neg_len = neg[k]
            n.append((padded_neg.to(device), neg_len))
        if torch.cuda.device_count() > 1:
            print("using", torch.cuda.device_count(), "GPUs")
            model = nn.DataParallel(model, device_ids=[2,3])
        model = model.to(device)
        output = model(center, center_len, context, context_len, neg)
        print(output)
        break
        
