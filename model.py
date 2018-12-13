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
        
