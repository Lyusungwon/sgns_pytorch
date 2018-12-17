import os
import torch
import numpy as np
from model import *
from configuration import get_config
import torch.optim as optim
from tensorboardX import SummaryWriter
import time
from dataloader import TextDataLoader
from torch import distributed, nn


def timefn(fn):
    def wrap(*args):
        t1 = time.time()
        result = fn(*args)
        t2 = time.time()
        print("@timefn:{} took {} seconds".format(fn.__name__, t2-t1))
        return result
    return wrap


class Trainer(object):
    def __init__(self, args, model, optimizer, writer, text_loader):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.writer = writer
        self.text_loader = text_loader
        self.device = args.device
        self.multi_node = args.multi_node
        self.monitor_loss = 0
        if self.multi_node:
            self.world_size = distributed.get_world_size()
            self.group = distributed.new_group(ranks=list(range(self.world_size)))
        else:
            self.world_size = 1

    def average_gradients(self):
        for p in self.model.parameters():
            if p.grad is not None:
                tensor = p.grad.data
                distributed.all_reduce(tensor, group=self.group, async_op=self.args.async)
                tensor /= float(self.world_size)
                p.grad.data = tensor.to(self.args.device)
            else:
                continue

    @timefn
    def train_epoch(self):
        self.text_loader.resample()
        for i, (center, context, neg) in enumerate(self.text_loader):
            self.optimizer.zero_grad()
            center = center.to(self.device)
            context = context.to(self.device)
            neg = neg.to(self.device)
            loss = self.model(center, context, neg).sum()
            loss.backward()
            if self.multi_node:
                self.average_gradients()
            self.optimizer.step()
            self.monitor_loss += loss.item()
            if i % self.args.log_interval == 0:
                print('Train dataset: {} [{}/{} ({:.0f}%)] Loss: {:.8f}'.format(
                    self.epoch, i * int(self.args.batch_size/self.world_size), len(self.text_loader.dataset),
                    100. * i / len(self.text_loader),
                    loss/self.args.batch_size*self.world_size))
                step = i // self.args.log_interval + self.epoch * (len(self.text_loader) // self.args.log_interval + 1)
                self.writer.add_scalar('Batch loss', loss / self.args.batch_size*self.world_size, step)
        return self.monitor_loss



def plot_embedding(args, model, text_loader):
    if args.multi_gpu:
        model = model.module
    words = torch.LongTensor([i for i in range(len(text_loader.vocabs))])
    features = model.center_embedding(words.to(args.device))
    return features

@timefn
def evaluate(args, model, text_loader):
    if args.multi_gpu:
        model = model.module
    words = torch.LongTensor([i for i in range(len(text_loader.vocabs))])
    features = model.center_embedding(words.to(args.device)).detach().cpu().numpy()
    e2 = np.matmul(features, features.T)
    piploss = ((text_loader.dataset.ground_truth - e2)**2).mean()
    return piploss

#
# def evaluation(args, writer, model, device, text_loader, k):
#     if args.model_name == "sgns":
#         sim_results = evaluate(model.eval(), device, True, text_loader.word2idx)
#         ana_results = evaluate(model.eval(), device, False, text_loader.word2idx)
#     else:
#         sim_results = evaluate(model.eval(), device, True)
#         ana_results = evaluate(model.eval(), device, False)
#     sim_score, sim_known = result2dict(sim_results)
#     ana_score, ana_known = result2dict(ana_results)
#     writer.add_scalars('Similarity score', sim_score, k)
#     writer.add_scalars('Similarity known', sim_known, k)
#     writer.add_scalars('Analogy score', ana_score, k)
#     writer.add_scalars('Analogy known', ana_known, k)



def init_process(args):
    os.environ['MASTER_ADDR'] = 'deepspark.snu.ac.kr'
    os.environ['MASTER_PORT'] = '19372'
    distributed.init_process_group(
        backend=args.backend,
        init_method=args.init_method,
        rank=args.rank,
        world_size=args.world_size
    )


def train(args):
    if args.multi_node:
        init_process(args)
    device = args.device
    text_loader = TextDataLoader(args.batch_size, args.multi_node, args.num_workers, args.data_dir, args.dataset,
                                 args.window_size, args.neg_sample_size, args.remove_th, args.subsample_th, args.embed_size, args.seed)
    model = SGNS(len(text_loader.dataset.vocabs), args.embed_size)
    if args.load_model is not None:
        model.load_state_dict(torch.load(args.log_dir + args.load_model, map_location=lambda storage,loc: storage))

    if args.multi_gpu:
        print("Let's use", args.num_gpu, "GPUs!")
        model = nn.DataParallel(model, device_ids=[i for i in range(args.num_gpu)])
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(args.log_dir)
    trainer = Trainer(args, model, optimizer, writer, text_loader)
    for epoch in range(args.epochs):
        epoch += 1
        trainer.monitor_loss = 0
        trainer.epoch = epoch
        start_time = time.time()
        loss = trainer.train_epoch()
        if not args.multi_node or (args.multi_node and distributed.get_rank == 0):
            piploss = evaluate(args, model, text_loader)
            print('====> Epoch: {} Average loss: {:.4f} / PIP loss: {:.4f} / Time: {:.4f}'.format(
                epoch, loss / len(text_loader.dataset), piploss, time.time() - start_time))
            writer.add_scalar('Epoch time', time.time() - start_time, epoch)
            writer.add_scalar('PIP loss', piploss, epoch)
            writer.add_scalar('Train loss', loss / len(text_loader.dataset), epoch)
            if epoch % args.save_interval == 0:
                torch.save(model.state_dict(), os.path.join(args.log_dir, 'model.pt'))
                features = plot_embedding(args, model, text_loader)
                writer.add_embedding(features, metadata=text_loader.vocabs, global_step=epoch)


if __name__ =='__main__':
    train(get_config())