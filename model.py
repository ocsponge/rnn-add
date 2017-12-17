# 定义模型

import numpy
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils import data
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau

logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

# 定义数据iter
class FeedDataset(data.Dataset):
    def __init__(self, mode):
        self.vocab = torch.load('vocab.pt')
        if mode == 'train':
            self.data = torch.load('data.pt')['train_data']
        if mode == 'valid':
            self.data = torch.load('data.pt')['valid_data']
        if mode == 'test':
            self.data = torch.load('data.pt')['test_data']
        for elem in self.data:
            elem[0] = [self.vocab[ix] for ix in elem[0]]
            elem[1] = [self.vocab[ix] for ix in elem[1]]
            elem[2] = [self.vocab[ix] for ix in elem[2]]

    def __getitem__(self, index):
        a = numpy.array(self.data[index][0])
        b = numpy.array(self.data[index][1])
        c = numpy.array(self.data[index][2])
        encode_length = 4
        decode_length = 5
        return a, b, c, encode_length, decode_length

    def __len__(self):
        return len(self.data)

# 定义加法器模型
class AddModel(nn.Module):
    def __init__(self, opt):
        super(AddModel, self).__init__()
        self.num_layers   = opt.num_layers
        self.hidden_size  = opt.hidden_size

        self.embed_layer  = nn.Embedding(opt.vocab_size, opt.embed_size)           # embedding层
        self.encode_layer = nn.LSTM(2*opt.embed_size, opt.hidden_size, opt.num_layers, batch_first=True)    # encoder
        self.decode_layer = nn.LSTM(opt.hidden_size, opt.hidden_size, opt.num_layers, batch_first=True)     # decoder
        self.logit_layer  = nn.Linear(opt.hidden_size, opt.vocab_size)
        self._init_weights_()

    def _init_weights_(self):
        initrange = 0.1
        self.logit_layer.bias.data.fill_(0)
        self.embed_layer.weight.data.uniform_(-initrange, initrange)
        self.logit_layer.weight.data.uniform_(-initrange, initrange)

    def _init_hidden_(self, batch_size):           # 初始状态
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, batch_size, self.hidden_size).zero_()).cuda(),
                Variable(weight.new(self.num_layers, batch_size, self.hidden_size).zero_()).cuda())

    def forward(self, input_a, input_b, encode_length, decode_length):
        batch_size  = input_a.size(0)
        init_states = self._init_hidden_(batch_size)
        input_a     = self.embed_layer(input_a)                                            # batch,length,embed
        input_b     = self.embed_layer(input_b)                                            # batch,length,embed
        input       = torch.cat((input_a, input_b), 2)
        packed      = pack_padded_sequence(input, encode_length.tolist(), batch_first=True)
        _, (hn, cn) = self.encode_layer(packed, init_states)                  # num_layers, batch, hidden_size
        hn          = hn[-1].unsqueeze(1).expand(batch_size, 5, self.hidden_size)     # hn[-1]表示最后一层的隐状态 batch, 5, hidden_size
        packed      = pack_padded_sequence(hn, decode_length.tolist(), batch_first=True)
        hiddens, _  = self.decode_layer(packed, init_states)
        output      = self.logit_layer(hiddens.data)
        return output

    def predict(self, input_a, input_b, encode_length, decode_length):
        batch_size  = input_a.size(0)
        states      = self._init_hidden_(batch_size)
        input_a     = self.embed_layer(input_a)                                            # batch,length,embed
        input_b     = self.embed_layer(input_b)                                            # batch,length,embed
        input       = torch.cat((input_a, input_b), 2)
        packed      = pack_padded_sequence(input, encode_length.tolist(), batch_first=True)
        _, (hn, cn) = self.encode_layer(packed)                  #num_layers, batch, hidden_size
        hn          = hn[-1].unsqueeze(1)           # batch,1,hidden_size

        result = []
        for i in range(5):
            hiddens, states = self.decode_layer(hn, states)          # (batch_size, 1, hidden_size)
            output          = self.logit_layer(hiddens.squeeze(1))            # (batch_size, vocab_size)
            pred            = output.data.max(1)[1].view(-1, 1)           # (batch_size, 1)
            result.append(pred)
        result = torch.cat(result, 1)                  # (batch_size, length)
        return result

############################################# 训练 验证 预测 ######################################################

def train(model, opt):
    logging.info('Start training...')
    model.train()
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    params    = model.parameters()
    optimizer = optim.Adam(params, lr=opt.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode=opt.scheduler_mode, factor=opt.lr_factor, verbose=True,
                                             patience=opt.patience, threshold=1e-3, threshold_mode='abs')
    train_dataset    = FeedDataset('train')
    train_dataloader = data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=6)
    train_batch_n    = len(train_dataloader)

    valid_dataset    = FeedDataset('valid')
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=6)

    for iepoch in range(opt.max_epochs):
        for ix, (input_a, input_b, label, encode_length, decode_length) in enumerate(train_dataloader):
            torch_input_a = Variable(input_a).cuda()
            torch_input_b = Variable(input_b).cuda()
            torch_label   = Variable(label).cuda()
            torch_label   = pack_padded_sequence(torch_label, decode_length.tolist(), batch_first=True).data
            torch_output  = model(torch_input_a, torch_input_b, encode_length, decode_length)
            batch_loss    = criterion(torch_output, torch_label)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            if (ix+1) % opt.log_steps == 0 and (ix+1) != train_batch_n:
                logging.info('Epoch {:>2}/{}   Step {:>3}/{}   Loss: {:.6f}'
                       .format(iepoch, (opt.max_epochs-1), (ix+1), train_batch_n, batch_loss.data[0]))

        valid_acc = 0
        for valid_input_a, valid_input_b, valid_label, valid_encode_length, valid_decode_length in valid_dataloader:
            valid_input_a = Variable(valid_input_a).cuda()
            valid_input_b = Variable(valid_input_b).cuda()
            valid_output  = model.predict(valid_input_a, valid_input_b, valid_encode_length, valid_decode_length)
            num_correct   = (valid_output == valid_label.cuda()).sum()
            valid_acc     += num_correct

        valid_acc = valid_acc/len(valid_dataset)/5
        logging.info('Epoch {:>2}/{}   Step {:>3}/{}   Loss: {:.6f}   Acc: {:.6f}'
               .format(iepoch, (opt.max_epochs-1), (ix+1), train_batch_n, batch_loss.data[0], valid_acc))

        scheduler.step(valid_acc)
        torch.save(model.state_dict(), 'model-{}.pkl'.format(iepoch))
        if optimizer.param_groups[0]['lr'] <= (1e-8+1e-15):    # 早停, 1e-15是可能存在的计算机误差
            logging.info('Early stopping...Finish!')
            break

def predict(model, opt):
    model.eval()
    model.cuda()
    test_dataset = FeedDataset('test')
    test_dataloader = data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=6)

    result = []
    for input_a, input_b, label, encode_length, decode_length in tqdm(test_dataloader, ncols=80):
        torch_input_a = Variable(input_a).cuda()
        torch_input_b = Variable(input_b).cuda()
        torch_output  = model.predict(torch_input_a, torch_input_b, encode_length, decode_length) # (batch_size, length)
        result.append(torch_output)
    return torch.cat(result).tolist()
