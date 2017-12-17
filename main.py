# 训练和预测

import torch

from parse import parse_opt
from model import AddModel, train, predict

def show_samples(result, sample_num):   # 打印一些结果
    vocab = torch.load('vocab.pt')
    vocab = {v:k for k,v in vocab.items()}
    test_data = torch.load('data.pt')['test_data'][:sample_num]
    for ix, elem in enumerate(test_data):
        a, b = elem[0], elem[1]
        a, b = a[::-1], b[::-1]           # 再反转回来, 将个位放在最后
        c = [vocab[id] for id in result[ix]]
        c = c[::-1]                       # 再反转回来, 将个位放在最后
        if c[0] == '0':             # 首位不能出现0
            c[0] = ' '
        print(''.join(a), '+', ''.join(b), '=', ''.join(c))

if __name__=='__main__':
    torch.manual_seed(2017)   # 随机数种子
    opt = parse_opt()

    if not opt.test:             # train
        add_model = AddModel(opt)
        train(add_model, opt)
    else:                        # test
        add_model = AddModel(opt)
        add_model.load_state_dict(torch.load('model-{}.pkl'.format(opt.model)))
        result = predict(add_model, opt)
        show_samples(result, 20)
