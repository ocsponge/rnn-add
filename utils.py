# 生成数据集
# 生成的数[1, 10000)

import torch
import random

def generate_data(data_size, cv_ratio):            # data_size: 样本数   cv_ratio: 数据集分割比例
    a = [random.randint(1, 9999) for i in range(data_size)]
    b = [random.randint(1, 9999) for i in range(data_size)]
    c = [a[i] + b[i] for i in range(data_size)]
    a = [(' ' * (4 - len(str(ix))) + str(ix))[::-1] for ix in a]     # 补全到4位, 并反转将个位数放到第一位
    b = [(' ' * (4 - len(str(ix))) + str(ix))[::-1] for ix in b]
    c = [(' ' * (5 - len(str(ix))) + str(ix))[::-1] for ix in c]

    samples=[[a[i], b[i], c[i]] for i in range(data_size)]
    train_data = samples[:int(len(samples) * cv_ratio)]              # 分割训练集, 测试集
    test_data = samples[int(len(samples) * cv_ratio):]
    valid_data = train_data[int(len(train_data) * cv_ratio):]
    train_data = train_data[:int(len(train_data) * cv_ratio)]        # 训练集里再切分为训练集和验证集

    vocab = '0123456789 '
    vocab = {v: k for k, v in enumerate(vocab)}

    data={'train_data': train_data, 'valid_data': valid_data, 'test_data': test_data}
    torch.save(data, 'data.pt')
    torch.save(vocab, 'vocab.pt')

if __name__ == '__main__':
    random.seed(2017)              # 随机数种子
    generate_data(100000, 0.8)     # 生成10万组数据
