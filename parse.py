# 定义常见参数

import argparse

def parse_opt():
    parser=argparse.ArgumentParser()

    # 模型参数
    model_group=parser.add_argument_group('model arguments')
    model_group.add_argument('-vocab-size', metavar='', type=int, default=11, help='size of vocabulary')
    model_group.add_argument('-hidden-size', metavar='', type=int, default=256, help='size of hidden nodes in each layer')
    model_group.add_argument('-num-layers', metavar='', type=int, default=1, help='number of layers in the rnn')
    model_group.add_argument('-embed-size', metavar='', type=int, default=256, help='embdding size of each token in the vocabulary')

    # 训练参数
    train_group=parser.add_argument_group('train arguments')
    train_group.add_argument('-max-epochs', metavar='', type=int, default=100, help='number of epochs')
    train_group.add_argument('-batch-size', metavar='', type=int, default=128, help='batch size')
    train_group.add_argument('-learning-rate', metavar='', type=float, default=1e-3, help='learning rate')
    train_group.add_argument('-scheduler-mode', metavar='', type=str, default='max', help='one of min, max')
    train_group.add_argument('-lr-factor', metavar='', type=float, default=0.1, help='factor by which the learning rate will be reduced')
    train_group.add_argument('-patience', metavar='', type=int, default=3, help='number of epochs with no improvement after which lr will be reduced')
    train_group.add_argument('-log-steps', metavar='', type=int, default=1000, help='number of steps after which training info will be printed')

    # 预测参数
    pred_group=parser.add_argument_group('pred arguments')
    pred_group.add_argument('-test', action='store_true', help='change to predict mode')
    pred_group.add_argument('-model', metavar='', type=int, default=100, help='the order number of saved models to predict')

    args = parser.parse_args()

    # 检查参数是否合法
    assert args.vocab_size > 0, "vocab size should be greater than 0"
    assert args.hidden_size > 0, "rnn size should be greater than 0"
    assert args.num_layers > 0, "num layers should be greater than 0"
    assert args.embed_size > 0, "embedding size should be greater than 0"
    assert args.max_epochs > 0, "max epochs should be greater than 0"
    assert args.batch_size > 0, "batch size should be greater than 0"
    assert args.learning_rate > 0, "learning rate should be greater than 0"

    return args

if __name__=='__main__':
    args=parse_opt()
    print(args)
