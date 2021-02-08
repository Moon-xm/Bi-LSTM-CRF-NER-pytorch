# coding: UTF-8
from dataprocess import build_dataset
from train_eval_faster import train_test
from models.BiLSTM_CRF_faster import config, Model  # 所有参数存在这

if __name__ == '__main__':
    print('Loading data...')
    config = config()  # 实例化各参数
    train_loader, dev_loader, test_loader = build_dataset(config)  # 构造可用于模型输入的训练、验证、测试数据
    print('Loading model...')
    model = Model(config).to(config.device)
    print(model)
    print('Training and evaluating on {}...'.format(config.device))
    train_test(config, model, train_loader, dev_loader, test_loader)
