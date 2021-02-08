# coding: UTF-8
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from tqdm import tqdm
import torch
import numpy as np

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字 padding符号
MIN_FREQ = 1  # 出现频率大于该参数的字才放入词典


def build_vocab(train_path, class_ls_path):  # 构建词典（默认是字符级）
    """
    函数说明： 根据传入的数据构建词表字典 词表最大长度（词表最多多少字）为MAX_VOCAB_SIZE
    最小词频为MIN_FREQ

    Parameter：
    ----------
        file_path - 需要构建词典的数据集（一般用训练集构建）
    Return:
    -------
        vocab - 构建好的词表字典  eg: vocab[‘我’]=56
    Author:
    -------
        Ming Chen
    Modify:
    -------
        2021-1-31 20:12:48
    """

    vocab_dic = {}  # 字典
    class_set = set()  # 集合
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            lin = line.strip()  # 去除头尾的空格 换行符 制表符
            if not lin:
                continue
            content, label = lin.split()
            vocab_dic[content] = vocab_dic.get(content, 0) + 1  # 每个字计数
            class_set.add(label)
        vocab_ls = sorted([_ for _ in vocab_dic.items() if _[1] >= MIN_FREQ],
                          key=lambda x: x[1], reverse=True)[:MAX_VOCAB_SIZE]
        class_ls = list(sorted(class_set))
        with open(class_ls_path, 'w', encoding='utf-8') as cf:
            cf.write('\n'.join(str(label) for label in class_ls))
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_ls)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})  # 将UNK和PAD索引加到词典最后
    return vocab_dic

def extract_vocab_tensor(config):  # 提取vocab内的预训练词向量
    """
    函数说明： 根据vocab内的字及索引构建词嵌入numpy(vocab是根据训练数据构建的词与索引对应的字典)

    Parameter：
    ----------
        config.embedding_type - 'random' or others
        config.vocab_path - vocab存储路径
        config.pretrain_dir - 预训练词嵌入向量
        config.embedding_dim - 词嵌入维度
    Return:
    -------
        embedding_pretrained - 由vocab内的词组成的词嵌入数组（numpy）
    Author:
    -------
        Ming Chen
    Modify:
    -------
        2021-2-1 11:23:12
    """
    if config.embedding_type == 'random':  # 随机初始化
        embedding_pretrained = None
    else:  # 加载预训练词向量
        vocab_tensor_path = 'data/' + config.embedding_type
        if os.path.exists(vocab_tensor_path):  # 已构建则直接加载
            embedding_pretrained = np.load(vocab_tensor_path)['embeddings'].astype('float32')
        else:  # 重新构建
            with open(config.vocab_path, 'rb') as vocab_f:
                word_to_id = pickle.load(vocab_f)
                pretrain_f = open(config.pretrain_dir, 'r', encoding='utf-8')
                embeddings = np.random.rand(len(word_to_id), config.embedding_dim)
                for i, line in enumerate(pretrain_f.readlines()):
                    if i == 0:  # 若第一行是标题，则跳过 部分预训练模型第一行是词数和词嵌入维度
                        continue
                    lin = line.strip().split(' ')
                    if lin[0] in word_to_id:
                        idx = word_to_id[lin[0]]
                        emb = [float(x) for x in lin[1:config.embedding_dim+1]]  # 取出对应词向量
                        embeddings[idx] = np.asarray(emb, dtype='float')
                pretrain_f.close()
                np.savez_compressed(vocab_tensor_path, embeddings=embeddings)  # embeddings 是数组名
                embedding_pretrained = embeddings.astype('float32')
    return embedding_pretrained

def build_dataset(config):
    """
    函数说明： 根据训练、验证、测试数据返回其由索引组成的张量

    Parameter：
    ----------
        config - ...
    Return:
    -------
        train_loader - 用于模型输入的训练集DataLoader
        dev_loader - 用于模型输入的验证集DataLoader
        test_loader - 用于模型输入的测试集DataLoader
    Author:
    -------
        Ming Chen
    Modify:
    -------
        2021-1-31 17:38:02
    """
    if os.path.exists(config.vocab_path):  # 加载词典
        vocab = pickle.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, config.class_ls_path)  # 用训练数据构建词典
        with open(config.vocab_path,'wb') as f:
            pickle.dump(vocab, f)  # 存储每个字及对应索引的字典 eg：我：56  vocab['我’]=56
    config.vocab_len = len(vocab)
    config.class_ls = [x.strip() for x in open(config.class_ls_path, 'r', encoding='utf-8').readlines()]
    print('\nVocab size: {}'.format(len(vocab)))

    train_data = Mydataset(config.train_path, config, vocab)
    dev_data = Mydataset(config.dev_path, config, vocab)
    train_loader = DataLoader(dataset=train_data,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=config.num_workers)
    dev_loader = DataLoader(dataset=dev_data,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=config.num_workers)
    if os.path.exists(config.test_path):
        test_data = Mydataset(config.test_path, config, vocab)
        test_loader = DataLoader(dataset=test_data,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers)
    else:  # 若无测试数据则加载验证集进行最终测试
        test_loader = dev_loader
    config.embedding_pretrained = torch.tensor(extract_vocab_tensor(config))  # 构建vocab内的词嵌入矩阵

    return train_loader, dev_loader, test_loader

class Mydataset(Dataset):
    def __init__(self, filepath, config, vocab):
        self.filepath = filepath
        self.vocab = vocab
        self.label_dic = self._getLabelDic(config)
        self.data_label = self._get_contents(config)
        self.x = make_tensor(torch.tensor([_[0] for _ in self.data_label]), config)
        # self.seq_len = make_tensor(torch.tensor([_[2] for _ in self.data_label_len]), config)
        self.y = make_tensor(torch.tensor([_[1] for _ in self.data_label]), config)
        self.len = len(self.x)

    def __getitem__(self, index):  # (x,seq_len)构成一个元组，并返回标签
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

    def _getLabelDic(self, config):
        label_dic ={}
        with open(config.class_ls_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                label = line.strip()
                label_dic[label] = idx
        return label_dic

    def _get_contents(self, config):
        """
        函数说明： 将filepath的句子中每个字转换为相应索引，并返回每个句子的标签和长度seq_len

        Parameter：
        ----------
            self.file_path - 需转换的文件
            self.vocab - 构建好的词典
            config.pad_size - 要进行embedding的句子长度,embedding层的input的句子长度  短补(<PAD>)长截
        Return:
        -------
            contents - 转化好的索引列表，标签,返回的是列表类型
                        contents[0] - 句子每个字对应的索引组成的列表
                        contents[1] - 句子标签
        Author:
        -------
            Ming Chen
        Modify:
        -------
            2021-1-31 20:12:48
        """
        contents = []
        with open(self.filepath,'r',encoding='utf-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                word, label = lin.split()
                word_id = self.vocab.get(word, self.vocab.get(UNK))  # dict.get(key, default=None)返回指定键的值，如果键不在字典中返回默认值 None 或者设置的默认值。这里默认值就是UNK对应的数值
                label_id = self.label_dic.get(label)
                contents.append((word_id, label_id))
            return contents  # [([...], 0), ([...], 1), ...]


def make_tensor(tensor, config):
        """
        函数说明： 将传入的tensor转换为LongTensor并转移到cpu或gpu内

        Parameter：
        ----------
            tensor - 需转换的张量
            config.device - cpu or cuda:0
        Return:
        -------
            tensor_ret - 转换好的LongTensor类型张量
        Author:
        -------
            Ming Chen
        Modify:
        -------
            2021-2-1 09:09:39
        """
        tensor_ret = torch.LongTensor(tensor).to(config.device)
        return tensor_ret






