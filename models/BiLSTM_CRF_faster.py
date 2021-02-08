# coding: UTF-8
import torch
import torch.nn as nn
from torchcrf import CRF

START_TAG = 'START'
STOP_TAG = 'STOP'

class config(object):
    def __init__(self):
        # 路径类 带*的是运行前的必要文件  未带*文件/文件夹若不存在则训练过程会生成
        self.train_path = 'data/train'  # *
        self.dev_path = 'data/test'  # *
        self.class_ls_path = 'data/class.txt'  # *
        self.pretrain_dir = '/data/sgns.sogou.char'  # 前期下载的预训练词向量*
        self.test_path = 'data/test.txt'  # 若该文件不存在会加载dev.txt进行最终测试
        self.vocab_path = 'data/vocab.pkl'
        self.model_save_dir = 'checkpoint'
        self.model_save_name = self.model_save_dir + '/BiLSTM_CRF_faster.ckpt'  # 保存最佳dev acc模型

        # 可调整的参数
        # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz,  若不存在则后期生成
        # 随机初始化:random
        self.embedding_type = 'embedding_SougouNews.npz'
        self.use_gpu = True  # 是否使用gpu(有则加载 否则自动使用cpu)
        self.batch_size = 128
        self.num_epochs = 40  # 训练轮数
        self.num_workers = 0  # 启用多线程
        self.learning_rate = 0.001  # 训练发现0.001比0.01收敛快(Adam)
        self.embedding_dim = 300  # 词嵌入维度
        self.hidden_size = 300  # 隐藏层维度
        self.num_layers = 2  # RNN层数
        self.bidirectional = True  # 双向 or 单向
        self.require_improvement = 1  # 1个epoch若在dev上acc未提升则自动结束

        # 由前方参数决定  不用修改
        self.class_ls = []
        self.num_class = len(self.class_ls)
        self.vocab_len = 0  # 词表大小(训练集总的字数(字符级)） 在embedding层作为参数 后期赋值
        self.embedding_pretrained = None  # 根据config.embedding_type后期赋值  random:None  else:tensor from embedding_type
        if self.use_gpu and torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained,
                                                          freeze=False)  # 表示训练过程词嵌入向量会更新
        else:
            self.embedding = nn.Embedding(config.vocab_len, config.embedding_dim)  # PAD索引填充
        if config.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.config = config
        self.rnn = nn.LSTM(config.embedding_dim, config.hidden_size, config.num_layers, batch_first=True,
                           bidirectional=config.bidirectional)

        self.tag_ls = self.getTagLs(config)
        self.tag2idx = self.getTagDic()
        # 转换参数矩阵 输入i,j是得分从j转换到i
        self.tagset_size = len(self.tag2idx)
        # 将lstm的输出映射到标记空间
        self.hidden2tag = nn.Linear(config.hidden_size*self.num_directions, self.tagset_size)  # -> (B, num_class+2)  加上了START END
        self.crf = CRF(self.tagset_size)

    def _forward_alg(self, feats):
        # 使用前向算法计算分区函数
        init_alphas = self._make_tensor(torch.full((1, self.tagset_size), -10000.))
        # START_TAG 包含所有得分
        init_alphas[0][self.tag2idx[START_TAG]] = 0.

        # 包装一个变量 以便获得自动反向提升
        forward_var = init_alphas

        # 通过句子迭代
        for feat in feats:
            alphas_t = []  # the forward tensor at this timestep
            for next_tag in range(self.tagset_size):
                # 广播发射得分：无论之前的标记是怎样的都是相同的
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # trans_score 的第i个条目是从i转移到next_tag的分数
                trans_score = self.transitions[next_tag].view(1, -1)
                # next_tag_var 的第i个条目是执行log-sum-exp之前的变（i -> next_tag）的值
                next_tag_var = forward_var + trans_score + emit_score
                # 此标记的转发变量是所有分数的log-sum-exp
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag2idx[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):
        # Give the score of a provided tag sequence
        score = self._make_tensor(torch.zeros(1))
        tags = self._make_tensor(torch.cat([self._make_tensor(torch.tensor([self.tag2idx[START_TAG]], dtype=torch.long)),tags]))
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i+1], tags[i]]+feat[tags[i+1]]
        score = score + self.transitions[self.tag2idx[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = self._make_tensor(torch.full((1, self.tagset_size), -10000.))
        init_vvars[0][self.tag2idx[START_TAG]] = 0

        # forward_var at step o holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # hold the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i]保存上一步的标签i的viterbi变量
                # 加上标签i转换到next_tag的分数 我们这里不包括emission分数 因为最大值不依赖于它们（在下面添加它们）
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # 现在添加emission分数 并将forward_var分配给刚计算的viterbi变量集
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # 过渡到STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag2idx[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # 按照后退指针解码最佳路径
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # 弹出开始标记（我们不想将器返回给调用者）
        start = best_path.pop()
        assert start == self.tag2idx[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def _get_lstm_features(self, x):
        # 数据预处理时，x被处理成是一个tuple,其内容是: (word, label).
        # x:b_size
        x = self.embedding(x)  # B -> (B, e_d)
        x = x.unsqueeze(1)  # (B, e_b) -> (B, 1, e_b)
        h_0, c_0 = self._init_hidden(batchs=x.size(0))
        out, (hidden, c) = self.rnn(x,(h_0, c_0))  # out:(B, 1, num_directions*hidden_size) hidden:(num_layer*nun_directions, B,  hidden_size)
        # out = out.squeeze(1)
        # output is batch_first but hidden not
        out = self.hidden2tag(out)  # (B,num_directions*hidden_size) -> (B, num_class)
        out = out.transpose(0, 1)
        return out

    def neg_log_likelihood(self, x, tags):  # 损失函数
        tags = tags.unsqueeze(0)
        feats = self._get_lstm_features(x)
        return -self.crf(feats, tags)

    def forward(self, x):
        # 数据预处理时，x被处理成是一个tuple,其内容是: (word, label).
        # x:b_size
        lstm_feats = self._get_lstm_features(x)  # 获取BiLSTM的emission分数

        out = self.crf.decode(lstm_feats)
        return out

    def _init_hidden(self, batchs):  # 初始化h_0和c_0 与GRU不同的是多了c_0（细胞状态）
        h_0 = torch.zeros(self.config.num_layers*self.num_directions, batchs,  self.config.hidden_size)
        c_0 = torch.zeros(self.config.num_layers*self.num_directions, batchs, self.config.hidden_size)
        return self._make_tensor(h_0), self._make_tensor(c_0)

    def _make_tensor(self, tensor):
        """
        函数说明： 将传入的tensor转移到cpu或gpu内

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
            2021-2-2 21:53:20
        """
        tensor_ret = tensor.to(self.config.device)
        return tensor_ret

    def getTagLs(self, config):
        tag_ls = config.class_ls
        tag_ls.append(START_TAG)
        tag_ls.append(STOP_TAG)
        return tag_ls

    def getTagDic(self):
        tag_dic = {}
        for idx, label in enumerate(self.tag_ls):
            tag_dic[label] = idx
        return tag_dic

    def idx2Tag(self, idx):
        return self.tag_ls[idx]

def argmax(vec):
    # 将argmax作为python int返回
    _, idx = torch.max(vec, 1)
    return idx.item()

# 以正向算法的数值稳定方式计算log sum exp
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))