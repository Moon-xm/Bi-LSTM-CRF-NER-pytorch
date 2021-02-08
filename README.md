# Bi-LSTM-CRF-NER-pytorch
使用Bi-LSTM-CRF进行命名实体识别

在models文件夹内包含了两个模型，要切换原始模型需修改run.py内调用的库models.BiLSTM_CRF_faster 及train_eval_faster为models.BiLSTM及train_eval。

两个models的对比：

BiLSTM_CRF的CRF层是按照[pytorch官网教程](https://pytorch123.com/FifthSection/Dynamic_Desicion_Bi-LSTM/)中修改而来，只能一条一条的样本进行解码，太慢了（且实测发现在colab的gpu上运行比电脑笔记本cpu运行速度慢1倍，原因未知）

BiLSTM_CRF_faster的CRF层是基于AllenNLP实现的[CRF包](https://github.com/yumoh/torchcrf.git)，速度快（实测发现在colab的gpu上运行比电脑笔记本cpu运行慢，cpu 30s 100个Iter, colab gpu 40s 100 Iter）

准确率都是在90%左右

