# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import DBSCAN,KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import Dense, Embedding, Input, SpatialDropout1D
from keras.layers import Conv1D,Conv2D, Flatten, Dropout, MaxPool1D,MaxPool2D, GlobalAveragePooling1D, concatenate, GlobalMaxPooling1D
from keras.callbacks import TensorBoard, EarlyStopping,ModelCheckpoint
from keras.utils import to_categorical
import time
import numpy as np
from keras import backend as K#keras后端
from sklearn.model_selection import StratifiedKFold
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
#配置运行参数
config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = K.tf.Session(config=config)

Fname = 'malware_CNN_'
Time = Fname + str(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
#tensorboard日志
tensorboard = TensorBoard(log_dir='./Logs/' + Time, histogram_freq=0, write_graph=False, write_images=False,
                          embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

# with open("dynamic_feature_train.csv.pkl", "rb") as f:
#     labels_d = pickle.load(f)
# with open("dynamic_feature_train.csv.pkl", "rb") as f:
#     labels = pickle.load(f)
#     files = pickle.load(f)
#
# maxlen = 2000
# labels = np.asarray(labels)
# labels = to_categorical(labels, num_classes=2)#label词向量
# tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
# x_train_word_ids = tokenizer.texts_to_sequences(files)#用于向量化文本,将文本转换为序列
# x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=maxlen)#将序列填充到maxlen长度
# vocab = tokenizer.word_index


MAX_NB_WORDS = 20000
class DbscanClustering():
    def __init__(self, stopwords_path=None):
        self.vectorizer = CountVectorizer()
        self.transformer = TfidfTransformer()

    def text_cnn_model(self):
        main_input = Input(shape=(maxlen,), dtype='float64')
        _embed = Embedding(min(len(vocab),MAX_NB_WORDS) + 1, 256, input_length=maxlen)(main_input)
        # _embed = SpatialDropout1D(0.25)(_embed)
        # _embed=GaussianNoise(0.125)(_embed)
        # _embed=BatchNormalization()(_embed)
        warppers = []
        num_filters = 128
        kernel_size = [2,3,4,5]
        conv_action = 'relu'

        for _kernel_size in kernel_size:
            # for dilated_rate in [1, 2]:#扩张率(dilation rate):该参数定义了卷积核处理数据时各值的间距
            conv1d = Conv1D(filters=num_filters, kernel_size=_kernel_size, activation=conv_action)(_embed)
            conv1d=GlobalMaxPooling1D()(conv1d)
            conv1d=BatchNormalization()(conv1d)
            warppers.append(conv1d)

        fc = concatenate(warppers)
        # fc = Dropout(0.5)(fc)
        fc = Dense(256, activation='relu')(fc)
        fc = Dropout(0.25)(fc)
        preds = Dense(2, activation='softmax')(fc)
        model = Model(inputs=main_input, outputs=preds)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model
    def preprocess_data(self):

        """
        文本预处理，每行一个文本
        :param corpus_path:
        :return:
        """
        # with open("dynamic_feature.csv.pkl", 'rb') as f:
        #         names=pickle.load(f)
        #         corpus=pickle.load(f)
        # print(len(names))
        corpus_path="..//..//API_name_feature_small.csv"
        corpus=[]
        names=[]
        data = pd.read_csv(corpus_path)
        goup_fileid = data.groupby('id')
        count=0
        for file_name, file_group in goup_fileid:
            api_sequence = ' '.join(file_group['api'])
            corpus.append(api_sequence)
            names.append(file_name)
            count+=1
            # if count>=1000:
            #     break
        with open("dynamic_feature.csv.pkl", 'wb') as f:
                pickle.dump(names, f)
                pickle.dump(corpus, f)
        return corpus

    def get_text_tfidf_matrix(self, corpus):
        tfidf = self.transformer.fit_transform(self.vectorizer.fit_transform(corpus))
        # 获取tfidf矩阵中权重
        weights = tfidf.toarray()
        return weights

    def pca(self, weights, n_components=2):
        """
        PCA对数据进行降维
        :param weights:
        :param n_components:
        :return:
        """
        pca = PCA(n_components=n_components)
        return pca.fit_transform(weights)

    def dbscan(self, eps, min_samples, fig=True):
        """
        DBSCAN：基于密度的文本聚类算法
        :param corpus_path: 语料路径，每行一个文本
        :param eps: DBSCA中半径参数
        :param min_samples: DBSCAN中半径eps内最小样本数目
        :param fig: 是否对降维后的样本进行画图显示
        :return:
        """
        corpus = self.preprocess_data()
        weights = self.get_text_tfidf_matrix(corpus)
        pca_weights = self.pca(weights)
        # clf = DBSCAN(eps=eps, min_samples=min_samples)
        clf = KMeans(n_clusters=400)
        y = clf.fit_predict(pca_weights)
        if fig:
            plt.scatter(pca_weights[:, 0], pca_weights[:, 1], c=y)
            plt.show()

        # 中心点
        # centers = clf.cluster_centers_

        # 用来评估簇的个数是否合适,距离约小说明簇分得越好,选取临界点的簇的个数
        # score = clf.inertia_

        # 每个样本所属的簇
        result = {}
        for text_idx, label_idx in enumerate(y):
            if label_idx not in result:
                result[label_idx] = [text_idx]
            else:
                result[label_idx].append(text_idx)
        return result


if __name__ == '__main__':
    dbscan = DbscanClustering()
    result=dbscan.dbscan(eps=0.1, min_samples=1000)
    id = []
    family_id = []
    with open("dynamic_feature.csv.pkl", 'rb') as f:
        names = pickle.load(f)
        # corpus = pickle.load(f)
    print(result)
    for key in result:
        for c in result[key]:
            family_id.append(key+1)
            id.append(names[c])
    dataframe = pd.DataFrame({'id': id, 'family_id': family_id})
    dataframe.to_csv("submit_2.csv", index=False, sep=',')
