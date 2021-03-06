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

with open("dynamic_feature_train.csv.pkl", "rb") as f:
    labels_d = pickle.load(f)
with open("dynamic_feature_train.csv.pkl", "rb") as f:
    labels = pickle.load(f)
    files = pickle.load(f)

maxlen = 2000
labels = np.asarray(labels)
labels = to_categorical(labels, num_classes=2)#label词向量
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
x_train_word_ids = tokenizer.texts_to_sequences(files)#用于向量化文本,将文本转换为序列
x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=maxlen)#将序列填充到maxlen长度
vocab = tokenizer.word_index


MAX_NB_WORDS = 20000
def text_cnn_model():
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


# meta_train = np.zeros(shape=(len(x_train_padded_seqs), 7))
# meta_test = np.zeros(shape=(len(x_out_padded_seqs), 8))
skf = StratifiedKFold(n_splits=10, random_state=4, shuffle=True)
for i, (tr_ind, te_ind) in enumerate(skf.split(x_train_padded_seqs, labels_d)):
    print('FOLD: {}'.format(str(i)))
    print(len(te_ind), len(tr_ind))
    X_train, X_train_label = x_train_padded_seqs[tr_ind], np.array(labels)[tr_ind]
    X_val, X_val_label = x_train_padded_seqs[te_ind], np.array(labels)[te_ind]

    model = text_cnn_model()
    # print(np.shape(X_val))
    # print(np.shape(X_val_label))
    model_save_path = './model/model_weight_train_cnn_{}.h5'.format(str(i))
    if i in [-1]:
        model = load_model(model_save_path)
        print(model.evaluate(X_val, X_val_label))
    else:
        ear = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='min', baseline=None,
                            restore_best_weights=False)

        model_checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, save_weights_only=False)
        history = model.fit(X_train, X_train_label,
                            batch_size=32,
                            epochs=200,
                            shuffle=True,
                            validation_data=(X_val, X_val_label), callbacks=[tensorboard,ear,model_checkpoint])
    K.clear_session()



