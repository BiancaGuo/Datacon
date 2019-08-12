import pickle
import xgboost as xgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
'''
XGBoost是boosting算法的其中一种。Boosting算法的思想是将许多弱分类器集成在一起形成一个强分类器。因为XGBoost是一种提升树模型，所以它是将许多树模型集成在一起，形成一个很强的分类器。而所用到的树模型则是CART回归树模型。
'''

with open("cnn_and_lstm_result_0.pkl", "rb") as f:
    cnn_lstm_train_result_0 = pickle.load(f)

with open("tfidf_result_0.pkl", "rb") as f:
    tfidf_train_result_0 = pickle.load(f)

with open("cnn_and_lstm_result_1.pkl", "rb") as f:
    cnn_lstm_train_result_1 = pickle.load(f)

with open("tfidf_result_1.pkl", "rb") as f:
    tfidf_train_result_1 = pickle.load(f)

with open("cnn_and_lstm_result_2.pkl", "rb") as f:
    cnn_lstm_train_result_2 = pickle.load(f)

with open("tfidf_result_2.pkl", "rb") as f:
    tfidf_train_result_2 = pickle.load(f)

with open("cnn_and_lstm_result_3.pkl", "rb") as f:
    cnn_lstm_train_result_3 = pickle.load(f)

with open("tfidf_result_3.pkl", "rb") as f:
    tfidf_train_result_3 = pickle.load(f)

with open("cnn_and_lstm_result_4.pkl", "rb") as f:
    cnn_lstm_train_result_4 = pickle.load(f)

with open("tfidf_result_4.pkl", "rb") as f:
    tfidf_train_result_4 = pickle.load(f)

with open("cnn_and_lstm_result_5.pkl", "rb") as f:
    cnn_lstm_train_result_5 = pickle.load(f)

with open("tfidf_result_5.pkl", "rb") as f:
    tfidf_train_result_5 = pickle.load(f)

with open("cnn_and_lstm_result_6.pkl", "rb") as f:
    cnn_lstm_train_result_6 = pickle.load(f)

with open("tfidf_result_6.pkl", "rb") as f:
    tfidf_train_result_6 = pickle.load(f)

with open("cnn_and_lstm_result_7.pkl", "rb") as f:
    cnn_lstm_train_result_7 = pickle.load(f)

with open("tfidf_result_7.pkl", "rb") as f:
    tfidf_train_result_7 = pickle.load(f)

with open("cnn_and_lstm_result_8.pkl", "rb") as f:
    cnn_lstm_train_result_8 = pickle.load(f)

with open("tfidf_result_8.pkl", "rb") as f:
    tfidf_train_result_8 = pickle.load(f)

with open("cnn_and_lstm_result_9.pkl", "rb") as f:
    cnn_lstm_train_result_9 = pickle.load(f)

with open("tfidf_result_9.pkl", "rb") as f:
    tfidf_train_result_9 = pickle.load(f)

with open("cnn_and_lstm_result_0_train.pkl", "rb") as f:
    cnn_lstm_train_result_0_train = pickle.load(f)

with open("tfidf_result_0_train.pkl", "rb") as f:
    tfidf_train_result_0_train = pickle.load(f)

with open("cnn_and_lstm_result_1_train.pkl", "rb") as f:
    cnn_lstm_train_result_1_train = pickle.load(f)

with open("tfidf_result_1_train.pkl", "rb") as f:
    tfidf_train_result_1_train = pickle.load(f)

with open("cnn_and_lstm_result_2_train.pkl", "rb") as f:
    cnn_lstm_train_result_2_train = pickle.load(f)

with open("tfidf_result_2_train.pkl", "rb") as f:
    tfidf_train_result_2_train = pickle.load(f)

with open("cnn_and_lstm_result_3_train.pkl", "rb") as f:
    cnn_lstm_train_result_3_train = pickle.load(f)

with open("tfidf_result_3_train.pkl", "rb") as f:
    tfidf_train_result_3_train = pickle.load(f)

with open("cnn_and_lstm_result_4_train.pkl", "rb") as f:
    cnn_lstm_train_result_4_train = pickle.load(f)

with open("tfidf_result_4_train.pkl", "rb") as f:
    tfidf_train_result_4_train = pickle.load(f)

with open("cnn_and_lstm_result_5_train.pkl", "rb") as f:
    cnn_lstm_train_result_5_train = pickle.load(f)

with open("tfidf_result_5_train.pkl", "rb") as f:
    tfidf_train_result_5_train = pickle.load(f)

with open("cnn_and_lstm_result_6_train.pkl", "rb") as f:
    cnn_lstm_train_result_6_train = pickle.load(f)

with open("tfidf_result_6_train.pkl", "rb") as f:
    tfidf_train_result_6_train = pickle.load(f)

with open("cnn_and_lstm_result_7_train.pkl", "rb") as f:
    cnn_lstm_train_result_7_train = pickle.load(f)

with open("tfidf_result_7_train.pkl", "rb") as f:
    tfidf_train_result_7_train = pickle.load(f)

with open("cnn_and_lstm_result_8_train.pkl", "rb") as f:
    cnn_lstm_train_result_8_train = pickle.load(f)

with open("tfidf_result_8_train.pkl", "rb") as f:
    tfidf_train_result_8_train = pickle.load(f)

with open("cnn_and_lstm_result_9_train.pkl", "rb") as f:
    cnn_lstm_train_result_9_train = pickle.load(f)

with open("tfidf_result_9_train.pkl", "rb") as f:
    tfidf_train_result_9_train = pickle.load(f)

with open("cnn_result_0.pkl", "rb") as f:
    cnn_result_0 = pickle.load(f)

with open("cnn_result_1.pkl", "rb") as f:
    cnn_result_1 = pickle.load(f)

with open("cnn_result_2.pkl", "rb") as f:
    cnn_result_2 = pickle.load(f)

with open("cnn_result_3.pkl", "rb") as f:
    cnn_result_3 = pickle.load(f)

with open("cnn_result_4.pkl", "rb") as f:
    cnn_result_4 = pickle.load(f)

with open("cnn_result_5.pkl", "rb") as f:
    cnn_result_5 = pickle.load(f)

with open("cnn_result_6.pkl", "rb") as f:
    cnn_result_6 = pickle.load(f)

with open("cnn_result_7.pkl", "rb") as f:
    cnn_result_7 = pickle.load(f)

with open("cnn_result_8.pkl", "rb") as f:
    cnn_result_8 = pickle.load(f)

with open("cnn_result_9.pkl", "rb") as f:
    cnn_result_9 = pickle.load(f)

with open("cnn_result_0_train.pkl", "rb") as f:
    cnn_result_0_train = pickle.load(f)

with open("cnn_result_1_train.pkl", "rb") as f:
    cnn_result_1_train = pickle.load(f)

with open("cnn_result_2_train.pkl", "rb") as f:
    cnn_result_2_train = pickle.load(f)

with open("cnn_result_3_train.pkl", "rb") as f:
    cnn_result_3_train = pickle.load(f)

with open("cnn_result_4_train.pkl", "rb") as f:
    cnn_result_4_train = pickle.load(f)

with open("cnn_result_5_train.pkl", "rb") as f:
    cnn_result_5_train = pickle.load(f)

with open("cnn_result_6_train.pkl", "rb") as f:
    cnn_result_6_train = pickle.load(f)

with open("cnn_result_7_train.pkl", "rb") as f:
    cnn_result_7_train = pickle.load(f)

with open("cnn_result_8_train.pkl", "rb") as f:
    cnn_result_8_train = pickle.load(f)

with open("cnn_result_9_train.pkl", "rb") as f:
    cnn_result_9_train = pickle.load(f)

with open("dynamic_feature_test.csv.pkl", "rb") as f:
    names = pickle.load(f)
    test_APIs = pickle.load(f)

with open("dynamic_feature_train.csv.pkl", "rb") as f:
    labels = pickle.load(f)
    train_APIs = pickle.load(f)

# train = np.hstack([tfidf_train_result, cnn_train_result, cnn_lstm_train_result,lstm_train_result])
# test = np.hstack([tfidf_out_result, cnn_out_result, lstm_out_result, cnn_lstm_out_result])
train = np.hstack([cnn_result_0_train,cnn_result_1_train,cnn_result_2_train,cnn_result_3_train,cnn_result_4_train,cnn_result_5_train,cnn_result_6_train,cnn_result_7_train,cnn_result_8_train,cnn_result_9_train,tfidf_train_result_0_train,tfidf_train_result_1_train,tfidf_train_result_2_train,tfidf_train_result_3_train,tfidf_train_result_4_train,tfidf_train_result_5_train,tfidf_train_result_6_train,tfidf_train_result_7_train,tfidf_train_result_8_train,tfidf_train_result_9_train,cnn_lstm_train_result_0_train,cnn_lstm_train_result_1_train,cnn_lstm_train_result_2_train,cnn_lstm_train_result_3_train,cnn_lstm_train_result_4_train,cnn_lstm_train_result_5_train,cnn_lstm_train_result_6_train,cnn_lstm_train_result_7_train,cnn_lstm_train_result_8_train,cnn_lstm_train_result_9_train])
test = np.hstack([cnn_result_0,cnn_result_1,cnn_result_2,cnn_result_3,cnn_result_4,cnn_result_5,cnn_result_6,cnn_result_7,cnn_result_8,cnn_result_9,tfidf_train_result_0,tfidf_train_result_1,tfidf_train_result_2,tfidf_train_result_3,tfidf_train_result_4,tfidf_train_result_5,tfidf_train_result_6,tfidf_train_result_7,tfidf_train_result_8,tfidf_train_result_9,cnn_lstm_train_result_0,cnn_lstm_train_result_1,cnn_lstm_train_result_2,cnn_lstm_train_result_3,cnn_lstm_train_result_4,cnn_lstm_train_result_5,cnn_lstm_train_result_6,cnn_lstm_train_result_7,cnn_lstm_train_result_8,cnn_lstm_train_result_9])

# train = np.hstack([cnn_result_0_train,cnn_result_1_train,cnn_result_2_train,cnn_result_3_train,tfidf_train_result_0_train,tfidf_train_result_1_train,tfidf_train_result_2_train,cnn_lstm_train_result_0_train,cnn_lstm_train_result_1_train,cnn_lstm_train_result_2_train])
# test = np.hstack([cnn_result_0,cnn_result_1,cnn_result_2,cnn_result_3,tfidf_train_result_0,tfidf_train_result_1,tfidf_train_result_2,cnn_lstm_train_result_0,cnn_lstm_train_result_1,cnn_lstm_train_result_2])

meta_test = np.zeros(shape=(len(test_APIs), 2))
skf = StratifiedKFold(n_splits=10, random_state=4, shuffle=True)
dout = xgb.DMatrix(test)
for i, (tr_ind, te_ind) in enumerate(skf.split(train, labels)):
    print('FOLD: {}'.format(str(i)))
    X_train, X_train_label = train[tr_ind], np.array(labels)[tr_ind]
    X_val, X_val_label = train[te_ind], np.array(labels)[te_ind]
    dtrain = xgb.DMatrix(X_train, label=X_train_label)
    dtest = xgb.DMatrix(X_val, X_val_label)  # label可以不要，此处需要是为了测试效果

    param = {'max_depth': 6, 'eta': 0.01, 'eval_metric': 'mlogloss', 'silent': 1, 'objective': 'multi:softprob',
             'num_class': 2, 'subsample': 0.9,
             'colsample_bytree': 0.85}  # 参数
    evallist = [(dtrain, 'train'), (dtest, 'val')]  # 测试 , (dtrain, 'train')
    num_round = 3000  # 循环次数
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=100)
    ans = bst.predict(dtest)

    predict_type_list = []
    for l in ans:
        l_tmp = l.tolist()
        predict_type = l_tmp.index(max(l_tmp))
        predict_type_list.append(predict_type)

    # 计算准确率
    cnt1 = 0
    cnt2 = 0
    for i in range(len(predict_type_list)):
        if X_val_label[i] == predict_type_list[i]:
            cnt1 += 1
        else:
            cnt2 += 1
    print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))

    preds = bst.predict(dout)
    meta_test += preds

meta_test /= 10.0
result = meta_test
# print(result)
predict_type_list = []
for l in result:
    l_tmp = l.tolist()
    predict_type = l_tmp.index(max(l_tmp))
    predict_type_list.append(predict_type)
dataframe = pd.DataFrame({'id': names, 'safe_type': predict_type_list})
dataframe.to_csv("submit_2.csv", index=False, sep=',')