# -*- coding: UTF-8 -*-
import re
from collections import *
import os
import pandas as pd
import json



def getOpcodeSequence(filename,path):
    opcode_seq = []
    data = pd.read_csv(path)
    goup_fileid = data.groupby('id')
    print(len(goup_fileid))
    for file_name, file_group in goup_fileid:
        if file_name==filename:
            result = file_group
            count=0
            for c in result['api']:
                opcode_seq.append(c)
                count+=1
                if count>=1000:
                    break
            break
    data = data.drop(data[(data.id == filename)].index.tolist())
    data.to_csv(path, index=False, sep=',')
    return opcode_seq

def getOpcodeNgram(ops, n=4):#n-gram，决定n的大小 可以改进！！
    opngramlist = [tuple(ops[i:i+n]) for i in range(len(ops)-n)]
    opngram = Counter(opngramlist)
    return opngram

def getStaticFeatures():
    file = open('selected_static_features.txt', 'r')
    js = file.read()
    dic = json.loads(js)
    file.close()
    return dic

if __name__=="__main__":
    subtrainLabel = pd.read_csv('..\\deep_learning\\submit_2.csv')
    subtrainLabel2 = pd.read_csv('..\\deep_learning\\result_2.csv')
    s1=subtrainLabel.safe_type
    s2=subtrainLabel2.safe_type
    count=0
    for i in range(0,len(s1)):
        if s1[i]!= s2[i]:
            count+=1
    print(count)

    # map3gram = defaultdict(Counter)
    # subtrain = pd.read_csv('..\\safe_type_train.csv')#训练集文件与类别对应关系
    # count = 1
    # for Id, Class in zip(subtrain.id, subtrain.safe_type):#id：文件md5值，safe_type：对应类别（2类）
    #     print(Id)
    #     count += 1
    #     path = r'../API_name_feature_train_washed2.csv'
    #     ops = getOpcodeSequence(Id,path) #当每个文件中的opcode放入list中
    #     op3gram = getOpcodeNgram(ops) #获得ngram组
    #     map3gram[Id] = op3gram #将每个文件的ngram组（opcodes：个数）防暑map3gram列表中
    #
    # cc = Counter([])
    #
    # count=1
    # for d in map3gram.values():
    #     print(count)
    #     count=count+1
    #     cc += d
    #
    # #cc:Counter({('push', 'add'): 84, ('push', 'mov'): 80})
    # selectedfeatures = {}
    # tc = 0
    # for k,v in cc.iteritems():
    #     if v > 5000:#ngram出现次数大于500次 特征选取的方式 可以改进！！！
    #         selectedfeatures[str(k)] = v
    #         print(k,v)
    #         tc += 1
    #
    # js = json.dumps(selectedfeatures)
    # file = open('selected_static_features_4gram.txt', 'w')
    # file.write(js)
    # file.close()
    #
    # dataframelist = []
    # #将每个文件中在selectedfeatures中的ngram挑选出来
    # for fid,op3gram in map3gram.iteritems():#{txt1: Counter({('push', 'add'): 84}), txt2: Counter({('push', 'mov'): 80})})
    #     standard = {}
    #     standard["Id"] = fid
    #     for feature in selectedfeatures:
    #         temp = feature.replace('(', '').replace(')', '').replace('\'', '')
    #         feature_tuple = tuple([str(i).strip() for i in temp.split(',')])
    #         if feature_tuple in op3gram:
    #             standard[feature_tuple] = op3gram[feature_tuple]
    #         else:
    #             standard[feature_tuple] = 0
    #     dataframelist.append(standard)
    # df = pd.DataFrame(dataframelist)
    # df.to_csv("4gramfeature.csv",index=False)