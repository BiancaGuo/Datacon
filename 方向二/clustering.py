import pandas as pd

corpus_path_2="API_name_feature_small.csv"
names_2=[]
data_2 = pd.read_csv(corpus_path_2)
goup_fileid_2 = data_2.groupby('id')
for file_name_2, file_group_2 in goup_fileid_2:
    names_2.append(file_name_2)

corpus_path_1="D:\\大四稳定起航\\Datacon\\方向一\\API_name_feature_train_washed.csv"
names_1=[]
data_1 = pd.read_csv(corpus_path_1)
goup_fileid_1 = data_1.groupby('id')
for file_name_1, file_group_1 in goup_fileid_1:
    names_1.append(file_name_1)

corpus_path_3="D:\\大四稳定起航\\Datacon\\方向一\\API_name_feature_test_washed.csv"
names_3=[]
data_3 = pd.read_csv(corpus_path_3)
goup_fileid_3 = data_3.groupby('id')
for file_name_3, file_group_3 in goup_fileid_3:
    names_3.append(file_name_3)

for n in names_2:
    if n in names_1:
        print("train:"+n)
    if n in names_3:
        print("test:"+n)
