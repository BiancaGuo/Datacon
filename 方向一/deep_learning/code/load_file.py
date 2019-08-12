import pandas as pd
import pickle


train_path = r'../API_name_feature_train_washed.csv'
test_path = r'../API_name_feature_test_washed.csv'

#'id': id, 'safe_type': safe_type, 'api': api
def read_train_file(path):
    labels = []
    apis = []
    data = pd.read_csv(path)
    # for data in data1:
    goup_fileid = data.groupby('id')
    print(len(goup_fileid))
    for file_name, file_group in goup_fileid:
        print(file_name)
        file_labels = file_group['safe_type'].values[0]
        # print(file_labels)
        result = file_group
        api_sequence = ' '.join(result['api'])
        labels.append(int(file_labels))
        apis.append(api_sequence)
    # print(len(labels))
    # print(len(files))
    # with open(path.split('/')[-1] + ".txt", 'w') as f:
    #     for i in range(len(labels)):
    #         f.write(str(labels[i]) + ' ' + files[i] + '\n')
    with open("dynamic_feature_train.csv.pkl", 'wb') as f:
        pickle.dump(labels, f)
        pickle.dump(apis, f)



def read_test_file(path):
    names = []
    files = []
    data = pd.read_csv(path)
    # for data in data1:
    goup_fileid = data.groupby('id')
    print(len(goup_fileid))
    for file_name, file_group in goup_fileid:

        api_sequence = ' '.join(file_group['api'])
        names.append(file_name)
        files.append(api_sequence)
    with open("dynamic_feature_test.csv.pkl", 'wb') as f:
        pickle.dump(names, f)
        pickle.dump(files, f)

# def load_train2h5py(path="security_train.csv.txt"):
#     labels = []
#     files = []
#     with open(path) as f:
#         for i in f.readlines():
#             i = i.strip('\n')
#             labels.append(i[0])
#             files.append(i[2:])
#     labels = np.asarray(labels)
#     print(labels.shape)
#     with open("security_train.csv.pkl", 'wb') as f:
#         pickle.dump(labels, f)
#         pickle.dump(files, f)


# def load_test2h5py(path="D:\ML_Malware\security_test.csv.txt"):
#     labels = []
#     files = []
#     with open(path) as f:
#         for i in f.readlines():
#             i = i.strip('\n')
#             labels.append(i[0])
#             files.append(' '.join(i.split(" ")[1:]))
#     labels = np.asarray(labels)
#     print(labels.shape)
#     with open("security_test.csv.pkl", 'wb') as f:
#         pickle.dump(labels, f)
#         pickle.dump(files, f)


if __name__ == '__main__':
    # print("read train file.....")
    # read_train_file(train_path)
    # print("read test file......")
    read_test_file(test_path)

