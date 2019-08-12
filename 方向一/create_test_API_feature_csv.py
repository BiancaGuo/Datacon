
#遍历文件夹下每一个文件，提取API序列
#API+文件名+label = API_feature.csv
import os
import  xml.dom.minidom
import pandas as pd

def get_all_files(rootdir):
    files = []
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isdir(path):
            files.extend(get_all_files(path))
        if os.path.isfile(path):
            files.append(path)
    return files

def get_api_from_file(filename):
    apis=[]
    dom = xml.dom.minidom.parse(filename)
    root = dom.documentElement
    itemlist = root.getElementsByTagName('action')
    api=itemlist[0].getAttribute("api_name")
    apis.append(api)
    for i in range(1,len(itemlist)):
        api = itemlist[i].getAttribute("api_name")
        if api!=itemlist[i-1].getAttribute("api_name"):
            apis.append(api)
    return apis



#black-1
#white-0
if __name__ =="__main__":
    # data_path="F:\\stage1_dataset\\test\\"
    # id = []
    # safe_type = []
    # api = []
    # count=0
    # test_files=get_all_files(data_path)
    # for file in test_files:
    #     print(file)
    #     apis = get_api_from_file(file)
    #     file_id = os.path.basename(file)[:-4]
    #     file_len = len(apis)
    #     for i in range(0, file_len):
    #         id.append(file_id)
    #         api.append(apis[i])
    #     count += 1
    #     if count==100:
    #         print(count)
    #         dataframe = pd.DataFrame({'id': id, 'api': api})
    #         dataframe.to_csv("API_name_feature_test_washed.csv", index=False, sep=',')
    #         id=[]
    #         safe_type=[]
    #         api=[]
    #     if count%1000==0:
    #         print(count)
    #         dataframe = pd.DataFrame({'id': id, 'api': api})
    #         dataframe.to_csv("API_name_feature_test_washed.csv", mode="a",index=False, sep=',')
    #         id = []
    #         safe_type = []
    #         api = []
    #

    #code_test
    # file="F:\\stage1_dataset\\train\\0\\5449ffb8198a42c2fe5703e1aae2113178f5f60d40128f190a09584fa135b.xml"
    # apis = get_api_from_file(file)
    # print(apis)
    # print(os.path.basename(file)[:-4])

    data = pd.read_csv(r"API_name_feature_test_washed.csv")
    print(data[(data.id == 'id')].index.tolist())
    data = data.drop(data[(data.id == 'id')].index.tolist())
    # print(data['api'].isnull().value_counts())
    # data=data.dropna(subset=['api'])
    data.to_csv("API_name_feature_test_washed.csv", index=False, sep=',')