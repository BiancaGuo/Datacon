# 算法文档说明

## 一、 create_train_API_feature_csv.py

- 遍历train文件夹下所有文件，提取出每个文档中的api_name项，结合对应类别存入csv文件中。删掉相邻且相同的API。

![](https://i.imgur.com/dzOkM6m.jpg)

## 二、 create_test_API_feature_csv.py

- 遍历test文件夹下所有文件，提取出每个文档中的api_name项，存入csv文件中。

![](https://i.imgur.com/f4VCVDy.jpg)

## 三、create_word_dictionary.py

- 使用TfidfVectorizer() 将文本中的词语转换为词频矩阵,将矩阵存入pkl文件中，用于tf-idf模型的建立。
- 使用Tokenizer()将文本转换为序列，存入pkl文件中用于text-cnn模型的建模。

## 四、load_file.py

- 按文件名将api和lable从csv文件中提取出来存入pkl文件中。

## 五、train_cnn_lstm.py

- 使用cnn结合lstm的网络进行分类器训练。

## 六、train_tfidf.py

- 训练tf-idf模型分类器。

## 七、train_text_cnn.py

- 训练text-CNN模型分类器

## 八、model_test.py

- 使用生成的各个模型对测试集进行分类。

## 九、stack_result.py

- 使用xgboost将各个模型集成，在训练集上再次训练，然后对测试集进行分类。
