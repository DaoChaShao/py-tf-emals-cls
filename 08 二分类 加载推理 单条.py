from sklearn.model_selection import train_test_split  # 用于划分训练集和测试集
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # 用于标签编码
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout  # 用于构建网络层

import chardet
import numpy as np
import os
import pickle
import random
import thulac
import tqdm
import tensorflow as tf

from utils import timer, lines, red, green, yellow


class DataProcessor(object):
    """ 处理数据集 """

    @classmethod
    def file_encod_detector(cls, file_path: str) -> str:
        """ 检测文件编码 """
        with open(file_path, "rb") as file:
            detector = file.read()
        file_details = chardet.detect(detector)
        file_code = file_details["encoding"].lower()
        if file_code == "gb2312":
            file_code = "gb18030"
        file_lang = file_details["language"]
        # print(f"文件编码：{file_code}")
        # print(f"文件语言：{file_lang}")
        # cui.draw_lines()
        return file_code

    @classmethod
    def stop_words_reader(cls, file_path: str) -> list:
        """ 获取停用词 """
        # 读取文件编码
        code_mode = cls.file_encod_detector(file_path)
        # print(f"停用词文件编码：{code_mode}")
        # 打开停用词文件
        with open(file_path, "r", encoding=code_mode) as file:
            # 将停用词添加至列表
            stop_words = [word.strip() for word in file.readlines()]
            # print(f"停用词列表长度：{len(stop_words)}")  # DEBUG
            # print(f"停用词列表内容：{stop_words}")  # DEBUG
            return stop_words

    @classmethod
    def add_words_reader(cls, file_path: str) -> list:
        """ 获取添加词 """
        # 读取文件编码
        code_mode = cls.file_encod_detector(file_path)
        # print(f"添加词文件编码：{code_mode}")
        # 打开添加词文件
        with open(file_path, "r", encoding=code_mode) as file:
            # 将添加词添加至列表
            add_words = [word.strip() for word in file.readlines()]
            # print(f"添加词列表长度：{len(add_words)}")  # DEBUG
            # print(f"添加词列表内容：{add_words}")  # DEBUG
            return add_words

    @classmethod
    def data_reader(cls, file_path: str) -> str | None:
        """ 获取微博评论数据 """
        # 打开数据集文件
        with open(file_path, "r", encoding="GBK") as file:
            # 由于数据结构是一行一行的，且第一行是标签，数据读取应该出去第一行的按行读取
            content = file.read()
            # print(f"微博评论内容：\n{weibo_comments}")
            # print(f"微博评论总数：{len(weibo_comments)}")  # 119988
            return content

    @classmethod
    def chinese_checker(cls, char: str) -> bool:
        """
        判断一个字符是否是汉字
        \u4e00 到 \u9fa5 是基本汉字的范围。
        \u9fa6 到 \u9fff 则包含了一些少用或扩展的汉字。
        :param char: str
        :return: bool value
        """
        if '\u4e00' <= char <= '\u9fff':
            return True
        else:
            return False

    @classmethod
    def chinese_keeper(cls, text: str) -> str:
        """
        保留中文字符
        :param text: str
        :return: str
        """
        # 初始化一个空字符串，用于存储提取出的汉字字符
        outcome = ""
        for char in text:
            if cls.chinese_checker(char):
                outcome += char
        return outcome

    @classmethod
    def tokenizer(cls, text: str, nlp_seg: thulac.thulac) -> list:
        """ 分词器 """
        words = nlp_seg.cut(text, text=True).split()
        return words

    @classmethod
    def file_opener(cls, root_path: str) -> list:
        """ 获取文件名列表 """
        # 初始化文件名列表
        file_names = []
        # 遍历目录
        for folder_names in os.listdir(root_path):
            # print(f"根目录下的文件夹：{folder_names}")
            # 排除隐藏文件
            if folder_names.startswith("."):
                continue
            # 筛出文件夹
            if os.path.isdir(os.path.join(root_path, folder_names)):
                # print(f"筛选出的文件夹：{folder_names}")
                file_name = os.path.join(root_path, folder_names)
                # print(f"文件夹路径：{file_name}")
                file_names.append(file_name)
        # 打印文件名列表
        # print(f"文件夹路径列表：{file_names}")
        return file_names

    @classmethod
    def label_filler(cls, path_list: list, range_num: int | None) \
            -> tuple[list, list, list, list, list]:
        """ 获取正式数据 """
        # 初始化文件夹路径列表
        path_normal = None
        path_spam = None
        path_test = None
        # 初始化文件列表
        raw_nor_labels = []
        raw_nor_contents = []
        raw_spam_labels = []
        raw_spam_contents = []
        raw_test_contents = []
        # 文件路径拆包
        for path in path_list:
            match path:
                case path if "normal" in path:
                    path_normal = path
                    # print(f"正常邮件总路径：{path_normal}")
                case path if "spam" in path:
                    path_spam = path
                    # print(f"垃圾邮件总路径：{path_spam}")
                case path if "test" in path:
                    path_test = path
                    # print(f"测试邮件总路径：{path_test}")
                case _:
                    print("无此文件夹")
        # 读取正常邮件数据
        for file_name in tqdm.tqdm(os.listdir(path_normal)[:range_num], desc="正常邮件读取中：", colour="green"):
            file_path = os.path.join(path_normal, file_name)
            # 读取文件内容
            content = cls.data_reader(file_path)
            raw_nor_contents.append(content)
            raw_nor_labels.append(int(0))
        # 读取垃圾邮件数据
        for file_name in tqdm.tqdm(os.listdir(path_spam)[:range_num], desc="垃圾邮件读取中：", colour="red"):
            file_path = os.path.join(path_spam, file_name)
            # 读取文件内容
            content = cls.data_reader(file_path)
            raw_spam_contents.append(content)
            raw_spam_labels.append(int(1))
        # 读取测试邮件数据
        for file_name in tqdm.tqdm(os.listdir(path_test)[:range_num], desc="测试邮件读取中：", colour="yellow"):
            file_path = os.path.join(path_test, file_name)
            content = cls.data_reader(file_path)
            raw_test_contents.append(content)
        # 打印数据
        # print(f"正常邮件标签：{raw_nor_labels}")
        # print(f"正常邮件内容：{raw_nor_contents}")
        # print(f"垃圾邮件标签：{raw_spam_labels}")
        # print(f"垃圾邮件内容：{raw_spam_contents}")
        # print(f"测试邮件内容：{raw_test_contents}")
        return raw_nor_labels, raw_nor_contents, raw_spam_labels, raw_spam_contents, raw_test_contents

    @classmethod
    def text_cleaner(cls, texts: list, thu_init: thulac.thulac, stop_words: list) -> list:
        """ 清理文本 """
        contents = []
        for text in texts:
            # 去除非中文字符
            comment = cls.chinese_keeper(text)
            # 进行分词
            words = cls.tokenizer(comment, thu_init)
            # 去除停用词
            words = [word for word in words if word not in stop_words]
            # 如果分词后为空，则跳过
            if not words:
                continue
            # 添加标签和评论
            contents.append(words)
        return contents

    @classmethod
    def get_formal_data(
            cls, thu_init: thulac.thulac,
            nor_data: list, spam_data: list, test_data: list,
            stop_words: list
    ) -> tuple[list, list, list]:
        """ 获取正式数据 """
        # 获取正常邮件的干净数据
        nor_clean_data = cls.text_cleaner(nor_data, thu_init, stop_words)
        # 获取垃圾邮件的干净数据
        spam_clean_data = cls.text_cleaner(spam_data, thu_init, stop_words)
        # 获取测试邮件的干净数据
        test_clean_data = cls.text_cleaner(test_data, thu_init, stop_words)
        # 打印数据
        # print(f"正常邮件数据：{nor_clean_data}")
        # print(f"垃圾邮件数据：{spam_clean_data}")
        # print(f"测试邮件数据：{test_clean_data}")
        return nor_clean_data, spam_clean_data, test_clean_data

    @classmethod
    def data_packer(
            cls,
            nor_labels: list, nor_contents: list,
            spam_labels: list, spam_contents: list
    ) -> tuple[list, list]:
        """ 数据打包 """
        # # 检查数据长度
        if len(nor_labels) != len(nor_contents):
            nor_labels = nor_labels[:len(nor_contents)]
        if len(spam_labels) != len(spam_contents):
            spam_labels = spam_labels[:len(spam_contents)]

        # 初始化标签列表
        labels = []
        contents = []
        # 数据混包
        nor_package = [(nor_labels[i], nor_contents[i]) for i in range(len(nor_labels))]
        spam_package = [(spam_labels[i], spam_contents[i]) for i in range(len(spam_labels))]
        package = nor_package + spam_package
        # print(f"数据打包：{package}")
        # print(f"数据总数：{cpc.PrintColour.red(len(package))}")
        random.shuffle(package)
        # print(f"打乱数据：{package}")
        # print(f"打乱后数据总数：{cpc.PrintColour.red(len(package))}")
        for label, content in package:
            labels.append(label)
            contents.append(content)
        return labels, contents

    @classmethod
    def data_classifier(cls, labels: list, contents: list) -> tuple[tuple[list, list], tuple[list, list]]:
        """ 数据分类 """
        # 训练集站 80%，验证集 20%
        train_y, valid_y, train_x, valid_x = train_test_split(labels, contents, test_size=0.2, random_state=27)
        # 输出各个数据集的大小
        # print(f"训练集数量：{cpc.PrintColour.red(len(train_x))}")
        # print(f"验证集数量：{cpc.PrintColour.red(len(valid_x))}")
        # print(f"数据集总数：{cpc.PrintColour.red(len(train_x) + len(valid_x))}")
        # for i, (label, content) in enumerate(zip(train_y, train_x)):
        #     print(f"训练集第 {i + 1} 条的标签是 {label}，评论：{content}")
        #     cui.draw_lines()
        # for i, (label, content) in enumerate(zip(valid_y, valid_x)):
        #     print(f"验证集第 {i + 1} 条的标签是 {label}，评论：{content}")
        return (train_y, train_x), (valid_y, valid_x)


class DataEncoder(object):
    """ 词编码器 """

    @classmethod
    def label_encoder(cls, train: tuple[list, list], valid: tuple[list, list]) \
            -> tuple[list, list]:
        """ 标签编码器 """
        # 数据拆包
        train_y = train[0]
        valid_y = valid[0]
        # 实例化标签编码器
        le = LabelEncoder()
        # 标签编码后，增加至二维
        train_y = le.fit_transform(train_y).reshape(-1, 1)
        valid_y = le.transform(valid_y).reshape(-1, 1)
        # 实例化 One-Hot 编码器
        ohe = OneHotEncoder()
        train_y_ohe = ohe.fit_transform(train_y).toarray()
        valid_y_ohe = ohe.transform(valid_y).toarray()
        # 输出各个数据集的标签
        # print(f"训练集标签编码：{train_y}")
        # print(f"验证集标签编码：{valid_y}")
        # print(f"训练集标签编码维度：{cpc.PrintColour.yellow(train_y_ec.shape)}")
        # print(f"验证集标签编码维度：{cpc.PrintColour.yellow(valid_y_ec.shape)}")
        lines()
        return train_y_ohe, valid_y_ohe

    @classmethod
    def word_encoder(
            cls, contents: list, vocab_size: int,
            train: tuple[list, list], valid: tuple[list, list], test: list,
    ) -> tuple[list, list, list]:
        """ 词编码器 """
        # 数据拆包
        train_x = train[1]
        valid_x = valid[1]
        test_x = test
        # 实例化 tensorflow 的分词器（如果测试集中包含训练集未见过的词，Tokenizer 会将这些词映射到 <UNK> 标记）
        tokens = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="UNK")
        # 统计频率（遍历所有文本，统计每个词出现的次数），排序（根据词频从高到低对词汇进行排序）并根据这些频率构建一个词汇列表
        tokens.fit_on_texts(contents)
        # 评论编码
        train_x_seq = tokens.texts_to_sequences(train_x)
        valid_x_seq = tokens.texts_to_sequences(valid_x)
        test_x_seq = tokens.texts_to_sequences(test_x)
        # 打印评论编码
        # print(f"训练集内容编码：{train_x_seq}")
        # print(f"验证集内容编码：{valid_x_seq}")
        # print(f"测试集内容编码：{test_x_seq}")
        # print(f"训练集内容编码长度：{cpc.PrintColour.green(len(train_x_seq))}")
        # print(f"验证集内容编码长度：{cpc.PrintColour.green(len(valid_x_seq))}")
        # print(f"测试集内容编码长度：{cpc.PrintColour.green(len(test_x_seq))}")
        return train_x_seq, valid_x_seq, test_x_seq

    @classmethod
    def word_filler(cls, train: list, valid: list, test: list, max_len: int) -> tuple[list, list, list]:
        """ 评论长度填充 """
        # 实例化 tensorflow 的 pad_sequences 函数
        pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
        # 训练集填充
        train_x_equal = pad_sequences(train, maxlen=max_len)
        # 验证集填充
        valid_x_equal = pad_sequences(valid, maxlen=max_len)
        # 测试集填充
        test_x_equal = pad_sequences(test, maxlen=max_len)
        # 打印填充后的数据
        # for line in train_x_equal[:1]:
        #     print(f"训练集内容填充：{line.shape}")
        # for line in valid_x_equal[:1]:
        #     print(f"验证集内容填充：{line.shape}")
        # for line in test_x_equal[:1]:
        #     print(f"测试集内容填充：{line.shape}")
        # cui.draw_lines()
        # for line in train_x_equal[:1]:
        #     print(f"训练集内容填充长度：{cpc.PrintColour.purple(len(line))}")
        # for line in valid_x_equal[:1]:
        #     print(f"验证集内容填充长度：{cpc.PrintColour.purple(len(line))}")
        # for line in test_x_equal[:1]:
        #     print(f"测试集内容填充长度：{cpc.PrintColour.purple(len(line))}")
        # cui.draw_lines()
        return train_x_equal, valid_x_equal, test_x_equal


class EmailClassifierRNNNet(object):
    """ 构建 RNN 网络 """

    def __init__(self, max_vocab_size: int, max_len: int):
        self.max_vocab_size = max_vocab_size
        self.max_len = max_len
        self.model = self.rnn_net_set()

    def rnn_net_set(self):
        # 定义输入层
        inputs = Input(name="inputs", shape=[self.max_len])  # 数据输入的维度
        # 嵌入层（1 是因为词汇表中通常包含一个用于填充（padding）的特殊标记）
        layer = Embedding(self.max_vocab_size + 1, 128)(inputs)
        # LSTM 层（128 是 LSTM 单元的数量，dropout 是随机失活的概率，recurrent_dropout 是循环神经元的随机失活的概率）
        layer = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(layer)
        # 全连接层 1（ReLU 将所有负值设置为 0，保持正值不变）
        layer = Dense(128, activation="relu", name="FC1")(layer)
        # Dropout 层（正则化技术：0.5: Dropout 率。训练过程中会随机将 50% 的神经元设为 0（即丢弃），以防止过拟合，提高泛化能力）
        layer = Dropout(0.5)(layer)
        # 全连接层 2（Softmax 将输出向量转换为概率分布，每个类别的概率值在 0 到 1 之间，总和为 1）
        layer = Dense(2, activation="softmax", name="FC2")(layer)

        # 模型实例化
        rnn_net = tf.keras.models.Model(inputs=inputs, outputs=layer)
        # 打印模型摘要
        rnn_net.summary()
        # 编译模型
        rnn_net.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return rnn_net


class NetInference(object):
    """ 网络推理 """

    @classmethod
    def inference_batch(cls, rnn_net: tf.keras.models.Model, test_contents: list):
        """ 单数据推理 """
        # 设置随机 index
        random_index = random.randint(0, len(test_contents) - 1)
        # 读取测试数据，获取预测标签
        piece_of_random_data = test_contents[random_index]
        # 预测标签(由于测试数据没有 label，所以将数据进入前是行向量，但模型要求输入是列向量，所以需要将数据转置)
        pred_label = rnn_net.predict(piece_of_random_data.reshape(1, -1))
        # 计算预测标签（argmax：选择概率最大的类别作为预测结果，axis=1：沿着每行进行操作，选择最大值的索引，即类别标签）
        pred_y = np.argmax(pred_label, axis=1)  # 返回预测标签 [0] 或 [1]
        match pred_y:
            case _ if pred_y == [0]:
                print(f"测试集邮件预测结果：{green('正常邮件')}")
            case _ if pred_y == [1]:
                print(f"测试集邮件预测结果：{red('垃圾邮件')}")
            case _:
                print(f"测试集邮件预测结果：{yellow('未知邮件')}")


@timer
def main():
    # 读取停用词表
    stop_words_path = "data/stop_words.txt"
    stop_words = DataProcessor.stop_words_reader(stop_words_path)
    # print(f"停用词列表内容：\n{stop_words}")
    # print(f"停用词列表长度：{cpc.PrintColour.red(len(stop_words))}")
    # cui.draw_lines()

    # 读取添加词表
    add_words_path = "data/add_words_thulac.txt"
    add_words = DataProcessor.add_words_reader(add_words_path)
    # print(f"添加词列表内容：\n{add_words}")
    # print(f"添加词列表长度：{cpc.PrintColour.red(len(add_words))}")
    # cui.draw_lines()

    # 获取文件路径
    root_path = "data"
    file_name_list = DataProcessor.file_opener(root_path)

    # 标签补充
    range_num = None
    nor_labels, nor_contents, spam_labels, spam_contents, test_contents = DataProcessor.label_filler(
        file_name_list,
        range_num
    )
    # print(f"正常标签数量：{len(nor_labels)}")
    # print(f"垃圾标签数量：{len(spam_labels)}")
    # print(f"正常内容数量：{len(spam_contents)}")
    # print(f"垃圾内容数量：{len(spam_contents)}")

    # 实例化 thulac 分词器：除空格，繁体转简体
    segmentation = thulac.thulac(seg_only=True, rm_space=True, T2S=True)

    # 数据处理
    nor_contents, spam_contents, test_contents = DataProcessor.get_formal_data(
        segmentation,
        nor_contents,
        spam_contents,
        test_contents,
        stop_words
    )
    # print(f"正常邮件数量：{len(nor_contents)}")
    # print(f"垃圾邮件数量：{len(spam_contents)}")

    # 数据打包
    labels, contents = DataProcessor.data_packer(nor_labels, nor_contents, spam_labels, spam_contents)
    # print(f"评论内容：{contents[0]}")
    # print(f"标签长度：{len(labels)}")
    # print(f"内容长度：{len(contents)}")

    # 保存训练集词表
    pickle.dump(contents, open("data/words.pkl", "wb"))

    # 数据分类
    train_data, valid_data = DataProcessor.data_classifier(labels, contents)

    # 标签编码
    train_labels_eho, valid_labels_eho = DataEncoder.label_encoder(train_data, valid_data)

    # 内容编码
    vocab_size = 5000
    train_contents_seq, valid_contents_seq, test_contents_seq = DataEncoder.word_encoder(
        contents, vocab_size,
        train_data, valid_data,
        test_contents
    )

    # 内容等长
    max_len = 600  # 设定最大序列长度（维度）为 600
    train_contents_equal, valid_contents_equal, test_contents_equal = DataEncoder.word_filler(
        train_contents_seq, valid_contents_seq, test_contents_seq,
        max_len
    )

    # 网络加载
    net_path = "data/model.keras"
    email_rnn_net = tf.keras.models.load_model(net_path)

    # 网络推理
    NetInference.inference_batch(email_rnn_net, test_contents_equal)


if __name__ == "__main__":
    main()
