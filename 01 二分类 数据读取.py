import chardet
import os
import random
import thulac
import tqdm

from utils import timer


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
        raw_test_labels = []
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
        print(f"正常邮件标签：{raw_nor_labels}")
        print(f"正常邮件内容：{raw_nor_contents}")
        print(f"垃圾邮件标签：{raw_spam_labels}")
        print(f"垃圾邮件内容：{raw_spam_contents}")
        print(f"测试邮件标签：{raw_test_labels}")
        print(f"测试邮件内容：{raw_test_contents}")
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
        print(f"正常邮件数据：{nor_clean_data}")
        print(f"垃圾邮件数据：{spam_clean_data}")
        print(f"测试邮件数据：{test_clean_data}")
        return nor_clean_data, spam_clean_data, test_clean_data

    @classmethod
    def data_packer(
            cls,
            nor_labels: list, nor_contents: list,
            spam_labels: list, spam_contents: list
    ) -> list[tuple[int, str]]:
        """ 数据打包 """
        nor_package = [(nor_labels[i], nor_contents[i]) for i in range(len(nor_labels))]
        spam_package = [(spam_labels[i], spam_contents[i]) for i in range(len(spam_labels))]
        package = nor_package + spam_package
        # print(f"数据打包：{package}")
        # print(f"数据总数：{cpc.PrintColour.red(len(package))}")
        random.shuffle(package)
        # print(f"打乱数据：{package}")
        # print(f"打乱后数据总数：{cpc.PrintColour.red(len(package))}")
        return package


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
    range_num = 5
    nor_labels, nor_contents, spam_labels, spam_contents, test_contents = DataProcessor.label_filler(
        file_name_list,
        range_num
    )

    # 实例化 thulac 分词器：除空格，繁体转简体
    segmentation = thulac.thulac(seg_only=True, rm_space=True, T2S=True)

    # # 数据处理
    # nor_contents, spam_contents, test_contents = DataProcessor.get_formal_data(
    #     segmentation,
    #     nor_contents,
    #     spam_contents,
    #     test_contents,
    #     stop_words
    # )
    #
    # # 数据打包
    # DataProcessor.data_packer(nor_labels, nor_contents, spam_labels, spam_contents)


if __name__ == "__main__":
    main()
