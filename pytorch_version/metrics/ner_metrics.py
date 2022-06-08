import torch
from collections import Counter
from processors.utils_ner import get_entities

from tools.common import init_logger, logger
import numpy as np

class SeqEntityScore(object):
    def __init__(self, id2label,markup='bios'):
        self.id2label = id2label
        self.markup = markup
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        # TP/(TP + FN)
        recall = 0 if origin == 0 else (right / origin)
        # TP/(TP + FP)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([x[0] for x in self.origins])
        found_counter = Counter([x[0] for x in self.founds])
        right_counter = Counter([x[0] for x in self.rights])
        # logger.info("**********origin_counter: {}**********".format(origin_counter))
        # logger.info("**********found_counter: {}**********".format(found_counter))
        # logger.info("**********right_counter: {}**********".format(right_counter))
        # # 06/01/2022 21:19:33 - INFO - root -   **********origin_counter: Counter({'Person': 9872, 'Book': 2408, 'Movie': 1971, 'Institute': 1595, 'Date': 1456, 'Song': 1115, 'Enterprise': 1085, 'School': 998, 'Country': 835, 'Text': 801, 'HistoricalPerson': 468, 'EntertainmentPerson': 456, 'Variety': 370, 'Number': 356, 'Location': 282, 'Award': 185, 'AdministrativeDistrict': 182, 'City': 114, 'Literature': 113, 'Album': 108, 'Climate': 99, 'Sight': 76, 'EnterpriseOrBrand': 64, 'Language': 9, 'Subject': 3})**********
        # # 06/01/2022 21:19:33 - INFO - root -   **********found_counter: Counter({'Person': 9226, 'Book': 2301, 'Movie': 2047, 'Date': 1564, 'Institute': 1447, 'School': 1039, 'Song': 1035, 'Enterprise': 871, 'Country': 776, 'Text': 637, 'EntertainmentPerson': 429, 'HistoricalPerson': 409, 'Number': 376, 'Variety': 295, 'AdministrativeDistrict': 223, 'Location': 170, 'Award': 110, 'Climate': 97, 'Literature': 73, 'EnterpriseOrBrand': 45, 'City': 38, 'Album': 36, 'Sight': 21})**********
        # # 06/01/2022 21:19:33 - INFO - root -   **********right_counter: Counter({'Person': 8183, 'Book': 2080, 'Movie': 1585, 'Date': 1327, 'Institute': 1166, 'Song': 851, 'School': 849, 'Country': 699, 'Enterprise': 693, 'Text': 509, 'HistoricalPerson': 306, 'Number': 287, 'EntertainmentPerson': 283, 'Variety': 220, 'AdministrativeDistrict': 140, 'Location': 131, 'Climate': 78, 'Award': 55, 'Literature': 34, 'City': 34, 'EnterpriseOrBrand': 31, 'Album': 29, 'Sight': 14})**********
        # 计算每个类的recall, precision, f1，存入class_info
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        # logger.info("**********origin: {}**********".format(origin))
        # logger.info("**********found: {}**********".format(found))
        # logger.info("**********right: {}**********".format(right))
        # # 06/01/2022 21:19:33 - INFO - root -   **********origin: 25021**********
        # # 06/01/2022 21:19:33 - INFO - root -   **********found: 23265**********
        # # 06/01/2022 21:19:33 - INFO - root -   **********right: 19584**********
        # 计算总的recall, precision, f1(micro f1)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'acc': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, label_paths, pred_paths):
        '''
        labels_paths: [[],[],[],....]
        pred_paths: [[],[],[],.....]

        :param label_paths:
        :param pred_paths:
        :return:
        Example:
            >>> labels_paths = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> pred_paths = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        '''
        for label_path, pre_path in zip(label_paths, pred_paths):
            # get_entities()函数将['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O']的形式转化为['MISC', 3, 5]的形式
            label_entities = get_entities(label_path, self.id2label,self.markup)
            pre_entities = get_entities(pre_path, self.id2label,self.markup)
            # 更新metric时，拿label_paths里的真实标签(get_entities()转换后的形式)，去扩展(extend)self.origins列表，得到的self.origins形如: [['Country', 0, 3], ['City', 88, 91], ['Movie', 46, 49]...
            # 更新metric时，拿pred_paths里的预测标签(get_entities()转换后的形式)，去扩展(extend)self.founds列表
            # extend()函数用于在列表末尾一次性追加另一个序列中的多个值(用新列表扩展原来的列表)
            
            # 当前句子的label_entities扩展self.origins，后续参与recall的计算
            self.origins.extend(label_entities)
            # 当前句子的pre_entities扩展self.founds，后续参与precision的计算
            self.founds.extend(pre_entities)
            # 当前句子中(每一句独立调用一次metric.update())，pre_entities中包含的预测标签，如果也在label_entities中，则是一个TP
            self.rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])
            # logger.info("**********self.origins: {}**********".format(self.origins))
            # logger.info("**********np.shape(self.origins): {}**********".format(np.shape(self.origins)))
            # logger.info("**********self.founds: {}**********".format(self.founds))
            # logger.info("**********np.shape(self.founds): {}**********".format(np.shape(self.founds)))
            # logger.info("**********self.rights: {}**********".format(self.rights))
            # logger.info("**********np.shape(self.rights): {}**********".format(np.shape(self.rights)))

class SpanEntityScore(object):
    def __init__(self, id2label):
        self.id2label = id2label
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([self.id2label[x[0]] for x in self.origins])
        found_counter = Counter([self.id2label[x[0]] for x in self.founds])
        right_counter = Counter([self.id2label[x[0]] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'acc': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, true_subject, pred_subject):
        self.origins.extend(true_subject)
        self.founds.extend(pred_subject)
        self.rights.extend([pre_entity for pre_entity in pred_subject if pre_entity in true_subject])



