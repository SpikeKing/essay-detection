#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 11.5.21
"""

import copy
import os
import sys
from urllib import parse

import cv2

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from processor.img_detector import ImgDetector
from myutils.project_utils import *
from myutils.cv_utils import min_iou, merge_boxes, rec2box, sorted_boxes_by_row, draw_box, draw_text, get_box_center, \
    get_box_size
from x_utils.vpf_utils import get_ocr_trt_dev_service
from root_dir import DATA_DIR, ROOT_DIR


class EssayGeneratorV2(object):
    def __init__(self):
        self.in_folder = os.path.join(ROOT_DIR, '..', 'datasets', 'essay_zip_files_v2_20210513')
        # self.in_folder = os.path.join(DATA_DIR, 'essay')
        self.out_folder = os.path.join(DATA_DIR, 'essay-out')
        self.error_file = os.path.join(DATA_DIR, 'essay-error.txt')
        mkdir_if_not_exist(self.out_folder)
        self.url_format = "https://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/essay-library/" \
                          "datasets/20210513/essay_zip_files_v2_20210513/{}/{}/{}"
        # self.url_format = "https://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/essay-library/" \
        #                   "datasets/20210420/essay/{}/{}/{}"
        self.img_detector = ImgDetector()

    @staticmethod
    def filer_boxes_by_size(boxes, r_thr=0.5):
        """
        根据是否重叠过滤包含在内部的框
        """
        if not boxes:
            return boxes

        size_list = []
        idx_list = []
        for idx, box in enumerate(boxes):
            size_list.append(get_box_size(box))
            idx_list.append(idx)

        size_list, sorted_idxes, sorted_boxes = \
            sort_three_list(size_list, idx_list, boxes, reverse=True)

        n_box = len(sorted_boxes)  # box的数量
        flag_list = [True] * n_box

        for i in range(n_box):
            if not flag_list[i]:
                continue
            x_boxes = [sorted_boxes[i]]
            for j in range(i + 1, n_box):
                box1 = sorted_boxes[i]
                box2 = sorted_boxes[j]
                r_iou = min_iou(box1, box2)
                if r_iou > r_thr:
                    flag_list[j] = False
                    x_boxes.append(box2)
            # print('[Info] i: {}, x_boxes: {}'.format(i, x_boxes))
            sorted_boxes[i] = merge_boxes(x_boxes)

        new_boxes = []
        for i in range(n_box):
            if flag_list[i]:
                new_boxes.append(sorted_boxes[i])

        return new_boxes

    @staticmethod
    def call_orc(url):
        """
        调用OCR服务
        img_url:
        https://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/essay-library/datasets/20210420/essay/
        5-3%E8%AF%AD%E6%96%87-%E5%88%9D%E4%B8%AD%E5%90%8C%E6%AD%A5%E4%BD%9C%E6%96%87-%E4%B8%83%E5%B9%B4%E7%BA%A7/
        bookcover/page-0001.jpg
        """
        res_dict = get_ocr_trt_dev_service(img_url=url)
        ocr_data = res_dict['data']['data']
        words_info = ocr_data['wordsInfo']
        return words_info

    @staticmethod
    def parse_pos(pos_list):
        """
        处理OCR结果的点
        """
        point_list = []
        for pos in pos_list:
            x = pos['x']
            y = pos['y']
            point_list.append([x, y])
        return point_list

    @staticmethod
    def call_ocr_and_parse(url):
        """
        处理OCR结果
        """
        words_info = EssayGeneratorV2.call_orc(url)
        box_list, word_list = [], []
        for word_info in words_info:
            word = word_info['word']
            pos = word_info["pos"]
            word_rec = EssayGeneratorV2.parse_pos(pos)
            word_box = rec2box(word_rec)
            box_list.append(word_box)
            word_list.append(word)

        sorted_boxes, sorted_idxes, num_row = sorted_boxes_by_row(box_list)
        sorted_words = filter_list_by_idxes(word_list, sorted_idxes)
        sorted_boxes = unfold_nested_list(sorted_boxes)
        sorted_words = unfold_nested_list(sorted_words)
        return sorted_boxes, sorted_words

    @staticmethod
    def merge_texts(text_list):
        s_text = ""
        for txt in text_list:
            s_text += txt + "\n"
        s_text = s_text.strip()
        s_text += "\n\n"
        return s_text

    def segment_img(self, img_url):
        print('[Info] url: {}'.format(img_url))
        _, img_bgr = download_url_img(img_url)
        seg_boxes = self.img_detector.detect_problems(img_bgr)
        seg_boxes = EssayGeneratorV2.filer_boxes_by_size(seg_boxes)

        # 排序
        seg_boxes, _, _ = sorted_boxes_by_row(seg_boxes)
        seg_boxes = unfold_nested_list(seg_boxes)
        # draw_box_list(img_bgr, seg_boxes, is_show=True, is_new=True)
        print('[Info] 段数: {}'.format(len(seg_boxes)))

        # 识别OCR
        word_boxes, word_texts = EssayGeneratorV2.call_ocr_and_parse(img_url)
        # draw_box_list(img_bgr, word_boxes, is_show=True, is_new=True)

        seg_boxes_dict = collections.defaultdict(list)
        seg_texts_dict = collections.defaultdict(list)
        other_boxes, other_texts = list(), list()

        for word_box, word_text in zip(word_boxes, word_texts):
            is_seg = False
            for idx, seg_box in enumerate(seg_boxes):
                iou = min_iou(word_box, seg_box)
                if iou >= 0.5:
                    is_seg = True
                    seg_boxes_dict[idx].append(word_box)
                    seg_texts_dict[idx].append(word_text)
                    break
            if not is_seg:
                other_boxes.append(word_box)
                other_texts.append(word_text)

        seg_merged_boxes, seg_merged_texts = [], []
        for idx in range(len(seg_boxes)):
            if idx not in seg_boxes_dict.keys():
                continue
            tmp_boxes = seg_boxes_dict[idx]
            tmp_texts = seg_texts_dict[idx]
            sorted_boxes, sorted_idxes, _ = sorted_boxes_by_row(tmp_boxes)

            sorted_boxes = unfold_nested_list(sorted_boxes)
            sorted_idxes = unfold_nested_list(sorted_idxes)
            sorted_texts = filter_list_by_idxes(tmp_texts, sorted_idxes)

            # large_box = merge_boxes(sorted_boxes)
            # seg_merged_boxes.append(large_box)
            seg_merged_boxes.append(seg_boxes[idx])  # 直接使用原始的box
            seg_merged_texts.append(EssayGeneratorV2.merge_texts(sorted_texts))

        # draw_box_list(img_bgr, seg_merged_boxes, is_show=True, is_new=True)

        res_boxes = seg_merged_boxes + other_boxes
        res_texts = seg_merged_texts + other_texts
        res_boxes, res_idxes, _ = sorted_boxes_by_row(res_boxes)
        res_boxes = unfold_nested_list(res_boxes)
        res_idxes = unfold_nested_list(res_idxes)

        res_texts = filter_list_by_idxes(res_texts, res_idxes)
        return res_boxes, res_texts, img_bgr

    @staticmethod
    def draw_res_box(img_bgr, boxes):
        img_bgr_x = copy.copy(img_bgr)
        for idx, box in enumerate(boxes):
            draw_box(img_bgr_x, box, color=(0, 0, 255), is_show=False, is_new=False)
            draw_text(img_bgr_x, str(idx+1), org=get_box_center(box), color=(0, 0, 255), thickness_x=3)
        return img_bgr_x

    @staticmethod
    def parse_out_path(img_url, out_dir):
        """
        生成输出路径
        """
        items = img_url.split('/')
        img_name = items[-1]
        clz_name = items[-2]
        book_name = parse.unquote(items[-3])
        book_dir = os.path.join(out_dir, book_name)
        mkdir_if_not_exist(book_dir)
        clz_dir = os.path.join(book_dir, clz_name)
        mkdir_if_not_exist(clz_dir)

        name_x = img_name.split('.')[0]

        out_img_path = os.path.join(clz_dir, name_x + ".out.jpg")
        out_txt_path = os.path.join(clz_dir, name_x + ".out.txt")
        ori_img_path = os.path.join(clz_dir, name_x + ".jpg")
        return out_img_path, out_txt_path, ori_img_path

    def process_url(self, idx, img_url, out_dir, error_file):
        """
        处理URL
        """

        try:
            box_list, word_list, img_bgr = self.segment_img(img_url)
            img_out = EssayGeneratorV2.draw_res_box(img_bgr, box_list)
            out_img_path, out_txt_path, ori_img_path = EssayGeneratorV2.parse_out_path(img_url, out_dir)
            cv2.imwrite(ori_img_path, img_bgr)  # 原始图像
            cv2.imwrite(out_img_path, img_out)  # 输出图像
            create_file(out_txt_path)  # 输出文本
            write_list_to_file(out_txt_path, word_list)
            print('[Info] 处理完成: {} - {}'.format(idx, img_url))
        except Exception as e:
            write_line(error_file, img_url)
            print('[Info] 处理失败: {} - {}'.format(idx, img_url))

    def get_processed_info(self, out_folder):
        paths_list, names_list = traverse_dir_files(out_folder)
        processed_urls = []
        for path, name in zip(paths_list, names_list):
            # print('[Info] path: {}'.format(path))
            items = path.split('/')
            img_name = items[-1]
            clz_name = items[-2]
            book_name = parse.quote(items[-3])
            img_url = self.url_format.format(book_name, clz_name, img_name)
            processed_urls.append(img_url)
        return processed_urls

    def process(self):
        paths_list, names_list = traverse_dir_files(self.in_folder)
        print('[Info] 处理文件夹: {}'.format(self.in_folder))
        print('[Info] 文件数: {}'.format(len(paths_list)))

        processed_urls = self.get_processed_info(self.out_folder)
        print('[Info] 已处理: {}'.format(len(processed_urls)))

        url_list = []
        for path, name in zip(paths_list, names_list):
            # print('[Info] path: {}'.format(path))
            items = path.split('/')
            img_name = items[-1]
            clz_name = items[-2]
            if '高考' not in items[-3]:
                continue
            book_name = parse.quote(items[-3])

            img_url = self.url_format.format(book_name, clz_name, img_name)
            if img_url in processed_urls:
                continue
            url_list.append(img_url)

        print('[Info] img_url num: {}'.format(len(url_list)))

        # pool = Pool(processes=5)

        for idx, img_url in enumerate(url_list):
            # if idx == 20:
            #     break
            print('[Info] ' + '-' * 50)
            self.process_url(idx, img_url, self.out_folder, self.error_file)
            # pool.apply_async(EssayGeneratorV2.process_url, (idx, img_url, self.out_folder, self.error_file))

        # pool.close()
        # pool.join()
        print('[Info] 全部处理完成: {}'.format(self.out_folder))


def main():
    eg2 = EssayGeneratorV2()
    eg2.process()


if __name__ == '__main__':
    main()
