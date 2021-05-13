#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 20.4.21
"""
import os
import sys

from multiprocessing.pool import Pool
from urllib import parse

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.project_utils import *
from myutils.cv_utils import *
from x_utils.vpf_utils import get_ocr_trt_dev_service
from root_dir import DATA_DIR


class EssayProcessor(object):
    def __init__(self):
        self.in_folder = os.path.join(DATA_DIR, 'essay')
        self.out_folder = os.path.join(DATA_DIR, 'essay-out')
        self.error_file = os.path.join(DATA_DIR, 'essay-error.txt')
        mkdir_if_not_exist(self.out_folder)
        self.url_format = "https://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/essay-library/" \
                          "datasets/20210420/essay/{}/{}/{}"

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
    def draw_res_box(img_bgr, boxes):
        img_bgr_x = copy.copy(img_bgr)
        for idx, box in enumerate(boxes):
            draw_box(img_bgr_x, box, color=(0, 0, 255), is_show=False, is_new=False)
            draw_text(img_bgr_x, str(idx+1), org=get_box_center(box), color=(0, 0, 255), thickness_x=3)
        return img_bgr_x

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
    def call_ocr_and_parse(url):
        """
        处理OCR结果
        """
        words_info = EssayProcessor.call_orc(url)
        box_list, word_list = [], []
        for word_info in words_info:
            word = word_info['word']
            pos = word_info["pos"]
            word_rec = EssayProcessor.parse_pos(pos)
            word_box = rec2box(word_rec)
            box_list.append(word_box)
            word_list.append(word)

        sorted_boxes, sorted_idxes, num_row = sorted_boxes_by_row(box_list)
        sorted_words = filter_list_by_idxes(word_list, sorted_idxes)
        sorted_boxes = unfold_nested_list(sorted_boxes)
        sorted_words = unfold_nested_list(sorted_words)

        return sorted_boxes, sorted_words

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

    @staticmethod
    def process_url(idx, img_url, out_dir, error_file):
        """
        处理URL
        """
        try:
            box_list, word_list = EssayProcessor.call_ocr_and_parse(img_url)
            _, img_bgr = download_url_img(img_url)
            img_out = EssayProcessor.draw_res_box(img_bgr, box_list)
            out_img_path, out_txt_path, ori_img_path = EssayProcessor.parse_out_path(img_url, out_dir)
            cv2.imwrite(ori_img_path, img_bgr)
            cv2.imwrite(out_img_path, img_out)
            write_list_to_file(out_txt_path, word_list)
            print('[Info] 处理完成: {} - {}'.format(idx, img_url))
            return len(word_list)
        except Exception as e:
            write_line(error_file, img_url)
            print('[Info] 处理失败: {} - {}'.format(idx, img_url))
            return 0

    def process(self):
        paths_list, names_list = traverse_dir_files(self.in_folder)
        print('[Info] 处理文件夹: {}'.format(self.in_folder))
        print('[Info] 文件数: {}'.format(len(paths_list)))

        url_list = []
        for path, name in zip(paths_list, names_list):
            # print('[Info] path: {}'.format(path))
            items = path.split('/')
            img_name = items[-1]
            clz_name = items[-2]
            book_name = parse.quote(items[-3])

            img_url = self.url_format.format(book_name, clz_name, img_name)
            url_list.append(img_url)
        print('[Info] img_url num: {}'.format(len(url_list)))

        pool = Pool(processes=5)

        for idx, img_url in enumerate(url_list):
            if idx == 20:
                break
            n_word = EssayProcessor.process_url(idx, img_url, self.out_folder, self.error_file)
            # pool.apply_async(EssayProcessor.process_url, (idx, img_url, self.out_folder, self.error_file))

        pool.close()
        pool.join()
        print('[Info] 全部处理完成: {}'.format(self.out_folder))


def main():
    ep = EssayProcessor()
    ep.process()


if __name__ == '__main__':
    main()
