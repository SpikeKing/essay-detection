#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 26.4.21
"""

import os
import sys
from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.cv_utils import *
from myutils.project_utils import *
from root_dir import DATA_DIR


class DatasetGenerator(object):
    """
    数据集生成
    """

    def __init__(self):
        pass

    @staticmethod
    def get_file_url_format():
        name_format_dict = {
            "grade_4_essay_rotated": "https://sm-transfer.oss-cn-hangzhou.aliyuncs.com/"
                                     "zhengsheng.wcl/essay-library/datasets/20210416/grade_4_essay_rotated/{}.jpg",
            "grade_5_essay_rotated": "https://sm-transfer.oss-cn-hangzhou.aliyuncs.com/"
                                     "zhengsheng.wcl/essay-library/datasets/20210416/grade_5_essay_rotated/{}.jpg",
            "grade_6_essay_rotated": "https://sm-transfer.oss-cn-hangzhou.aliyuncs.com/"
                                     "zhengsheng.wcl/essay-library/datasets/20210416/grade_6_essay_rotated/{}.jpg",
            "5-3语文-初中同步作文": "https://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/essay-library/"
                            "datasets/20210426/essay_prelabel_x/"
                            "5-3%E8%AF%AD%E6%96%87-%E5%88%9D%E4%B8%AD%E5%90%8C%E6%AD%A5%E4%BD%9C%E6%96%87/{}.jpg",
            "小学生开心同步作文下册": "https://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/essay-library/"
                           "datasets/20210426/essay_prelabel_x/"
                           "%E5%B0%8F%E5%AD%A6%E7%94%9F%E5%BC%80%E5%BF%83%E5%90%8C%E6%AD%A5%E4%BD%9C%E6%96%87%E4%B8%8B%E5%86%8C/{}.jpg",
            "小学语文同步作文下册": "https://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/essay-library/"
                          "datasets/20210426/essay_prelabel_x/"
                          "%E5%90%8C%E6%AD%A5%E4%BD%9C%E6%96%87/{}.jpg"

        }
        return name_format_dict

    @staticmethod
    def convert(iw, ih, box):
        """
        将标注的xml文件标注转换为darknet形的坐标
        """
        iw = float(iw)
        ih = float(ih)
        dw = 1. / iw
        dh = 1. / ih
        x = (box[0] + box[2]) / 2.0 - 1
        y = (box[1] + box[3]) / 2.0 - 1
        w = box[2] - box[0]
        h = box[3] - box[1]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h

    @staticmethod
    def generate_file(file_path, file_name, file_idx):
        name_format_dict = DatasetGenerator.get_file_url_format()
        file_idx = str(file_idx).zfill(4)
        print('[Info] file_path: {}, file_name: {}, file_idx: {}'.format(file_path, file_name, file_idx))

        file_name_x = file_name.split('.')[0]
        url_format = name_format_dict[file_name_x]

        out_dataset_dir = os.path.join(DATA_DIR, 'essay_ds_v1_1')

        out_images_dir = os.path.join(out_dataset_dir, 'images')
        out_images_train_dir = os.path.join(out_images_dir, 'train')
        out_images_val_dir = os.path.join(out_images_dir, 'val')

        out_labels_dir = os.path.join(out_dataset_dir, 'labels')
        out_labels_train_dir = os.path.join(out_labels_dir, 'train')
        out_labels_val_dir = os.path.join(out_labels_dir, 'val')

        mkdir_if_not_exist(out_dataset_dir)
        mkdir_if_not_exist(out_images_dir)
        mkdir_if_not_exist(out_images_train_dir)
        mkdir_if_not_exist(out_images_val_dir)
        mkdir_if_not_exist(out_labels_dir)
        mkdir_if_not_exist(out_labels_train_dir)
        mkdir_if_not_exist(out_labels_val_dir)

        print('[Info] 处理数据开始: {}'.format(file_path))
        data_line = read_file(file_path)[0]
        data_dict = json.loads(data_line)
        print('[Info] keys: {}'.format(data_dict.keys()))
        images = data_dict['images']

        id_name_dict = {}
        for idx, img in enumerate(images):
            img_id = img['id']
            image_name = img['file_name'].split('.')[0]
            height = img['height']
            width = img['width']

            # print('[Info] img: {}'.format(img))
            # print('[Info] img_id: {}, file_name: {}'.format(img_id, image_name))
            id_name_dict[img_id] = [image_name, height, width]
            # if idx == 20:
            #     break

        annotations = data_dict["annotations"]

        image_dict = collections.defaultdict(list)
        for idx, anno in enumerate(annotations):
            image_id = anno['image_id']
            image_name, ih, iw = id_name_dict[image_id]
            wh_box = anno['bbox']
            bbox = [wh_box[0], wh_box[1], wh_box[0] + wh_box[2], wh_box[1] + wh_box[3]]
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            bbox_yolo = DatasetGenerator.convert(iw, ih, bbox)
            bbox_yolo = [str(round(i, 6)) for i in bbox_yolo]
            # print('[Info] image_id: {}, ih: {}, iw: {}, bbox: {}, bbox_yolo: {}'
            #       .format(image_name, ih, iw, bbox, bbox_yolo))

            image_dict[image_name].append(" ".join(["0", *bbox_yolo]))

        print('[Info] 样本数: {}'.format(len(image_dict.keys())))

        image_name_list = list(image_dict.keys())
        gap = len(image_name_list) // 10
        image_train_list = image_name_list[:gap * 9]
        image_val_list = image_name_list[gap * 9:]
        print('[Info] 训练: {}, 验证: {}'.format(len(image_train_list), len(image_val_list)))

        for idx, image_name in enumerate(image_train_list):
            print('[Info] idx: {}'.format(idx))
            bbox_yolo_list = image_dict[image_name]

            image_url = url_format.format(image_name)
            is_ok, img_bgr = download_url_img(image_url)

            out_name = "train_{}_{}".format(file_idx, str(idx).zfill(6))
            img_path = os.path.join(out_images_train_dir, '{}.jpg'.format(out_name))
            cv2.imwrite(img_path, img_bgr)  # 写入图像
            print('[Info] img_path: {}'.format(img_path))

            lbl_path = os.path.join(out_labels_train_dir, '{}.txt'.format(out_name))
            write_list_to_file(lbl_path, bbox_yolo_list)
            print('[Info] lbl_path: {}'.format(lbl_path))

            print('[Info] ' + "-" * 100)
            # if idx == 20:
            #     break

        for idx, image_name in enumerate(image_val_list):
            print('[Info] idx: {}'.format(idx))
            bbox_yolo_list = image_dict[image_name]

            image_url = url_format.format(image_name)
            is_ok, img_bgr = download_url_img(image_url)

            out_name = "val_{}_{}".format(file_idx, str(idx).zfill(6))
            img_path = os.path.join(out_images_val_dir, '{}.jpg'.format(out_name))
            cv2.imwrite(img_path, img_bgr)  # 写入图像
            print('[Info] img_path: {}'.format(img_path))

            lbl_path = os.path.join(out_labels_val_dir, '{}.txt'.format(out_name))
            write_list_to_file(lbl_path, bbox_yolo_list)
            print('[Info] lbl_path: {}'.format(lbl_path))

            print('[Info] ' + "-" * 100)
            # if idx == 20:
            #     break
        print('[Info] 处理完成! {}'.format(file_path))

    @staticmethod
    def check_darknet_data(img_bgr, data_line):
        ih, iw, _ = img_bgr.shape
        items = data_line.split(" ")
        items = [float(i) for i in items]
        x, y, w, h = items[1:]
        x, y, w, h = x * iw, y * ih, w * iw, h * ih
        x_min, x_max = x - w // 2, x + w // 2
        y_min, y_max = y - h // 2, y + h // 2
        img_out = draw_box(img_bgr, [x_min, y_min, x_max, y_max], is_show=False)
        return img_out

    def check_dataset(self, check_dir, out_dir):
        images_dir = os.path.join(check_dir, 'images', 'train')
        labels_dir = os.path.join(check_dir, 'labels', 'train')
        mkdir_if_not_exist(out_dir)

        n_check = 20

        paths_list, names_list = traverse_dir_files(images_dir)
        paths_list, names_list = shuffle_two_list(paths_list, names_list)
        paths_list, names_list = paths_list[:n_check], names_list[:n_check]
        print('[Info] 检查样本数: {}'.format(len(paths_list)))

        for path, name in zip(paths_list, names_list):
            label_name = name.split('.')[0] + ".txt"
            img_bgr = cv2.imread(path)
            label_path = os.path.join(labels_dir, label_name)
            data_lines = read_file(label_path)
            for idx, data_line in enumerate(data_lines):
                img_bgr = self.check_darknet_data(img_bgr, data_line)

            out_path = os.path.join(out_dir, name)
            cv2.imwrite(out_path, img_bgr)


def process():
    dir_path = os.path.join(DATA_DIR, 'essay_ds_v1_1_json')
    paths_list, names_list = traverse_dir_files(dir_path)

    pool = Pool(processes=80)

    for file_idx, (path, name) in enumerate(zip(paths_list, names_list)):
        DatasetGenerator.generate_file(path, name, file_idx)
        print('[Info] path: {}'.format(path))
        # pool.apply_async(DatasetGeneratorV2.generate_file, (path, file_idx))
    pool.close()
    pool.join()
    print('[Info] 全部处理完成: {}'.format(dir_path))


def check():
    check_dir = os.path.join(DATA_DIR, 'essay_ds_v1')
    out_dir = os.path.join(DATA_DIR, 'essay_checked_{}'.format(get_current_time_str()))
    dg = DatasetGenerator()
    dg.check_dataset(check_dir, out_dir)


def main():
    process()
    # check()


if __name__ == '__main__':
    main()
