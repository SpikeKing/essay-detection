#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 29.4.21
"""

import base64

import requests

from myutils.cv_utils import *
from myutils.project_utils import *


class SampleTuner(object):
    def __init__(self):
        pass

    @staticmethod
    def image_to_base64(image_np):
        """
        图像转换为base64
        """
        image = cv2.imencode('.jpg', image_np)[1]
        image_code = str(base64.b64encode(image))[2:-1]
        return image_code

    @staticmethod
    def call_vpf_service(image_encode, service_name="qcMYzowHcmRB4Zx3QK9hGP", timeout=10000):
        """
        调研VPF服务
        """
        api = 'http://vpf-api-service-online-shenma.alibaba-inc.com/api'
        data = dict()
        data['service_name'] = service_name
        data['image_encode'] = image_encode
        # data['image_url'] = image_url
        # data['image_content'] = image_content

        data = json.dumps(data)
        r = requests.post(api, data, timeout=timeout)
        code = r.status_code
        if code != 200:
            return None
        output_data = json.loads(r.text)
        data_dict = output_data['data']
        return data_dict

    def get_img_ocr_data(self, image_np):
        """
        获取图像的OCR数据
        """
        image_code = self.image_to_base64(image_np)
        data_dict = self.call_vpf_service(image_code)
        return data_dict

    @staticmethod
    def cal_angle(words_info):
        """
        根据words_info计算图像的角度
        """
        n_words = len(words_info)
        if n_words == 0:
            return 0  # 计算角度

        sum_angle, n_txt = 0, 0
        for word_info in words_info:
            if safe_div(word_info['width'], word_info['height']) < 1.5:  # 避免旋转方形文字
                continue

            if 'angle' in word_info.keys() and 'word' in word_info.keys():
                word_txt = word_info['word']
                n_word_txt = len(word_txt)

                if abs(word_info['angle']) > 35:  # 去掉异常的角度cat
                    continue

                n_txt += n_word_txt
                x_angle = word_info['angle'] * n_word_txt
                sum_angle += x_angle

        if n_txt != 0:
            avg_angle = sum_angle // n_txt
        else:
            avg_angle = 0
        return int(avg_angle)

    @staticmethod
    def parse_pos(pos_list):
        """
        处理点
        """
        point_list = []
        for pos in pos_list:
            x = pos['x']
            y = pos['y']
            point_list.append([x, y])
        return point_list

    @staticmethod
    def warp_point(x, y, M):
        Q = np.dot(M, np.array(((x,), (y,), (1,))))
        return int(Q[0][0]), int(Q[1][0])

    @staticmethod
    def warp_rec_list(rec_list, M):
        new_boxes = []
        for points in rec_list:
            new_rec = []
            for pnt in points:
                x, y = pnt
                x, y = SampleTuner.warp_point(x, y, M)
                new_rec.append([x, y])
            new_box = rec2box(new_rec)
            new_boxes.append(new_box)
        return new_boxes

    @staticmethod
    def warp_box_list(box_list, M):
        new_boxes = []
        for box in box_list:
            nb0, nb1 = SampleTuner.warp_point(box[0], box[1], M)
            nb2, nb3 = SampleTuner.warp_point(box[2], box[3], M)
            new_box = [nb0, nb1, nb2, nb3]
            new_boxes.append(new_box)
        return new_boxes

    @staticmethod
    def parse_label_box(img_bgr, data_line):
        ih, iw, _ = img_bgr.shape
        items = data_line.split(" ")
        items = [float(i) for i in items]
        x, y, w, h = items[1:]
        x, y, w, h = x * iw, y * ih, w * iw, h * ih
        x_min, x_max = x - w // 2, x + w // 2
        y_min, y_max = y - h // 2, y + h // 2
        box = [int(x) for x in [x_min, y_min, x_max, y_max]]
        return box

    def parse_label(self, img_bgr, label_lines):
        box_list = []
        for idx, data_line in enumerate(label_lines):
            box = self.parse_label_box(img_bgr, data_line)
            box_list.append(box)
        return box_list

    def parse_img(self, img_bgr):
        data_dict = self.get_img_ocr_data(img_bgr)
        words_info = data_dict['data']['wordsInfo']
        angle = self.cal_angle(words_info)
        print('[Info] angle: {}'.format(angle))
        img_rotated, M = rotate_img_with_bound(img_bgr, angle)
        show_img_bgr(img_rotated)

        rec_list = []
        for word_info in words_info:
            pos = word_info['pos']
            word_rec = self.parse_pos(pos)
            rec_list.append(word_rec)

        word_boxes = self.warp_rec_list(rec_list, M)
        draw_box_list(img_rotated, word_boxes, is_show=True)
        return angle, M, word_boxes, img_rotated

    def parse_img_path(self, img_path):
        img_bgr = cv2.imread(img_path)
        self.parse_img(img_bgr)

    def parse_img_url(self, img_url):
        _, img_bgr = download_url_img(img_url)
        angle, M, word_boxes, img_rotated = self.parse_img(img_bgr)
        return angle, M, word_boxes, img_rotated

    def parse_label_path(self, img_path, label_path):
        img_bgr = cv2.imread(img_path)
        data_lines = read_file(label_path)
        box_list = self.parse_label(img_bgr, data_lines)
        return box_list

    def parse_label_url(self, img_url, label_url):
        _, img_bgr = download_url_img(img_url)
        _, data_lines = download_url_txt(label_url, is_split=True)
        box_list = self.parse_label(img_bgr, data_lines)
        return box_list

    def process(self):
        # img_path = os.path.join(DATA_DIR, 'essay_ds_v1/images/train', 'train_0000_000000.jpg')
        # label_path = os.path.join(DATA_DIR, 'essay_ds_v1/labels/train', 'train_0000_000000.txt')
        img_url = "https://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/essay-library/datasets/20210429/essay_ds_v1_2/images/val/val_0005_000054.jpg"
        label_url = "https://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/essay-library/datasets/20210429/essay_ds_v1_2/labels/val/val_0005_000054.txt"

        _, img_bgr = download_url_img(img_url)
        _, data_lines = download_url_txt(label_url, is_split=True)

        # self.parse_img_path(img_path)
        # box_list = self.parse_label_path(img_path, label_path)
        angle, M, word_boxes, img_rotated = self.parse_img(img_bgr)
        box_list = self.parse_label_url(img_url, label_url)

        new_box_list = self.warp_box_list(box_list, M)

        idx_boxes_dict = collections.defaultdict(list)
        for word_box in word_boxes:
            iou_list = []
            for seg_box in new_box_list:
                v_iou = min_iou(word_box, seg_box)
                iou_list.append(v_iou)
            max_idx = iou_list.index(max(iou_list))
            idx_boxes_dict[max_idx].append(word_box)

        new_seg_boxes = []
        for boxes in idx_boxes_dict.values():
            x_box = merge_boxes(boxes)
            new_seg_boxes.append(x_box)

        new_seg_boxes, _ = filer_boxes_by_size(new_seg_boxes)
        # img_bgr = cv2.imread(img_path)
        _, img_bgr = download_url_img(img_url)

        draw_box_list(img_rotated, new_seg_boxes, is_show=True)


def main():
    st = SampleTuner()
    st.process()


if __name__ == '__main__':
    main()
