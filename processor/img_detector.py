#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 28.12.20
"""

import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

from myutils.cv_utils import *
from myutils.project_utils import *
from root_dir import DATA_DIR


class ImgDetector(object):
    """
    图像检测
    """
    def __init__(self):
        self.weights = os.path.join(DATA_DIR, 'models', 'essay_20210510.1.pt')

        self.img_size = 640
        self.conf_thres = 0.3
        self.iou_thres = 0.6
        self.nc = 1

        self.device = select_device()  # 自动选择环境
        print('[Info] device: {}'.format(self.device))
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model, self.img_size = self.load_model()  # 加载模型

    def load_model(self):
        """
        加载模型
        """
        # Load model
        model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        img_size = check_img_size(self.img_size, s=gs)  # check img_size

        if self.half:
            model.half()
        model.eval()

        if self.device.type != 'cpu':
            model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(model.parameters())))

        return model, img_size

    def load_image_and_resize(self, img_bgr):
        """
        加载resize图像
        """
        h0, w0 = img_bgr.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            img_bgr = cv2.resize(img_bgr, (int(w0 * r), int(h0 * r)),
                                 interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
        return img_bgr, (h0, w0), img_bgr.shape[:2]  # img, hw_original, hw_resized

    def preprocess_data(self, img_bgr):
        """
        图像预处理
        """
        img_bgr, (h0, w0), (h, w) = self.load_image_and_resize(img_bgr)

        lb_shape = (self.img_size, self.img_size)
        img_bgr, ratio, pad = letterbox(img_bgr, new_shape=lb_shape, auto=False, scaleup=False)

        # Convert
        img_bgr = img_bgr[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img_bgr = np.ascontiguousarray(img_bgr)

        img = torch.from_numpy(img_bgr)

        img = img.to(self.device, non_blocking=True)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        x_shape = (h0, w0), ((h / h0, w / w0), pad)

        return img, x_shape

    def detect_problems(self, img_bgr):
        """
        检测逻辑
        """
        img, x_shape = self.preprocess_data(img_bgr)  # 预处理数据

        with torch.no_grad():
            pred, train_out = self.model(img, augment=False)  # inference and training outputs

        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, labels=[], multi_label=True, agnostic=False)  # NMS后处理

        box_list = []  # 最终输出
        for i, det in enumerate(pred):  # detections per image
            # 回复图像尺寸
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], x_shape[0], x_shape[1]).round()
            det = det.tolist()

            for *xyxy, conf, cls in reversed(det):  # 绘制图像
                xyxy = [int(i) for i in xyxy]
                box_list.append(xyxy)

        return box_list


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
        for j in range(i+1, n_box):
            box1 = sorted_boxes[i]
            box2 = sorted_boxes[j]
            r_iou = min_iou(box1, box2)
            if r_iou > r_thr:
                flag_list[j] = False
                x_boxes.append(box2)
        print('[Info] i: {}, x_boxes: {}'.format(i, x_boxes))
        sorted_boxes[i] = merge_boxes(x_boxes)

    new_boxes = []
    for i in range(n_box):
        if flag_list[i]:
            new_boxes.append(sorted_boxes[i])

    return new_boxes


def predict_one_img():
    name_x = "val_0004_000030.jpg"
    out_name_x = name_x.split('.')[0] + ".out.jpg"
    img_path = os.path.join(DATA_DIR, 'imgs', name_x)
    out_path = os.path.join(DATA_DIR, 'imgs', out_name_x)
    img_bgr = cv2.imread(img_path)
    img_detector = ImgDetector()
    box_list = img_detector.detect_problems(img_bgr)
    # box_list = filer_boxes_by_size(box_list)
    draw_box_list(img_bgr, box_list, is_show=True, save_name=out_path)


def predict_imgs_file():
    file_path = os.path.join(DATA_DIR, 'xxx.txt')
    urls = read_file(file_path)
    all_box, n_sample = 0, 0
    img_detector = ImgDetector()

    for idx, url in enumerate(urls):
        if idx == 20:
            break
        try:
            _, img_bgr = download_url_img(url)
        except:
            continue
        box_list = img_detector.detect_problems(img_bgr)
        all_box += len(box_list)
        print('[Info] n box: {}'.format(len(box_list)))
        if box_list != 0:
            n_sample += 1

    n_avg = safe_div(all_box, n_sample)
    print('[Info] n_avg: {}'.format(n_avg))


def main():
    predict_one_img()
    # predict_imgs_file()


if __name__ == '__main__':
    main()
