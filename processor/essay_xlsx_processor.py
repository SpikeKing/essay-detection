#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 12.5.21
"""

import os
import sys

from myutils.project_utils import *
from root_dir import DATA_DIR


class EssayXlsxProcessor(object):
    def __init__(self):
        self.file_path = os.path.join(DATA_DIR, "自采买作文影印文件0511.xlsx")
        self.out_download_file = os.path.join(DATA_DIR, '自采买作文影印文件0511.download.txt')
        self.out_unzip_file = os.path.join(DATA_DIR, '自采买作文影印文件0511.unzip.txt')
        create_file(self.out_download_file)
        create_file(self.out_unzip_file)

    def process(self):
        data_lines = read_excel_file(self.file_path)

        out_download_lines, out_unzip_lines = [], []  # 下载数据列表
        for idx, data_line in enumerate(data_lines):
            if idx == 0:
                continue
            url = data_line[9]
            name = data_line[5].replace('/', '_')
            print('[Info] url: {}'.format(url))
            print('[Info] name: {}'.format(name))
            out_download_line = "wget -O {} {}".format(name+".zip", url)
            out_unzip_line = "unzip {} -d {}".format(name+".zip", name)
            out_download_lines.append(out_download_line)
            out_unzip_lines.append(out_unzip_line)

        print('[Info] 文件数: {}'.format(len(out_download_lines)))
        print('[Info] 文件数: {}'.format(len(out_unzip_lines)))

        write_list_to_file(self.out_download_file, out_download_lines)
        write_list_to_file(self.out_unzip_file, out_unzip_lines)
        print('[Info] 写入完成: {}'.format(self.out_download_file))


def main():
    ex_processor = EssayXlsxProcessor()
    ex_processor.process()


if __name__ == '__main__':
    main()

