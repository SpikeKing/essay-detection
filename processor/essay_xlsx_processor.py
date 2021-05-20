#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 12.5.21
"""

import os
import sys
from urllib import parse

from myutils.project_utils import *
from root_dir import DATA_DIR
from x_utils.oss_utils import traverse_oss_folder


class EssayXlsxProcessor(object):
    def __init__(self):
        # self.file_path = os.path.join(DATA_DIR, "DHSD-ZW-003.xlsx")
        self.file_path = os.path.join(DATA_DIR, "DHSD-ZW-003.xls")
        self.out_download_file = os.path.join(DATA_DIR, 'DHSD-ZW-003.download.txt')
        self.out_unzip_file = os.path.join(DATA_DIR, 'DHSD-ZW-003.unzip.txt')
        self.out_zip_file = os.path.join(DATA_DIR, 'DHSD-ZW-003.zip.txt')
        create_file(self.out_download_file)
        create_file(self.out_unzip_file)
        create_file(self.out_zip_file)

    def process(self):
        data_lines = read_excel_file(self.file_path)

        out_download_lines, out_unzip_lines, out_zip_lines = [], [], []  # 下载数据列表
        for idx, data_line in enumerate(data_lines):
            if idx == 0:
                continue
            url = data_line[20]
            name = data_line[6].replace('/', '_')
            print('[Info] url: {}'.format(url))
            print('[Info] name: {}'.format(name))
            out_download_line = "wget -O {} {}".format(name+".zip", url)
            out_unzip_line = "unzip {} -d {}".format(name+".zip", name)
            out_zip_line = "zip -r {}.zip {}".format(name, name)
            out_download_lines.append(out_download_line)
            out_unzip_lines.append(out_unzip_line)
            out_zip_lines.append(out_zip_line)

        print('[Info] 文件数: {}'.format(len(out_download_lines)))
        print('[Info] 文件数: {}'.format(len(out_unzip_lines)))

        write_list_to_file(self.out_download_file, out_download_lines)
        write_list_to_file(self.out_unzip_file, out_unzip_lines)
        write_list_to_file(self.out_zip_file, out_zip_lines)
        print('[Info] 写入完成: {}'.format(self.out_download_file))

    def generate_res_xslx(self):
        url_list = traverse_oss_folder("zhengsheng.wcl/essay-library/datasets/20210513/essay-v2-zip-out/", ext='zip')
        print('[Info] 文件数: {}'.format(len(url_list)))
        out_path = os.path.join(DATA_DIR, 'essay-v2-out.xlsx')

        excel_list = []
        for url in url_list:
            name = url.split('/')[-1]
            book_name = parse.unquote(name)
            excel_list.append([book_name, url])

        write_list_to_excel(out_path, ["书名", "url"], excel_list)
        print('[Info] 写入完成: {}'.format(out_path))


def main():
    ex_processor = EssayXlsxProcessor()
    ex_processor.process()
    # ex_processor.generate_res_xslx()


if __name__ == '__main__':
    main()

