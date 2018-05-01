# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from logger import create_logger
from utils import get_filenames, filter_filenames
from preprocessing import DataReader, DataParser

against = 'protoss'
output_size = 128

if __name__ == "__main__":
    # 0. Instantiate a 'logger'
    base_logger = create_logger(name='base', level='INFO', stream=True, file=False)

    # 1. Instantiate a 'DataReader'
    reader = DataReader(logger=base_logger)

    # 2. Get important 'getNames'
    unit_names = reader.read_unit_names(
        filepath='Z:/1. 프로젝트/2018_삼성SDS_스타크래프트/Supervised/Importance_getName_{}.xlsx'.format(against.lower()),
        threshold=2
    )

    # 3. Get important 'colNames'
    col_names = reader.read_column_names(
        filepath='Z:/1. 프로젝트/2018_삼성SDS_스타크래프트/Supervised/Importance_column_revised.xlsx',
        threshold=2
    )

    # 4. Instantiate a 'DataParser'
    parser = DataParser(
        logger=base_logger,
        units_to_keep=unit_names,
        columns_to_keep=col_names,
        output_size=output_size)

    # 4. Get names of csv files from which to import replay data
    filedir = 'Z:/1. 프로젝트/2018_삼성SDS_스타크래프트/Supervised/parsing 참조 파일/'
    filenames = get_filenames(filedir, logger=base_logger)
    for filename in filenames:
        # 5. Read data from csv files
        replay_info = reader.get_replay_info(filename)
        versus = replay_info.get('versus')
        if not filter_filenames(versus=versus, against=against):
            continue
        abspath = os.path.join(filedir, filename)
        replay = reader.read_frames_from_replay(abspath)
        samples, sample_infos = parser.parse(replay=replay, sparse=False)
        break
