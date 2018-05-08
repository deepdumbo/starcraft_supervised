# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
import deepdish

from logger import create_logger
from preprocessors import SimpleDataReader, SimpleDataParser
from utils import get_filenames, filter_filenames, get_game_result

against = 'protoss'
output_size = 128

if __name__ == "__main__":

    # Step 0. Instantiate a 'logger'
    base_logger = create_logger(name='base', level='INFO', stream=True, file=False)

    # Step 1. Instantiate a 'DataReader'
    reader = SimpleDataReader(logger=base_logger)

    # Step 2. Get important 'getNames'
    unit_names = reader.read_unit_names(
        filepath='Z:/1. 프로젝트/2018_삼성SDS_스타크래프트/Supervised/Importance_getName_{}.xlsx'.format(against.lower()),
        threshold=2
    )

    # Step 3. Get important 'colNames'
    col_names = reader.read_column_names(
        filepath='Z:/1. 프로젝트/2018_삼성SDS_스타크래프트/Supervised/Importance_column_revised.xlsx',
        threshold=2
    )

    # Step 4. Instantiate a 'DataParser'
    parser = SimpleDataParser(
        logger=base_logger,
        units_to_keep=unit_names,
        columns_to_keep=col_names,
        output_size=output_size)

    # Step 5. Get names of csv files from which to import replay data
    # filedir = 'Z:/1. 프로젝트/2018_삼성SDS_스타크래프트/Supervised/parsing 참조 파일/'
    filedir = 'D:/parsingData/data(선수별)/박성균'
    filenames = get_filenames(filedir, logger=base_logger)

    # Step 6. Read, parse, and save replay data
    i = 0
    for filename in filenames:
        if i == 10:
            break
        # 6-1. Read basic information regarding the current replay
        replay_info = reader.get_replay_info(filename)
        assert isinstance(replay_info, dict)

        # 6-2. Skip current file if it does not match our interest
        versus = replay_info.get('versus')
        if not filter_filenames(versus=versus, against=against):
            base_logger.info('Skipping {}, none of our interest.'.format(filename))
            continue

        # 6-3. Read sample frames from replay (at a 72-frame interval)
        abspath = os.path.join(filedir, filename)
        replay = reader.read_frames_from_replay(abspath)
        assert isinstance(replay, list)

        # 6-4. Parse replay data to a list of samples and a list of sample infos
        samples, sample_infos = parser.parse(replay=replay)
        assert isinstance(samples, list) and isinstance(sample_infos, list)

        # 6-5. Write replay data as a single h5 file
        win_or_lose =  get_game_result(versus=versus, against=against)
        writedir = 'D:/trainingData_v0/data(선수별)/박성균/'
        if not os.path.isdir(writedir):
            os.makedirs(writedir)
        writefile = '{}_{}_{}_{}_{}.h5'.format(
            versus, win_or_lose, replay_info.get('pro'), replay_info.get('max_frame'), replay_info.get('map_hash')
        )
        try:
            parser.save(writefile=writefile,
                        samples=samples,
                        sample_infos=sample_infos,
                        replay_info=replay_info,
                        sparse=True)
            i += 1
            base_logger.info('{} replay files have been parsed.'.format(i))
        except OverflowError as e:
            base_logger.info('Unable to write; {}'.format(str(e)))
        # 8. Read back written data (optional)
        #if False:
        #    tmp = deepdish.io.load(writefile)
