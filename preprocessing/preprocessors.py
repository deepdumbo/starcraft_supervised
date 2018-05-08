# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os
import csv
import time
import collections

import numpy as np
import pandas as pd
import scipy.sparse as sp

import pickle


class AbstractDataReader(abc.ABC):
    """Abstract class for defining a data reader."""
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def read_unit_names(self, filepath, threshold):
        pass

    @abc.abstractmethod
    def read_column_names(self, filepath, threshold):
        pass

    @abc.abstractmethod
    def read_frames_from_replay(self, filepath):
        pass


class SimpleDataReader(AbstractDataReader):
    # TODO: Add class docstring.
    def __init__(self, logger):
        self.logger = logger
        super(SimpleDataReader, self).__init__()

    def read_unit_names(self, filepath, threshold=2):
        # TODO: Add function docstring.
        filepath = os.path.abspath(filepath)
        self.logger.info(
            "Reading unit names from {}".format(filepath)
        )

        dataframe = pd.read_excel(filepath)

        mask_1 = dataframe['importance'] >= int(threshold)
        mask_2 = dataframe['category'] == 'getType().getName()'
        mask = [a * b for a, b in zip(mask_1, mask_2)]
        mask = np.array(mask, dtype=np.bool)

        result = dataframe.get('getName')[mask]
        result = result.tolist()
        result = self.remove_whitespaces(result)
        self.logger.info(
            "Returning important {} unit names.".format(result.__len__())
        )
        return result

    def read_column_names(self, filepath, threshold=3):
        # TODO: Add function docstring.
        filepath = os.path.abspath(filepath)
        self.logger.info(
            "Reading column names from {}".format(filepath)
        )
        dataframe = pd.read_excel(filepath)
        mask = dataframe['importance'] >= int(threshold)
        result = dataframe.get('column')[mask]
        result = result.tolist()
        result = self.remove_whitespaces(result)
        self.logger.info(
            "Returning {} important column names.".format(result.__len__())
        )
        return result

    def read_frames_from_replay(self, filepath):
        # TODO: Must also read lines between 'start - currentSituationReport' and
        # TODO: 'end - currentSituationReport'
        filepath = os.path.abspath(filepath)
        self.logger.info(
            "Reading game replay data from {}".format(filepath)
        )
        start = time.time()
        result = list()
        save = False
        with open(filepath, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for index, line in enumerate(reader):
                # Whether to save line
                if not line:
                    continue  # handling empty lines
                if line[0] == 'start-activeList':
                    frame = collections.defaultdict(list)
                    save = True
                    continue
                elif line[0] == 'end-activeList':
                    frame['data'] = np.stack(frame.get('data'), axis=0)
                    result.append(frame)
                    save = False
                    continue
                # Save line by line into dictionary (=frame)
                elif save:
                    # TODO: Fix the key names of the data dictionaries
                    if line[0].startswith('Frame Count'):
                        # Save the current frame count number
                        frame['frame_count'] = line[0].split(' : ')[-1]
                    elif line[0].startswith('size'):
                        # Save the number of units (my + opponent's)
                        frame['num_units'] = line[0].split(' : ')[-1]
                    elif line[0].startswith('playerId'):
                        # Save all column names of the current frame (remove whitespaces as well)
                        frame['colnames'] = [s.replace(' ', '') for s in line]
                    else:
                        # Save feature values (remove whitespaces as well)
                        frame['data'].append([s.replace(' ', '') for s in line])
                else:
                    pass
        elapsed = time.time() - start
        # Return main data as a list comprised of dictionaries for each frame count
        assert type(result) == list
        self.logger.info(
            "({:.4} seconds) Returning {} frames as a list from {}".format(
                elapsed, result.__len__(), filepath)
        )
        return result

    @staticmethod
    def get_replay_info(filepath):
        # TODO: Add function docstring.
        # Example: '투혼13_a7312474690f6e5dd86c8bf6d20616f5a201e17f_4_P_7_6_L_T_7_116_W_6624_Terran 박성균.csv'
        replay_info = dict()
        tokens = filepath.split('_')
        replay_info['map_name'] = tokens[0]
        replay_info['map_hash'] = tokens[1]
        replay_info['num_players'] = tokens[2]
        replay_info['location_p0'] = (tokens[4], tokens[5])
        replay_info['location_p1'] = (tokens[8], tokens[9])
        replay_info['versus'] = ''.join([v for i, v in enumerate(tokens) if i in [3, 6, 7, 10]])
        replay_info['max_frame'] = tokens[11]
        replay_info['my_race'], replay_info['pro'] = tokens[-1].split('.')[0].split(' ')
        return replay_info

    @staticmethod
    def remove_whitespaces(list_or_dict):
        # TODO: Add support for numpy 1D array & pandas.Series
        # FIXME: Must only remove whitespaces in the forefront
        if isinstance(list_or_dict, list):
            result = [s.replace(' ', '') for s in list_or_dict]
            return result
        elif isinstance(list_or_dict, dict):
            result = [(k, v.replace(' ', '')) for k, v in list_or_dict]
            result = dict(result)
            return result
        else:
            raise TypeError(
                "Expected 'list' or 'dictionary', got {}.".format(type(list_or_dict))
            )


class AbstractDataParser(abc.ABC):
    """Abstract class for defining a data parser."""
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def parse(self, replay):
        pass

    @abc.abstractmethod
    def make_sample(self, dataframe, output_size):
        pass

    @abc.abstractmethod
    def save(self, writefile, samples, sample_infos, replay_info, sparse):
        pass


class SimpleDataParser(AbstractDataParser):
    # TODO: Add class docstring.
    def __init__(self, logger, units_to_keep, columns_to_keep, output_size):
        self.logger = logger
        self.units_to_keep = units_to_keep
        self.columns_to_keep = columns_to_keep
        self.channel_keys = self.make_channel_keys()
        self.output_size = output_size
        self.original_size = 4096
        super(SimpleDataParser, self).__init__()

    def parse(self, replay):
        # TODO: Add function docstring.
        if not isinstance(replay, list):
            raise TypeError(
                "Expected a 'list', received {}.".format(type(replay))
            )
        samples = list()
        sample_infos = list()
        start = time.time()
        for i, frame_dict in enumerate(replay):
            assert isinstance(frame_dict, dict)
            # TODO: Consider using 'pandas.SparseDataFrame' for efficient memory
            sample_df = pd.DataFrame(frame_dict.get('data'), columns=frame_dict.get('colnames'))
            sample = self.make_sample(sample_df, output_size=self.output_size)
            sample = sample.astype(np.float32)

            # Create a dictionary containing information about the current sample
            sample_info = collections.defaultdict(list)
            sample_info['channel_names'] = self.channel_keys
            for key in frame_dict.keys():
                if key == 'data':
                    # TODO: Change this into a separate function
                    keep_cols = [c for c in self.columns_to_keep if c not in ['getName']]
                    sample_info['dataframe'] = sample_df[keep_cols].astype(np.int32)
                    try:
                        sample_info['getName'] = sample_df['getName'].astype(np.str_)
                    except KeyError as e:
                        print(str(e))
                else:
                    if key == 'colnames':
                        continue
                    else:
                        sample_info[key] = frame_dict[key]

            # Save them in separate lists
            samples.append(sample)
            sample_infos.append(sample_info)

        elapsed = time.time() - start
        self.logger.info(
            "({:.4} seconds) Parsed {} frames.".format(
                elapsed, samples.__len__())
        )

        return samples, sample_infos

    def make_sample(self, dataframe, output_size):
        """
        Make a sample provided a single frame of data.\n
        Arguments:\n
            dataframe: a pandas.DataFrame object of shape (units, columns).\n
            output_size: integer, only square output shapes are currently supported.\n
        Returns:\n
            sample: a dictionary of 2D numpy arrays.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "Expected a 'pandas.DataFrame', received {}.".format(type(dataframe))
            )

        if isinstance(self.units_to_keep, list):
            sample = dict([(k, np.zeros(shape=(output_size, output_size))) for k in self.units_to_keep])

        for i, row in dataframe.iterrows():
            unit_type = row['getName']  # This must correspond to a distinct channel
            if unit_type in self.units_to_keep:
                pos_x = int(row['getPosition.x']) * output_size // self.original_size
                pos_y = int(row['getPosition.y']) * output_size // self.original_size
                assert (pos_x, pos_y) <= (output_size, output_size)
                sample.get(unit_type)[pos_x, pos_y] += 1
            else:
                continue
        sample = np.stack([s for s in sample.values()], axis=-1)
        return sample

    def save(self, writefile, samples, sample_infos, replay_info, sparse=True):
        """Save replay to pickle file."""
        assert len(samples) == len(sample_infos)

        start = time.time()
        if sparse:
            channel_size = samples[0].shape[-1]
            samples = [np.reshape(s, (-1, channel_size)) for s in samples]
            samples = [sp.csr_matrix(s) for s in samples]

        replay_data = collections.defaultdict(list)
        for i in range(len(samples)):
            replay_data[i].append(samples[i])
            replay_data[i].append(sample_infos[i])

        replay = {'replay_data': replay_data, 'replay_info': replay_info}
        with open(writefile, 'wb') as f:
            pickle.dump(replay, f)

        elapsed = time.time() - start
        self.logger.info(
            "({:.4} seconds) Saved current replay to {}.".format(elapsed, writefile)
        )

    @property
    def num_channels(self):
        return len(self.units_to_keep) * len(self.columns_to_keep)

    def make_channel_keys(self):
        if isinstance(self.units_to_keep, list):
            return self.units_to_keep

    def make_channel_keys_v2(self):
        if isinstance(self.units_to_keep, list) and isinstance(self.columns_to_keep, list):
            return ["_".join([i, j]) for i in self.units_to_keep for j in self.columns_to_keep]
        else:
            raise ValueError("Expected type 'list' for unit names and column names.")


if __name__ == '__main__':
    pass
