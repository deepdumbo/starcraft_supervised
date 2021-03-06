# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os
import csv
import time
import itertools
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
    def read_reports_from_replay(self, filepath):
        pass

    @abc.abstractmethod
    def read_frames_from_replay(self, filepath):
        pass

    @abc.abstractmethod
    def read_geographicInfo_from_mapInfo(self, filepath):
        pass


class SimpleDataReader(AbstractDataReader):
    """SimpleDataReader."""
    def __init__(self, logger):
        self.logger = logger
        super(SimpleDataReader, self).__init__()

    def read_unit_names(self, filepath):
        # TODO: Add function docstring.
        filepath = os.path.abspath(filepath)
        self.logger.info(
            "Reading unit names from {}".format(filepath)
        )

        with open(filepath, 'r') as f:
            lines = f.readlines()
            lines = [l.strip() for l in lines]

        splitters = ['[Protoss_Units]',
                     '[Protoss_Buildings]',
                     '[Terran_Units]',
                     '[Terran_Buildings]'
                     ]

        result = {}
        for k, g in itertools.groupby(lines, lambda s: s in splitters):
            if k:
                key = list(g)[0][1:-1]  # ex. [example] --> example
            else:
                result[key] = list(g)

        self.logger.info(
            "Returning {} important unit names.".format(result.__len__())
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

    def read_reports_from_replay(self, filepath):
        """Read situation reports from replay."""
        filepath = os.path.abspath(filepath)
        self.logger.info(
            "Reading situation reports from the following replay:\n{}".format(filepath)
        )
        start = time.time()
        result = list()
        save = False
        with open(filepath, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for index, line in enumerate(reader):
                if not line:
                    continue  # skip empty lines
                if line[0] == 'start - currentSituationReport':
                    # Start save mode
                    report = []
                    colnames = None
                    save = True
                    continue
                elif line[0] == 'end - currentSituationReport':
                    try:
                        report = np.array(report, dtype=np.int8)
                        report = pd.DataFrame(report, columns=colnames)
                        result.append(report)
                    except ValueError as e:
                        self.logger.warning(
                            "Unable to parse {}, '{}'.".format(
                                filepath, str(e)
                            )
                        )
                        return None  # check main
                    # End save mode
                    save = False
                    continue

                elif save:
                    if line[0].startswith('Frame Count'):
                        # No need to pass
                        pass
                    elif line[0].startswith('playerID'):
                        colnames = [s.replace(' ', '') for s in line]
                    else:
                        report.append([s.replace(' ', '') for s in line])
                else:
                    pass
        elapsed = time.time() - start
        assert type(result) == list
        self.logger.info(
            "({:.4} seconds) Returning {} situation reports as a list from {}".format(
                elapsed, result.__len__(), filepath)
        )
        return result

    def read_frames_from_replay(self, filepath):
        """Read game frames from replay."""
        filepath = os.path.abspath(filepath)
        self.logger.info(
            "Reading game frames from the following replay:\n{}".format(filepath)
        )
        start = time.time()
        result = list()
        save = False
        with open(filepath, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for index, line in enumerate(reader):
                if not line:
                    continue  # skip empty lines
                if line[0] == 'start-activeList':
                    # Start save mode
                    frame = collections.defaultdict(list)
                    save = True
                    continue
                elif line[0] == 'end-activeList':
                    try:
                        frame['data'] = np.stack(frame.get('data'), axis=0)
                        result.append(frame)
                    except ValueError as e:
                        self.logger.warning(
                            "Unable to parse {}, '{}'.".format(
                                filepath, str(e)
                            )
                        )
                        return None  # check main
                    # End save mode
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
    
    def read_geographicInfo_from_mapInfo(self, filepath):
        """Reads map information from map configuration file."""
        filepath = os.path.abspath(filepath)
        self.logger.info(
            "Reading game map data from {}".format(filepath)
        )
        arr = []
        with open(filepath, 'r') as f:
            # Note that the mapInfo files use single-whitespace delimiters
            reader = csv.reader(f, delimiter=' ')
            save = False
            for index, line in enumerate(reader):
                if not line:
                    continue
                elif line[0].startswith('[Start]'):
                    map_info = dict()
                    continue
                elif line[0].startswith('지나갈'):
                    title = 'walkable'
                    save = False
                    continue
                elif line[0].startswith('높이에'):
                    # Save first map feature
                    arr = np.asarray(arr, dtype=np.int8)
                    if arr.shape != (512, 512):
                        self.logger.warning(
                            'Received map size of shape {}: {}'.format(arr.shape, filepath)
                        )
                    map_info[title] = arr
                    # Reset variables for next map feature
                    title = 'altitude'
                    arr = []
                    save = False
                    continue
                elif line[0].startswith('미네랄'):
                    # Save second map feature
                    arr = np.asarray(arr, dtype=np.int8)
                    if arr.shape != (512, 512):
                        self.logger.warning(
                            'Received map size of shape {}: {}'.format(arr.shape, filepath)
                        )
                    map_info[title] = arr
                    # Reset variables for next map feature
                    title = 'resource'
                    arr = []
                    save = False
                    continue
                elif line[0].startswith('mapWidthWalkRes'):
                    # This line is redundant, save after this line
                    save = True
                    continue
                elif line[0].startswith('[End]'):
                    # Save third map feature
                    arr = np.asarray(arr, dtype=np.int8)
                    if arr.shape != (512, 512):
                        self.logger.warning(
                            'Received map size of shape {}: {}'.format(arr.shape, filepath)
                        )
                    map_info[title] = arr
                    # This will be the end of this for loop
                    save = False
                    continue
                elif save:
                    # If 'save' is True, and none of above, save line
                    arr.append(line)
                else:
                    pass
        return map_info
        
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
        self.units_to_keep = units_to_keep  # Do not use this attribute
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

        if isinstance(self.channel_keys, list):
            sample = dict([(k, np.zeros(shape=(output_size, output_size))) for k in self.channel_keys])

        for i, row in dataframe.iterrows():
            unit_type = row['getName']  # This must correspond to a distinct channel
            if unit_type in self.channel_keys:
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
        assert isinstance(samples, list)
        assert isinstance(sample_infos, list)
        assert isinstance(replay_info, dict)
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
        # FIXME: deprecated, REMOVE!
        return len(self.units_to_keep) * len(self.columns_to_keep)

    def make_channel_keys(self):
        if isinstance(self.units_to_keep, list):
            return self.units_to_keep
        elif isinstance(self.units_to_keep, dict):
            as_list = []
            for k, v in self.units_to_keep.items():
                as_list.extend(v)
            return as_list
        else:
            raise ValueError

    def make_channel_keys_v2(self):
        # FIXME: deprecated, REMOVE!
        if isinstance(self.units_to_keep, list) and isinstance(self.columns_to_keep, list):
            return ["_".join([i, j]) for i in self.units_to_keep for j in self.columns_to_keep]
        else:
            raise ValueError("Expected type 'list' for unit names and column names.")


if __name__ == '__main__':
    pass
