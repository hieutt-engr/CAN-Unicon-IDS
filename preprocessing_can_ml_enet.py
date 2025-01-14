import pandas as pd
# import vaex
import numpy as np
import glob
import dask.dataframe as dd
import json
import math
import csv
import time
import _warnings
import tensorflow as tf
from tqdm import tqdm
import argparse
import os

can_ml_attributes = ['timestamp', 'arbitration_id', 'data_field', 'attack']

def pad_or_truncate(sequence, target_length):
    """Pad the sequence with zeros or truncate to the target length."""
    if len(sequence) < target_length:
        sequence = sequence + [0] * (target_length - len(sequence))  # pad with zeros
    return sequence[:target_length]  # truncate if longer

def reshape_to_target_size(array, target_size=64):
    """Reshape or pad a flattened array into a target_size x target_size matrix."""
    padded = pad_or_truncate(array, target_size * target_size)  # Ensure target_size^2 elements
    return np.array(padded).reshape((target_size, target_size))  # Reshape into target_size x target_size

def split_into_list(string, type):
    res = []
    if type == 'payload':
        res = list(string)

    elif type == 'canID':
        res = list(bin(int(string))[2:].zfill(64))
    
    return [int(bit) for bit in res]

def parse_and_stack_timestamps(timestamps, target_size=64):
    """
    Parse and stack timestamps into a 2D array where each row is a parsed timestamp.
    """
    def parse_single_timestamp(timestamp):
        # Normalize timestamp and convert to binary array
        normalized = (timestamp - np.floor(timestamp)) * (2 ** 32)
        parsed = np.array(list(bin(int(normalized))[2:].zfill(64))).astype(float)
        return parsed

    # Parse and stack each timestamp
    stacked_timestamps = np.array([parse_single_timestamp(ts) for ts in timestamps])

    # Ensure the number of rows matches target_size by truncating or padding
    if stacked_timestamps.shape[0] > target_size:
        stacked_timestamps = stacked_timestamps[:target_size, :]
    elif stacked_timestamps.shape[0] < target_size:
        padding = np.zeros((target_size - stacked_timestamps.shape[0], stacked_timestamps.shape[1]))
        stacked_timestamps = np.vstack((stacked_timestamps, padding))

    return stacked_timestamps

def normalize_time_zscore(time_series):
    """Normalize time using Z-score normalization."""
    mean_val = time_series.mean()
    std_val = time_series.std()
    normalized = (time_series - mean_val) / std_val
    return normalized


def normalize_timestamp(timestamp):
    """
    Normalize timestamp using delta time and Min-Max scaling.

    Args:
        timestamp (numpy.ndarray): Array of timestamp values.

    Returns:
        numpy.ndarray: Normalized representation of timestamp values.
    """
    # Step 1: Tính delta time
    delta_time = timestamp - timestamp[0]  # Tính khoảng thời gian so với giá trị đầu tiên
    
    # Step 2: Min-Max Scaling
    min_val = np.min(delta_time)
    max_val = np.max(delta_time)
    scaled = (delta_time - min_val) / (max_val - min_val)  # Chuẩn hóa giá trị trong khoảng [0, 1]
    
    # Step 3: Trả về giá trị đã chuẩn hóa (float)
    return scaled

def serialize_example(x, y): 
    """converts x, y to tf.train.Example and serialize"""
    id_seq, data_seq, timestamp = x
    id_seq = tf.train.Int64List(value = np.array(id_seq).flatten())
    data_seq = tf.train.Int64List(value = np.array(data_seq).flatten())
    timestamp = tf.train.FloatList(value = np.array(timestamp).flatten())

    label = tf.train.Int64List(value = np.array([y]))

    features = tf.train.Features(
        feature = {
            "id_seq": tf.train.Feature(int64_list = id_seq),
            "data_seq": tf.train.Feature(int64_list = data_seq),
            "timestamp": tf.train.Feature(float_list = timestamp),
            "label" : tf.train.Feature(int64_list = label)
        }
    )
    example = tf.train.Example(features = features)
    return example.SerializeToString()

def write_tfrecord(data, filename):
    tfrecord_writer = tf.io.TFRecordWriter(filename)
    for _, row in tqdm(data.iterrows()):
        X = (row['id_seq'], row['data_seq'], row['timestamp'])
        Y = row['label']
        tfrecord_writer.write(serialize_example(X, Y))
    tfrecord_writer.close() 

def split_data(file_name, attack_id, window_size, strided_size, target_size=64):
    if not os.path.exists(file_name):
        print(file_name, ' does not exist!')
        return None

    print("Window size = {}, strided = {}".format(window_size, strided_size))
    df = pd.read_csv(file_name)
    df.columns = can_ml_attributes
    print("Reading {}: done".format(file_name))
    df = df.sort_values('timestamp', ascending=True)

    # Binary 'can data' (CAN Data)
    df['Data'] = df['data_field'].apply(lambda x: split_into_list(x, 'payload'))
    # Binary 'canID' (CAN ID)
    df['canID'] = df['arbitration_id'].apply(lambda x: split_into_list(x, 'canID'))

    # Normalize and stack timestamp
    df['timestamp'] = normalize_time_zscore(df['timestamp'].values)

    print("CAN-ML pre-processing: Done")

    print("Initial attack count before windowing:", df[df['attack'] != 0].shape[0])
    as_strided = np.lib.stride_tricks.sliding_window_view

    canid = as_strided(df.canID.values, window_shape=window_size)[::strided_size]
    data = as_strided(df.Data.values, window_shape=window_size)[::strided_size]
    timestamp = as_strided(df.timestamp.values, window_shape=window_size)[::strided_size]

    label = as_strided(df.attack.values, window_shape=window_size)[::strided_size]

    df = pd.DataFrame({
        'id_seq': pd.Series(canid.tolist()),
        'data_seq': pd.Series(data.tolist()),
        'timestamp': pd.Series(timestamp.tolist()),
        'label': pd.Series(label.tolist())
    }, index=range(len(canid)))

    df['label'] = df['label'].apply(lambda x: attack_id if any(x) else 0)

    print("Aggregating data: Done")
    print('#Normal: ', df[df['label'] == 0].shape[0])
    print('#Attack: ', df[df['label'] != 0].shape[0])

    df['id_seq'] = df['id_seq'].apply(lambda x: reshape_to_target_size([item for sublist in x for item in sublist], target_size))
    df['data_seq'] = df['data_seq'].apply(lambda x: reshape_to_target_size([item for sublist in x for item in sublist], target_size))
    # df['timestamp'] = df['timestamp'].apply(lambda x: parse_and_stack_timestamps(x, target_size))
    df['timestamp'] = df['timestamp'].apply(lambda x: reshape_to_target_size(x, target_size))
    return df[['id_seq', 'data_seq', 'timestamp', 'label']].reset_index().drop(['index'], axis=1)


def main(indir, outdir, attacks, window_size, strided, target_size=64):
    print(outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    data_info = {}

    for attack_id, attack in enumerate(attacks):
        print('Attack: {} ==============='.format(attack))
        finput = '{}/{}.csv'.format(indir, attack)
        print("Attack Id: ", attack_id)
        df = split_data(finput, attack_id + 1, window_size, strided, target_size=target_size)
        print("Writing...................")

        foutput_attack = '{}/{}'.format(outdir, attack)
        foutput_normal = '{}/Normal_{}'.format(outdir, attack)
        df_attack = df[df['label'] != 0].sample(frac=0.2, random_state=42)
        df_normal = df[df['label'] == 0].sample(frac=0.2, random_state=42)
        write_tfrecord(df_attack, foutput_attack)
        write_tfrecord(df_normal, foutput_normal)
        
        data_info[foutput_attack] = df_attack.shape[0]
        data_info[foutput_normal] = df_normal.shape[0]
    
    json.dump(data_info, open('{}/datainfo.txt'.format(outdir), 'w'))
    print("DONE!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default="./data/can-ml/2011-chevrolet-impala/merged")
    parser.add_argument('--outdir', type=str, default="./data/can-ml/2011-chevrolet-impala/preprocessed/all_features_30/TFRecord")
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--strided', type=int, default=32)
    args = parser.parse_args()
    
    attack_types = ["combined", "DoS", "fuzzing", "gear", "interval", "rpm", 
                "speed", "standstill", "systematic"]
    if args.strided is None:
        args.strided = args.window_size
        
    outdir =  args.outdir + '_w{}_s{}'.format(args.window_size, args.strided)
    main(args.indir, outdir, attack_types, args.window_size, args.strided)