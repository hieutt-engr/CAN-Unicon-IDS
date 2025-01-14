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

# from copy import copy, deepcopy
# import swifter

road_attributes = ['Timestamp', 'canID', 'Data', 'TimeDiffs', 'Flag']

# def pad_or_truncate(sequence, target_length):
#     """Pad the sequence with zeros or truncate to the target length."""
#     if len(sequence) < target_length:
#         sequence = sequence + [0] * (target_length - len(sequence))  # pad with zeros
#     return sequence[:target_length]  # truncate if longer

# def reshape_to_32x32(array):
#     """Reshape or pad a flattened array into a 32x32 matrix."""
#     padded = pad_or_truncate(array, 32 * 32)  # make sure the array has exactly 1024 elements
#     return np.array(padded).reshape((32, 32))  # reshape into 32x32 matrix

def pad_or_truncate(array, target_length):
    """Pad or truncate an array to a specific length using interpolation for padding."""
    if len(array) > target_length:
        return array[:target_length]
    elif len(array) < target_length:
        # Tạo phần padding sử dụng nội suy
        padding = np.linspace(array[-1], np.mean(array), target_length - len(array))
        return np.concatenate((array, padding))
    return array

def reshape_to_32x32(array):
    """Reshape or pad a flattened array into a 32x32 matrix using linspace padding."""
    padded = pad_or_truncate(array, 32 * 32)  # make sure the array has exactly 1024 elements
    return np.array(padded).reshape((32, 32))  # reshape into 32x32 matrix

def split_into_list(string, type):
    res = []
    
    if type == 'payload':
        # payload: convert from hex to binary
        for i in range(7):
            # take two hex characters and convert to binary with length 8 bits
            hex_value = string[:2]
            binary_value = bin(int(hex_value, 16))[2:].zfill(8)  # convert from hex to binary
            res.extend(list(binary_value))  # add each bit to the list
            string = string[2:]
        # process the last two characters
        hex_value = string[-2:]
        binary_value = bin(int(hex_value, 16))[2:].zfill(8)
        res.extend(list(binary_value))
    
    elif type == 'canID':
        # handle CAN ID: convert from decimal to 32-bit binary
        res = list(bin(string)[2:].zfill(32))
    
    # convert binary string values ​​to integers (0 or 1)
    res = [int(bit) for bit in res]
    
    return res

def normalize_timestamp(timestamp):
    """
    Normalize timestamp using delta time and Min-Max scaling.

    Args:
        timestamp (numpy.ndarray): Array of timestamp values.

    Returns:
        numpy.ndarray: Normalized representation of timestamp values.
    """
    delta_time = timestamp - timestamp[0]
    
    # Step 2: Min-Max Scaling
    min_val = np.min(delta_time)
    max_val = np.max(delta_time)
    scaled = (delta_time - min_val) / (max_val - min_val)
    
    return scaled

def normalize_time_zscore(time_series):
    """Normalize time using Z-score normalization."""
    mean_val = time_series.mean()
    std_val = time_series.std()
    normalized = (time_series - mean_val) / std_val
    return normalized

def serialize_example(x, y): 
    """converts x, y to tf.train.Example and serialize"""
    #Need to pay attention to whether it needs to be converted to numpy() form
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

def  split_data(file_name, attack_id, window_size, strided_size):
    if not os.path.exists(file_name):
        print(file_name, ' does not exist!')
        return None

    print("Window size = {}, strided = {}".format(window_size, strided_size))
    df = pd.read_csv(file_name)
    df.columns = road_attributes
    print("Reading {}: done".format(file_name))
    df = df.sort_values('Timestamp', ascending=True)

    # binary 'can data' (CAN Data)
    df['Data'] = df['Data'].apply(lambda x: split_into_list(x, 'payload'))
    # binary 'canID' (CAN ID)
    df['canID'] = df['canID'].apply(lambda x: split_into_list(x, 'canID'))
    # normalize timestamp
    df['Timestamp'] = normalize_timestamp(df['Timestamp'])

    df = df.fillna(0)
    print("ROAD pre-processing: Done")

    as_strided = np.lib.stride_tricks.as_strided
    output_shape = ((len(df) - window_size) // strided_size + 1, window_size)
    canid = as_strided(df.canID, output_shape, (8 * strided_size, 8))
    data = as_strided(df.Data, output_shape, (8 * strided_size, 8)) 
    timestamp = as_strided(df.Timestamp, output_shape, (8 * strided_size, 8))

    label = as_strided(df.Flag, output_shape, (1 * strided_size, 1))

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

    df['id_seq'] = df['id_seq'].apply(lambda x: reshape_to_32x32([item for sublist in x for item in sublist]))
    df['data_seq'] = df['data_seq'].apply(lambda x: reshape_to_32x32([item for sublist in x for item in sublist]))
    df['timestamp'] = df['timestamp'].apply(lambda x: reshape_to_32x32(x))

    return df[['id_seq', 'data_seq', 'timestamp', 'label']].reset_index().drop(['index'], axis=1)


def main(indir, outdir, attacks, window_size, strided, attack_types):
    print(outdir)
    print("========================================================================================")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    data_info = {} 
    
    # process attack data
    for attack_id, attack in enumerate(attacks):
        # Split to get number of dataset
        attack_name = attack.split(',')[0]
        attack_ver = attack.split(',')[1]
        print('Attack: {} ==============='.format(attack_name))
        
        if int(attack_ver) == 1:
            if attack_types == 'road_fab':
                finput = '{}/{}_dataset.csv'.format(indir, attack_name)
                df = split_data(finput, attack_id + 1, window_size, strided)
            elif attack_types == 'road_mas':
                finput = '{}/{}_masquerade_dataset.csv'.format(indir, attack_name)
                df = split_data(finput, attack_id + 1, window_size, strided)
            else:
                df = []
                finput = '{}/{}_dataset.csv'.format(indir, attack_name)
                df_1 = split_data(finput, attack_id + 1, window_size, strided)
                df.append(df_1)
                finput = '{}/{}_masquerade_dataset.csv'.format(indir, attack_name)
                df_2 = split_data(finput, attack_id + 1, window_size, strided)
                df.append(df_2)
                df = pd.concat(df)
                
            print("Writing...................")
            foutput_attack = '{}/{}'.format(outdir, attack_name)
            foutput_normal = '{}/Normal_{}'.format(outdir, attack_name)
            df_attack = df[df['label'] != 0]
            df_normal = df[df['label'] == 0]
            write_tfrecord(df_attack, foutput_attack)
            write_tfrecord(df_normal, foutput_normal)
            data_info[foutput_attack] = df_attack.shape[0]
            data_info[foutput_normal] = df_normal.shape[0]
        else:
            for index in range(int(attack_ver)):
                if attack_types == 'road_fab':
                    finput = '{}/{}_{}_dataset.csv'.format(indir, attack_name, index+1)
                    df = split_data(finput, attack_id + 1, window_size, strided)
                elif attack_types == 'road_mas':
                    finput = '{}/{}_{}_masquerade_dataset.csv'.format(indir, attack_name, index+1)
                    df = split_data(finput, attack_id + 1, window_size, strided)
                else:
                    df = []
                    finput = '{}/{}_{}_dataset.csv'.format(indir, attack_name, index+1)
                    df_1 = split_data(finput, attack_id + 1, window_size, strided)
                    df.append(df_1)
                    finput = '{}/{}_{}_masquerade_dataset.csv'.format(indir, attack_name, index+1)
                    df_2 = split_data(finput, attack_id + 1, window_size, strided)
                    df.append(df_2)
                    df = pd.concat(df) 
                    
                print("Writing...................")
                foutput_attack = '{}/{}_{}'.format(outdir, attack_name, index+1)
                foutput_normal = '{}/Normal_{}_{}'.format(outdir, attack_name, index+1)
                df_attack = df[df['label'] != 0]
                df_normal = df[df['label'] == 0]
                write_tfrecord(df_attack, foutput_attack)
                write_tfrecord(df_normal, foutput_normal)
                
                data_info[foutput_attack] = df_attack.shape[0]
                data_info[foutput_normal] = df_normal.shape[0]
    print("Write record DONE!!!")
    json.dump(data_info, open('{}/datainfo.txt'.format(outdir), 'w'))
    print("DONE!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default="./data")
    parser.add_argument('--outdir', type=str, default="./data/road/preprocessed/fab_multi/TFRecord")
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--strided', type=int, default=32)
    parser.add_argument('--attack_type', type=str, default="all", nargs='+')
    args = parser.parse_args()
    
    if args.attack_type[0] == 'road_fab':
        attack_types = ['max_engine_coolant_temp_attack,1', 'fuzzing_attack,3', 'max_speedometer_attack,3', 'reverse_light_on_attack,3', 'reverse_light_off_attack,3', 'correlated_signal_attack,3']
    elif args.attack_type[0] == 'road_mas':
        attack_types = ['max_engine_coolant_temp_attack,1', 'max_speedometer_attack,3', 'reverse_light_on_attack,3', 'reverse_light_off_attack,3', 'correlated_signal_attack,3']
    elif args.attack_type[0] == 'all':
        attack_types = ['max_engine_coolant_temp_attack,1', 'fuzzing_attack,3', 'max_speedometer_attack,3', 'reverse_light_on_attack,3', 'reverse_light_off_attack,3', 'correlated_signal_attack,3']
    else:
        attack_types = [args.attack_type]
    
    if args.strided == None:
        args.strided = args.window_size
        
    outdir =  args.outdir + '_w{}_s{}'.format(args.window_size, args.strided)
    main(args.indir, outdir, attack_types, args.window_size, args.strided, args.attack_type[0])