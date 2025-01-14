"""
Used to convert .csv into tfrecord format
"""
import os
import pandas as pd
import numpy as np
import glob
import swifter
import json
# from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm
import argparse

attributes = ['Timestamp', 'canID', 'DLC', 
                           'Data0', 'Data1', 'Data2', 
                           'Data3', 'Data4', 'Data5', 
                           'Data6', 'Data7', 'Flag']
def fill_flag(sample):
    if not isinstance(sample['Flag'], str):
        col = 'Data' + str(sample['DLC'])
        sample['Flag'], sample[col] = sample[col], sample['Flag']
    return sample

   
def serialize_example(x, y):
    """Converts x (which will now be 32x32x3), y to tf.train.Example and serializes"""
    id_seq, data_seq = x

    # Flatten the 32x32 matrices into a single array before saving
    id_seq = tf.train.Int64List(value = np.array(id_seq).flatten())
    data_seq = tf.train.Int64List(value = np.array(data_seq).flatten())
    # timestamp = tf.train.Int64List(value = np.array(timestamp).flatten())

    label = tf.train.Int64List(value=np.array([y]))

    features = tf.train.Features(
        feature={
            "id_seq": tf.train.Feature(int64_list=id_seq),
            "data_seq": tf.train.Feature(int64_list=data_seq),
            # "timestamp": tf.train.Feature(int64_list=timestamp),
            "label": tf.train.Feature(int64_list=label)
        }
    )
    example = tf.train.Example(features=features)
    return example.SerializeToString()

def write_tfrecord(data, filename):
    tfrecord_writer = tf.io.TFRecordWriter(filename)
    for _, row in tqdm(data.iterrows()):
        X = (row['id_seq'], row['data_seq'])
        Y = row['label']
        tfrecord_writer.write(serialize_example(X, Y))
    tfrecord_writer.close()   

def normalize_timestamp(timestamp_series):
    """Normalize timestamp to range [0, 255]."""
    min_val = timestamp_series.min()
    max_val = timestamp_series.max()
    normalized = (timestamp_series - min_val) / (max_val - min_val) * 255
    return normalized.astype(int)  # Scale to [0, 255] and convert to integers

def pad_or_truncate(sequence, target_length):
    """Pad the sequence with zeros or truncate to the target length."""
    if len(sequence) < target_length:
        sequence = sequence + [0] * (target_length - len(sequence))  # pad with zeros
    return sequence[:target_length]  # truncate if longer

def reshape_to_32x32(array):
    """Reshape or pad a flattened array into a 32x32 matrix."""
    padded = pad_or_truncate(array, 32 * 32)  # make sure the array has exactly 1024 elements
    return np.array(padded).reshape((32, 32))  # reshape into 32x32 matrix

def preprocess(file_name, attack_id, window_size=32, strided_size=32):
    print("Window size = {}, strided = {}".format(window_size, strided_size))
    df = pd.read_csv(file_name, header=None, names=attributes)
    print("Reading {}: done".format(file_name))
    df = df.sort_values('Timestamp', ascending=True)
    df = df.swifter.apply(fill_flag, axis=1)  # Parallelization is faster
    
    # Change CAN ID from hex string to binary 32-bits length
    df['canID'] = df['canID'].apply(int, base=16).apply(bin).str[2:] \
        .apply(lambda x: x.zfill(32)).apply(list) \
        .apply(lambda x: list(map(int, x)))
    
    # Change Data bytes from hex string to binary 8-bits length
    num_data_bytes = 8
    for x in range(num_data_bytes):
        # Safely handle NaN values by skipping them during conversion
        df['Data' + str(x)] = df['Data' + str(x)].map(
            lambda x: bin(int(x, 16))[2:].zfill(8) if pd.notna(x) else [0]*8
        ).apply(list).apply(lambda x: list(map(int, x)))
    
    df = df.fillna(0)
    
    # Combine the individual Data bytes into one column 'Data' as a list of binary values
    data_cols = ['Data{}'.format(x) for x in range(num_data_bytes)]
    df['Data'] = df[data_cols].values.tolist()
    df['Data'] = df['Data'].apply(lambda x: [item for sublist in x for item in sublist])  # Flatten list
    
    df['Flag'] = df['Flag'].apply(lambda x: True if x == 'T' else False)
    
    # Normalize the timestamp for use in 32x32 image format
    # df['Timestamp'] = normalize_timestamp(df['Timestamp'])

    print("Pre-processing: Done")

    as_strided = np.lib.stride_tricks.as_strided
    output_shape = ((len(df) - window_size) // strided_size + 1, window_size)
    canid = as_strided(df.canID, output_shape, (8 * strided_size, 8))
    data = as_strided(df.Data, output_shape, (8 * strided_size, 8)) 
    # timestamp = as_strided(df.Timestamp, output_shape, (8 * strided_size, 8))
    
    label = as_strided(df.Flag, output_shape, (1 * strided_size, 1))

    df = pd.DataFrame({
        'id_seq': pd.Series(canid.tolist()),
        'data_seq': pd.Series(data.tolist()),
        # 'timestamp': pd.Series(timestamp.tolist()),
        'label': pd.Series(label.tolist())
    }, index=range(len(canid)))
    
    df['label'] = df['label'].apply(lambda x: attack_id if any(x) else 0)
    
    print("Aggregating data: Done")
    print('#Normal: ', df[df['label'] == 0].shape[0])
    print('#Attack: ', df[df['label'] != 0].shape[0])

    # Reshape sequences into 32x32 matrices
    df['id_seq'] = df['id_seq'].apply(lambda x: reshape_to_32x32([item for sublist in x for item in sublist]))
    df['data_seq'] = df['data_seq'].apply(lambda x: reshape_to_32x32([item for sublist in x for item in sublist]))
    # df['timestamp'] = df['timestamp'].apply(lambda x: reshape_to_32x32(x))

    return df[['id_seq', 'data_seq', 'label']].reset_index().drop(['index'], axis=1)


def main(indir, outdir, attacks, window_size, strided):
    print(outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    data_info = {}
    for attack_id, attack in enumerate(attacks):
        print('Attack: {} ==============='.format(attack))
        finput = '{}/{}_dataset.csv'.format(indir, attack)
        df = preprocess(finput, attack_id + 1, window_size, strided)
        print("Writing...................")
        foutput_attack = '{}/{}'.format(outdir, attack)
        foutput_normal = '{}/Normal_{}'.format(outdir, attack)
        df_attack = df[df['label'] != 0]
        df_normal = df[df['label'] == 0]
        write_tfrecord(df_attack, foutput_attack)
        write_tfrecord(df_normal, foutput_normal)
        
        data_info[foutput_attack] = df_attack.shape[0]
        data_info[foutput_normal] = df_normal.shape[0]
        
    json.dump(data_info, open('{}/datainfo.txt'.format(outdir), 'w'))
    print("DONE!")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default="./data/Car-Hacking")
    parser.add_argument('--outdir', type=str, default="./data/Car-Hacking/TFRecord")
    parser.add_argument('--window_size', type=int, default=None)
    parser.add_argument('--strided', type=int, default=None)
    parser.add_argument('--attack_type', type=str, default="all", nargs='+')
    args = parser.parse_args()
    
    if args.attack_type == 'all':
        attack_types = ['DoS', 'Fuzzy', 'gear', 'RPM']
    else:
        attack_types = [args.attack_type]
    
    if args.strided == None:
        args.strided = args.window_size
        
    outdir =  args.outdir + '_w{}_s{}'.format(args.window_size, args.strided)
    main(args.indir, outdir, attack_types, args.window_size, args.strided)
