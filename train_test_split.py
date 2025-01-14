import tensorflow as tf
import numpy as np
import json
import argparse
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
import os
print('Tensorflow version', tf.__version__)

class TFwriter:
    def __init__(self, outdir, start_idx=0):
        print('Writing to: ', outdir)
        self._outdir = outdir
        self._start_idx = start_idx
        
    def serialize_example(self, x, y):
        """Converts x (which will now be 32x32x3), y to tf.train.Example and serializes"""
        id_seq, data_seq = x

        # Flatten the 32x32 matrices into a single array before saving, and ensure int64
        id_seq = tf.train.Int64List(value=np.array(id_seq).flatten().tolist())
        data_seq = tf.train.Int64List(value=np.array(data_seq).flatten().tolist())
        # timestamp = tf.train.Int64List(value=np.array(timestamp).flatten().tolist())

        label = tf.train.Int64List(value=[int(y)])

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

    def write(self, data, label):
        filename = os.path.join(self._outdir, str(self._start_idx)+'.tfrec')
        with tf.io.TFRecordWriter(filename) as outfile:
            outfile.write(self.serialize_example(data, label))
        self._start_idx += 1

        
def read_tfrecord(example, window_size):
    """Parses a TFRecord into the expected features."""
    
    feature_description = {
        'id_seq': tf.io.FixedLenFeature([32, 32], tf.int64),
        'data_seq': tf.io.FixedLenFeature([32, 32], tf.int64),
        # 'timestamp': tf.io.FixedLenFeature([32, 32], tf.int64),
        'label': tf.io.FixedLenFeature([1], tf.int64)
    }
    # parsed_example = tf.io.parse_single_example(example, feature_description)
    parsed_example = tf.io.parse_single_example(example, feature_description)
    # return parsed_example

    # # Convert the parsed features into a dictionary for easy access
    features = {
        'id_seq': parsed_example['id_seq'],
        'data_seq': parsed_example['data_seq'],
        # 'timestamp': parsed_example['timestamp']
    }
    
    # Return the input features and the label
    return features, parsed_example['label']
def write_tfrecord(dataset, tfwriter):
    for batch_data in iter(dataset):
        # Access the features from batch_data[0]
        features = zip(batch_data[0]['id_seq'], batch_data[0]['data_seq'])
        
        # Access the labels directly from batch_data[1]
        labels = batch_data[1]  # Assuming labels are directly available in batch_data[1]
        
        for x, y in zip(features, labels):
            tfwriter.write(x, y)  # Write the feature and label pair

            
def train_test_split(**args):
    """
    """
    if args['strided'] == None:
        args['strided'] = args['window_size']
        
    if args['car_model'] is None:
        data_dir = f"{args['data_path']}/TFRecord_w{args['window_size']}_s{args['strided']}"
    else:
        data_dir = f"{args['data_path']}/TFRecordTFrecord_{args['car_model']}_w{args['window_size']}_s{args['strided']}"
        
    out_dir = data_dir + '/{}'.format(args['rid'])
    train_dir = os.path.join(out_dir, 'train')
    val_dir = os.path.join(out_dir, 'val')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    data_info = json.load(open(data_dir + '/datainfo.txt'))
    train_writer = TFwriter(train_dir)
    val_writer = TFwriter(val_dir)
    
    train_ratio = 0.7
    batch_size = 1000

    total_train_size = 0
    total_val_size = 0

    for filename, dataset_size in data_info.items():
        print('Read from {}: {} records'.format(filename, dataset_size))
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(lambda x: read_tfrecord(x, args['window_size']), 
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(50000)

        train_size = int(dataset_size * train_ratio)
        val_size = (dataset_size - train_size)
        
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size)
        
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)
            
        write_tfrecord(train_dataset, train_writer)
        write_tfrecord(val_dataset, val_writer)
        
        total_train_size += train_size
        total_val_size += val_size
        
    print('Total training: ', total_train_size)
    print('Total validation: ', total_val_size)
    
            
if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/Car-Hacking')
    parser.add_argument('--car_model', type=str, default=None)
    parser.add_argument('--window_size', type=int)
    parser.add_argument('--strided', type=int) 
    parser.add_argument('--rid', type=int, default=1) 
    
    args = vars(parser.parse_args())
    print(args)
    train_test_split(**args)
