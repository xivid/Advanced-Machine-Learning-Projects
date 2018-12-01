import tensorflow as tf
from get_data import get_videos_from_folder,get_target_from_csv
import sys
import numpy as np

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def save_tf_record(x,file_name,y = None):
	writer = tf.python_io.TFRecordWriter(file_name)
	if y is None:
		for video in x:
			sys.stdout.flush()
			feature = {'len': _int64_feature(video.shape[0]),
						'height': _int64_feature(video.shape[1]),
						'width': _int64_feature(video.shape[2]),
						'video': _bytes_feature(tf.compat.as_bytes(video.tostring()))}
			example = tf.train.Example(features=tf.train.Features(feature=feature))
			writer.write(example.SerializeToString())
	else:
		for video,label in zip(x,y):
			sys.stdout.flush()
			feature = {'len': _int64_feature(video.shape[0]),
						'height': _int64_feature(video.shape[1]),
						'width': _int64_feature(video.shape[2]),
						'video': _bytes_feature(tf.compat.as_bytes(video.tostring())),
						'label': _int64_feature(label)}
			example = tf.train.Example(features=tf.train.Features(feature=feature))
			writer.write(example.SerializeToString())

	writer.close()
	sys.stdout.flush()

def prob_positive_class_from_prediction(pred):
	return np.array([p['probabilities'][1] for p in pred])

def decode(serialized_example):
	features = tf.parse_single_example(
		serialized_example,
		features={
			'len': tf.FixedLenFeature([], tf.int64),
			'height': tf.FixedLenFeature([], tf.int64),
			'width': tf.FixedLenFeature([], tf.int64),
			'label': tf.FixedLenFeature([], tf.int64,default_value = 0),
			'video': tf.FixedLenFeature([], tf.string),
		})
	video = tf.decode_raw(features['video'], tf.uint8)
	height = features['height']
	width = features['width']
	length = features['len']
	shape = tf.stack([length,height,width])
	video = tf.reshape(video,shape)
	label = features['label']
	features = {'video':video}
	return features,label

def input_fn_from_dataset(files,batch_size = 1,num_epochs = None,shuffle = True):
	data_set = tf.data.TFRecordDataset(files)
	if shuffle:
		data_set = data_set.shuffle(buffer_size=len(files)) 
	data_set = data_set.map(decode)
	data_set = data_set.padded_batch(batch_size,padded_shapes= ({'video':[212,100,100]},[]))
	data_set = data_set.repeat(num_epochs)
	data_set = data_set.prefetch(batch_size)
	
	return data_set

def decode_frame(serialized_example):
	features = tf.parse_single_example(
		serialized_example,
		features={
			'len': tf.FixedLenFeature([], tf.int64),
			'height': tf.FixedLenFeature([], tf.int64),
			'width': tf.FixedLenFeature([], tf.int64),
			'label': tf.FixedLenFeature([], tf.int64,default_value = 0),
			'video': tf.FixedLenFeature([], tf.string),
		})
	video = tf.decode_raw(features['video'], tf.uint8)
	height = features['height']
	width = features['width']
	length = features['len']
	shape = tf.stack([length,height,width])
	video = tf.reshape(video,shape)
	label = features['label']
	label = tf.expand_dims(label,axis=-1)
	label = tf.tile(label,tf.expand_dims(length,axis=-1))
	features = {'frame':video}
	return features,label

def input_fn_frame_from_dataset(files,batch_size = 1,num_epochs = None):
	data_set = tf.data.TFRecordDataset(files)
	data_set = data_set.shuffle(buffer_size=len(files)) 
	data_set = data_set.map(decode_frame)
	data_set = data_set.apply(tf.contrib.data.unbatch())
	data_set = data_set.shuffle(buffer_size=batch_size)
	data_set = data_set.batch(batch_size)
	data_set = data_set.repeat(num_epochs)
	data_set = data_set.prefetch(batch_size)
	
	return data_set
