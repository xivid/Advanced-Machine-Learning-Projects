import numpy as np
import skvideo.io  
import os 
import sys 
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))

def get_videos_from_folder(data_folder):
	'''
	get a list of video x wehre each video is a numpy array in the format [n_frames,width,height] 
	with uint8 elements.
	argument: relative path to the data_folder from the source folder.
	'''
	data_folder = os.path.join(dir_path,data_folder)
	x = []
	file_names = []

	if os.path.isdir(data_folder):
		for dirpath, dirnames, filenames in os.walk(data_folder):
			for filename in filenames:
				file_path = os.path.join(dirpath, filename)
				statinfo = os.stat(file_path)
				if statinfo.st_size != 0:
					video = skvideo.io.vread(file_path, outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
					x.append(video)
					file_names.append(int(filename.split(".")[0]))

	indices = sorted(range(len(file_names)), key=file_names.__getitem__)
	x = np.take(x,indices)
	return x

def get_target_from_csv(csv_file):
	'''
	get a numpy array y of labels. the order follows the id of video. 
	argument: relative path to the csv_file from the source folder.
	'''
	csv_file = os.path.join(dir_path,csv_file)
	with open(csv_file, 'r') as csvfile:
		label_reader = pd.read_csv(csvfile)
		y = label_reader['y']
	
	y = np.array(y)
	return y