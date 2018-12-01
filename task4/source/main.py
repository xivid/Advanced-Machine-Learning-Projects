from get_data import get_videos_from_folder,get_target_from_csv
import os
import numpy as np
from utils import save_solution

dir_path = os.path.dirname(os.path.realpath(__file__))
train_folder = os.path.join(dir_path,"../train/")
test_folder = os.path.join(dir_path,"../test/")

train_target = os.path.join(dir_path,'../train_target.csv')
my_solution_file = os.path.join(dir_path,'../solution.csv')

x_train = get_videos_from_folder(train_folder)
y_train = get_target_from_csv(train_target)
x_test = get_videos_from_folder(test_folder)

dummy_solution = 0.1*np.ones(len(x_test))
save_solution(my_solution_file,dummy_solution)
