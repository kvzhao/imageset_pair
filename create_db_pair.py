# !/usr/bin/env python

from subprocess import call, check_call
import re
import os, sys

# dir_paths
project_path = '/home/kv/research/trackers_ranking/'
root_path = project_path + '/caffe_for_trackerRank/'
#db_save_path = root_path + 'data/optflow_same_observer/'
db_save_path = project_path + 'dataset/optflow_same_observer_pair/'
try:
    os.stat(db_save_path)
except:
    os.mkdir(db_save_path)

## Regression
train_list_path = project_path + 'dataset/ALOV_optical/CSK/data_list_all_attr_train.txt'
test_list_path = project_path + 'dataset/ALOV_optical/CSK/data_list_all_attr_test.txt'

## Classification 
#train_list_path = project_path + 'dataset/ALOV_optical/CSK/data_list_all_attr_labeled_train.txt'
#test_list_path = project_path + 'dataset/ALOV_optical/CSK/data_list_all_attr_labeled_test.txt'

# file_names
db_name_prefix = 'optflow_dataset_observer_csk_pair_reg'

# commandline options
backend_flag = '-backend lmdb'
shuffle_flag = '-shuffle=false'
width_flag  = '-resize_height=64'
height_flag = '-resize_width=64'
#encode_flag = '-encode_type=jpg'
gray_flag = '-gray=false'

# program
image_converter = root_path + 'build/tools/convert_imagepair.bin'
image_mean_computer = root_path + 'build/tools/compute_image_mean.bin'

# delete the existing files
os.system('rm -rf ' + db_save_path + '*')
print 'convert train'
db_filename = db_save_path + db_name_prefix + '_train'
call([image_converter, height_flag, width_flag, shuffle_flag, project_path, train_list_path, db_filename])
call([image_mean_computer, db_filename, db_filename +'/mean.binaryproto'])
print 'convert test'
db_filename = db_save_path + db_name_prefix + '_test'
call([image_converter, height_flag, width_flag, shuffle_flag, project_path, test_list_path, db_filename])
call([image_mean_computer, db_filename, db_filename +'/mean.binaryproto'])
