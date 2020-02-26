import os

target_folders = ['training_set', 'test_set']
cur_path = '.\\'
folders = ['dashed', 'continuous', 'nopatch']

for target in target_folders:
    for fold in folders:
        os.mkdir(cur_path + "{}\\{}".format(target, fold))
