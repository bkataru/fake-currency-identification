import os

target_folders = ['training_set', 'test_set']
cur_path = '.\\'
folders = ['yes_watermark', 'no_watermark']

for target in target_folders:
    for fold in folders:
        os.mkdir(cur_path + "{}\\{}".format(target, fold))
