import os, shutil
folder = '.\\training_set\\'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    if len(file_path.split('.')) == 3:
        continue
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))