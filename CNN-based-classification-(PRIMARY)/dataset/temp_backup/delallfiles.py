import os

folders = []
for name in os.listdir():
    if not ('.' in name and len(name.split('.')) > 1):
        folders.append(name)

for fold in folders:
    for filename in os.listdir(fold):
        file_path = os.path.join(fold, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))