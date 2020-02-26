import os

folders = []
for name in os.listdir():
    if not ('.' in name and len(name.split('.')) > 1):
        folders.append(name)

folders = ['test_set', 'training_set']

total = 0
for fold in folders:
    folds2 = os.listdir(fold)
    for fold2 in folds2:
        fold2_files = os.listdir(fold + '\\' + fold2)
        file_count = len(fold2_files)
        print(fold2, file_count)
        total += file_count

print('=' * 50)
print(total)