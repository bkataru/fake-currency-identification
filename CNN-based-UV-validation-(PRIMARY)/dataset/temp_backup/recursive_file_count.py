import os

folders = []
for name in os.listdir():
    if not ('.' in name and len(name.split('.')) > 1):
        folders.append(name)

folders = ['test_set', 'training_set']

file_count_excep = []

total = 0
for fold in folders:
    folds2 = os.listdir(fold)
    for fold2 in folds2:
        fold2_files = os.listdir(fold + '\\' + fold2)
        file_count = len(fold2_files)
        if file_count != 200 and file_count != 100:
            file_count_excep.append(fold + '\\' + fold2)
        print(fold2, file_count)
        total += file_count

print('=' * 50)
print(total)
print(file_count_excep)