import os

folders = []
for name in os.listdir():
    if not ('.' in name and len(name.split('.')) > 1):
        folders.append(name)

total = 0
for fold in folders:
    files = os.listdir(fold)
    file_count = len(files)
    print(fold, file_count)
    total += file_count

print('=' * 50)
print(total)