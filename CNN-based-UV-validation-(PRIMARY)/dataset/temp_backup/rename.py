import os

count = 0
for filename in os.listdir():
    if not filename.split('.')[1] == 'py':
        name = filename.split('.')[0]
        os.rename(filename, '{}.jpg'.format(count))
        count += 1