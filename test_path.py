import csv
import os
traindir = "D:\\GATech\\Sorghum-Kaggle\\train_images"
file_path = os.path.join(traindir, 'train_cultivar_mapping.csv')
img = []
cnt = 0
target_dict = {}
target_cnt = 0
with open(file_path, 'r') as f:
    f.readline()
    reader = csv.reader(f, delimiter=',')
    for line in reader:
        target_name = line[1]
        if target_name in target_dict:
            pass
        else:
            target_dict[target_name] = target_cnt
            target_cnt += 1
        path = os.path.join(traindir, line[0])
        item = (path, target_dict[target_name])
        img.append(item)
        cnt += 1
