import os 
import numpy as np
import torch
import copy
import options

TrainingFile = 'ModelNet10'
path = os.walk('./Data/' + TrainingFile)

obj_num = 0
obj_class = dict()
TrainingSet = list()
TestSet = list()
current_class = 'nothing'
now = 0
opt = options.Option()
global TrainingSize
global TestSize

def get_TrainingSize():
    global TrainingSize
    return TrainingSize

def shuffle_TrainingSet():
    np.random.shuffle(TrainingSet)

def normalize_pointnum(point):
    if point.shape[0] == opt.pointnum:
        return point
    elif point.shape[0] < opt.pointnum:
        idx =  np.random.choice(range(0, point.shape[0]), opt.pointnum - point.shape[0])
        temp = copy.deepcopy(point[idx])
        res = np.concatenate((point, temp), axis=0)
        return res
    else:
        idx = np.random.choice(range(0, point.shape[0]), opt.pointnum)
        res = copy.deepcopy(point[idx])
        return res


def get_train_data(start, batch):
    device_name = opt.device_name
    data = np.load(TrainingSet[start][0])
    data = data.reshape((-1, opt.pointnum, data.shape[1]))
    labels = np.array([obj_class[TrainingSet[start][1]]])
    for idx in range(start + 1, start + batch):
        if idx == TrainingSize:
            break
        
        temp = np.load(TrainingSet[idx][0])
        temp = temp.reshape((-1, opt.pointnum, temp.shape[1]))
        temp_label = np.array([obj_class[TrainingSet[idx][1]]])
        
        data = np.concatenate((data, temp), axis = 0)
        labels = np.concatenate((labels, temp_label), axis = 0)
    return torch.from_numpy(data).to(torch.device(device_name)).float(), labels
        

def get_test_data(start, batch):
    device_name = opt.device_name
    data = np.load(TestSet[start][0])
    data = data.reshape((-1, opt.pointnum, data.shape[1]))
    labels = np.array([obj_class[TestSet[start][1]]])
    for idx in range(start + 1, start + batch):
        if idx == TestSize:
            break
        
        temp = np.load(TestSet[idx][0])
        temp = temp.reshape((-1, opt.pointnum, temp.shape[1]))
        temp_label = np.array([obj_class[TestSet[idx][1]]])
        
        data = np.concatenate((data, temp), axis = 0)
        labels = np.concatenate((labels, temp_label), axis = 0)
    return torch.from_numpy(data).to(torch.device(device_name)).float(), labels

def get_single_train_data():
    device_name = opt.device_name
    if now == len(TrainingSet):
        now = 0
        np.random.shuffle(TrainingSet)
    return torch.from_numpy(np.load(TrainingSet[now][0])).to(torch.device(device_name)).float(), obj_class[TrainingSet[now][1]]

def get_single_test_data():
    device_name = opt.device_name
    if now == len(TrainingSet):
        now = 0
        np.random.shuffle(TestSet)
    return torch.from_numpy(np.load(TestSet[now][0])).to(torch.device(device_name)).float(), obj_class[TestSet[now][1]]


for root, dirs, files in path:
    print(root)
    dir = root.split('/')[-1]
    if dir == TrainingFile:
        pass
    elif dir == 'train':
        for item in files:
            if item[-1] != 'f':
                continue
            filename = os.path.splitext(item)[0]
            TrainingSet.append((root + '/' + filename + '.npy', current_class))
    elif dir == 'test':
        for item in files:
            if item[-1] != 'f':
                continue
            filename = os.path.splitext(item)[0]
            TestSet.append((root + '/' + filename + '.npy', current_class))
    else:
        current_class = dir
        obj_class[dir] = obj_num
        obj_num = obj_num + 1
        if obj_num == 2:
            break
        
shuffle_TrainingSet()
global TrainingSize, TestSize
TrainingSize = len(TrainingSet)
TestSize = len(TestSet)