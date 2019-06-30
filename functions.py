import torch 
import numpy as np
import options 
import copy

opt = options.Option()
global order_message

def cal_k_nearnest(point):
    temp = point.reshape(point.shape[0], point.shape[1], 1, point.shape[2])
    temp = temp.repeat(1, 1, opt.pointnum, 1)
    temp2 = copy.deepcopy(temp)
    temp2 = temp2.permute([0, 2, 1, 3])
    temp = temp - temp2
    temp = temp * temp
    temp = torch.sum(temp, dim=3)
    temp = torch.sqrt(temp)

    global order_message
    order_message = torch.topk(temp, opt.k, dim = 2)[1]

#input batch * 10000 * size
#return batch * 10000 * k * 2size
def add_k_nearnest(point):
    temp = point.reshape(point.shape[0], point.shape[1], 1, point.shape[2])
    temp = temp.repeat(1, 1, opt.k, 1)
    temp2 = torch.stack([ 
                torch.stack([
                    torch.stack([
                        point[batch][order_message[batch][x][k]] for k in range(opt.k)
                    ]) for x in range(opt.pointnum)
                ]) for batch in range(point.shape[0])
            ])
    res = torch.cat((temp, temp2), dim = 3)
    return res

def add_all_nearnest(point):
    temp = point.reshape(point.shape[0], point.shape[1], 1, point.shape[2])
    temp = temp.repeat(1, 1, opt.pointnum, 1)
    temp2 = temp.permute([0, 2, 1, 3])
    res = torch.cat((temp, temp2), dim = 3)

    return res

def find_nearnest(source, target):
    temp = source.reshape(source.shape[0], source.shape[1], 1, source.shape[2])
    temp = temp.repeat(1, 1, opt.pointnum, 1)
    temp2 = target.reshape(target.shape[0], target.shape[1], 1, target.shape[2])
    temp2 = temp2.repeat(1, 1, opt.pointnum, 1)
    temp2 = temp2.permute([0, 2, 1, 3])
    temp = temp - temp2
    temp = temp * temp
    temp = torch.sum(temp, dim=3)
    temp = torch.sqrt(temp)

    order = torch.argmax(temp, dim=2)

    res = torch.stack([
        torch.stack([
            target[batch][order[batch][k]] for k in range(opt.pointnum)
        ]) for batch in range(source.shape[0])
    ])

    return res