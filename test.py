import numpy as np
import random
import math
import torch
import torch.optim as optim
import functions
import model
import data_loader
import options
import os
from tensorboardX import SummaryWriter


if __name__ == '__main__':

    writer = SummaryWriter()
    opt = options.Option()

    net = model.Net()

    #net.load_state_dict(torch.load('./network_model/model.para'))
    net.to(torch.device(opt.device_name))

    sample_data, sample_label = data_loader.get_train_data(0, 5)
    functions.cal_k_nearnest(sample_data)
    out = net(sample_data)

    with open('Output.off', 'w') as f:
        f.write('OFF\n')
        f.write(str(opt.pointnum) + ' 0 0\n')
        for i in range(opt.pointnum):
            f.write(str(out[0][i][0].item()) + ' ' + str(out[0][i][1].item()) + ' ' + str(out[0][i][2].item()) + '\n')
    
    with open('STD.off', 'w') as f:
        f.write('OFF\n')
        f.write(str(opt.pointnum) + ' 0 0\n')
        for i in range(opt.pointnum):
            f.write(str(sample_data[0][i][0].item()) + ' ' + str(sample_data[0][i][1].item()) + ' ' + str(sample_data[0][i][2].item()) + '\n')
    

    


        




