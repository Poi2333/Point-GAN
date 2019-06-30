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

    net.to(torch.device(opt.device_name))
    optimizer = optim.Adam(net.parameters(), opt.learning_rate, betas=(0.9, 0.99))
    print(net)

    iter = 0
    for epoch in range(opt.total_epoch):
        print('Epoch:{}'.format(epoch))

        Epoch_loss = 0
        data_loader.shuffle_TrainingSet()

        for now in range(0, data_loader.get_TrainingSize(), opt.batch_size):

            sample_data, sample_label = data_loader.get_train_data(now, opt.batch_size)
            functions.cal_k_nearnest(sample_data)

            optimizer.zero_grad()
            out = net(sample_data)
            loss = net.lossfunc(out, sample_data)
            loss.backward()
            optimizer.step()

            Epoch_loss = Epoch_loss + loss.item()
            writer.add_scalar('Train/Loss', loss.item(), iter)
            iter = iter + 1
            print('{}/{}:loss: {}'.format(now, data_loader.get_TrainingSize(), loss.item()))

            if (now // opt.batch_size) % 10 == 9:
                torch.save(net.state_dict(), './network_model/model.para')
                os.system('cp ./network_model/model.para ./network_model/epoch' + str(epoch) + 'It' + str(now) + '.para')
                
        
        Epoch_loss = Epoch_loss/(data_loader.get_TrainingSize() / opt.batch_size)
        writer.add_scalar('Train/Epoch_loss', Epoch_loss, epoch)
        print('Epoch loss: {}'.format(Epoch_loss))



        




