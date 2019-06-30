import torch
import torch.nn as nn
import torch.nn.functional as F
import functions
import options

opt = options.Option()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.H1 = nn.Sequential(
            nn.Linear(in_features = 6, out_features = 16),
            nn.LeakyReLU(),
            nn.Linear(in_features = 16, out_features = 64),
            nn.LeakyReLU(),
            nn.Linear(in_features = 64, out_features = 32),
            nn.LeakyReLU()
        )
        self.H2 = nn.Sequential(
            nn.Linear(in_features = 64, out_features = 256),
            nn.LeakyReLU(),
            nn.Linear(in_features = 256, out_features = 128),
            nn.LeakyReLU()
        )
        self.H3 = nn.Sequential(
            nn.Linear(in_features = 256, out_features = 128),
            nn.LeakyReLU()
        )

        self.FC1 = nn.Sequential(
            nn.Linear(in_features = 128, out_features = 128),
            nn.LeakyReLU()
        )
        self.FC2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU()
        )
        self.FC3 = nn.Sequential(
            nn.Linear(128, opt.pointnum*3)
        )

    def forward(self, x):
        #x (batch * 10000 * 3)
        x1 = functions.add_k_nearnest(x)
        x2 = self.H1(x1)

        #x2 (batch * 10000 * k * 32)
        x2 = torch.sum(x2, dim = 2)
        x2 = functions.add_k_nearnest(x2)
        x3 = self.H2(x2)

        #x3 (batch * 10000 * k * 128)
        x3 = torch.sum(x3, dim = 2)
        x3 = functions.add_k_nearnest(x3)
        x4 = self.H3(x3)

        #x4 (batch * 10000 * k * 1024)
        x4 = torch.sum(x4, dim = 2)
        x5 = torch.max(x4, dim = 1)[0]

        #x5 (batch * 1024)
        x6 = self.FC1(x5)

        #x6 (batch * 512)
        x7 = self.FC2(x6)

        #x7 (batch * 512)
        x8 = self.FC3(x7)

        #x8 (batch * (10000*3))
        out = x8.reshape(x8.shape[0], opt.pointnum, 3)

        return out

    def lossfunc(self, out, y):
        #out y (batch * 10000 * 3)
        temp = functions.find_nearnest(y, out)
        temp = temp - y
        temp = temp * temp
        temp = torch.sum(temp, dim = 2)
        temp = torch.sqrt(temp)
        temp = temp / opt.pointnum
        return torch.sum(temp) / temp.shape[0]
