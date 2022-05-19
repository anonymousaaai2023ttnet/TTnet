# encoding: utf-8

import torch.nn as nn
import torch
from eevbnn.net_bin import Binarize01Act, InputQuantizer
from src.nn.BaseBlock_general import BaseBlock_general_kernel_general

#from src.utils.utils import InputQuantizer, activation_quantize_fn, dec2bin, linear_Q_fn, Binarize01Act


class model_general(nn.Module):
    # 98.5
    def __init__(self, args):
        #Linear = linear_Q_fn(w_bit=args.wbit)
        super(model_general, self).__init__()
        self.output_size = 10
        self.BN0 = nn.BatchNorm2d(args.nchannel)
        self.args = args
        #self.padding_input = nn.ZeroPad2d(2)
        self.inputquant = InputQuantizer(args.step_quantization)
        self.act_bin = Binarize01Act()
        self.blocks = []
        self.nchannel = args.nchannel
        self.filters = args.filters  # [32,32,64,128]
        self.amplifications = args.amplifications  # [10,10,10,10]
        self.strides = args.strides  # [False, True, True, True]
        self.channels = args.groups  # ["diff", "diff", "same", "same"]
        self.kernelsizes = args.kernelsizes  # [3, 3, 3, 3]
        #print(self.kernelsizes)
        last = False
        f0 = args.nchannel

        self.coefinter = int(self.filters[-1]/10)

        self.coefinter2 = int(self.filters[-1]/self.channels[0])


        assert len(self.filters)==len(self.strides)==len(self.channels)==len(self.kernelsizes)

        for iff, f in enumerate(self.filters):
            #print(self.filters, f,f0, self.ts, self.types, self.kernelsizes, self.downsamples)
            #print()
            if iff == len(self.filters) - 1:
                last = True
            b = BaseBlock_general_kernel_general(self.args, f0, f, self.kernelsizes[iff], t=self.amplifications[iff],channel=self.channels[iff],
                                                 stride=self.strides[iff], last=last)
            self.blocks.append(b)
            f0 = f
        self.layer = nn.Sequential(*self.blocks)
        self.fc0 = nn.Linear(int(args.fc/10), 1, bias=False) #nn.Linear(args.fc, self.output_size, bias=False)
        self.fc1 = nn.Linear(int(args.fc/10), 1, bias=False) #nn.Linear(args.fc, self.output_size, bias=False)
        self.fc2 = nn.Linear(int(args.fc/10), 1, bias=False) #nn.Linear(args.fc, self.output_size, bias=False)
        self.fc3 = nn.Linear(int(args.fc/10), 1, bias=False) #nn.Linear(args.fc, 1, bias=False)
        self.fc4 = nn.Linear(int(args.fc/10), 1, bias=False) #nn.Linear(args.fc, 1, bias=False)
        self.fc5 = nn.Linear(int(args.fc/10), 1, bias=False) #nn.Linear(args.fc, 1, bias=False)
        self.fc6 = nn.Linear(int(args.fc/10), 1, bias=False)  # nn.Linear(args.fc, 1, bias=False)
        self.fc7 = nn.Linear(int(args.fc/10), 1, bias=False)  # nn.Linear(args.fc, 1, bias=False)
        self.fc8 = nn.Linear(int(args.fc/10), 1, bias=False)  # nn.Linear(args.fc, 1, bias=False)
        self.fc9 = nn.Linear(int(args.fc/10), 1, bias=False)  # nn.Linear(args.fc, self.output_size, bias=False)
        self.fc_all = [self.fc0,
        self.fc1,
        self.fc2,
        self.fc3,
        self.fc4,
        self.fc5,
        self.fc6,
        self.fc7,
        self.fc8,
        self.fc9]



    def preprocessing(self, inputs):
        x = self.inputquant(inputs)
        x = self.BN0(x)
        self.inputnim_post_process = self.act_bin(x)
        x = 2 * self.act_bin(x) - 1
        return x

    def forward(self, inputs):

        x = self.preprocessing(inputs)
        feat = self.layer(x)
        self.featuresf = feat.view(feat.shape[0], -1)

        if self.args.nchannel == 1:
            res = None
            for i in range(10):
                x = feat[:,self.coefinter*i:self.coefinter*i+self.coefinter,:,:]
                x = x.view(x.shape[0], -1)
                x = self.fc_all[i](x)
                if res is None:
                    res = x.clone()
                else:
                    res = torch.cat([res, x],dim=1 )
        else:
            feat_R = feat[:, :self.coefinter2, :, :]
            feat_G = feat[:, self.coefinter2:2*self.coefinter2, :, :]
            feat_B = feat[:, 2*self.coefinter2:3*self.coefinter2, :, :]
            res = None
            for i in range(10):
                xr = feat_R[:, int(self.coefinter/3)*i:int(self.coefinter/3)*i+int(self.coefinter/3),:,:]
                xg = feat_G[:, int(self.coefinter/3) * i:int(self.coefinter/3) * i + int(self.coefinter/3), :, :]
                xb = feat_B[:, int(self.coefinter/3) * i:int(self.coefinter/3) * i + int(self.coefinter/3), :, :]
                x = torch.cat([xr, xg, xb], dim=1)
                x = x.view(x.shape[0], -1)
                x = self.fc_all[i](x)
                if res is None:
                    res = x.clone()
                else:
                    res = torch.cat([res, x], dim=1)



        return res








