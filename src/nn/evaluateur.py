import time
import os
import torch
import torch.nn as nn
import torchattacks
from torch.autograd import Variable
#from torch.nn.utils import prune
from torch.nn.utils import prune
from tqdm import tqdm


class Evaluateur:

    def __init__(self, dataloaders, args, device):
        """
        :param args:
        :param writer:
        :param device:
        :param rng:
        :param path_save_model:
        """
        self.args = args
        self.epochs = self.args.n_epoch
        self.batch_size = self.args.batch_size_test
        self.t = Variable(torch.Tensor([0.5]))
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.dataloaders = dataloaders
        self.gs = 0




    def eval(self, model, val_phase=['train', 'val']):
        since = time.time()
        #model.eval()
        with torch.no_grad():
            for phase in val_phase:
                running_corrects = 0.0
                nbre_sample = 0
                tk0 = tqdm(self.dataloaders[phase], total=int(len(self.dataloaders[phase])))
                for i, data in enumerate(tk0):
                    inputs, labels = data
                    #print(inputs.shape)
                    outputs = model(inputs.to(self.device))
                    _, predicted = torch.max(outputs.data, 1)
                    running_corrects += (predicted == labels.to(self.device)).sum().item()
                    nbre_sample += labels.size(0)
                acc = running_corrects / nbre_sample
                print('{} Acc: {:.4f}'.format(
                    phase, acc))
            time_elapsed = time.time() - since
            print('Evaluation complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
        return acc



    def prunning_model_v2(self, model, global_sparsity):
        with torch.no_grad():
            if len(self.args.filters) ==2:

                parameters_to_prune = []
                for layer, module in enumerate(model.features):
                    if layer == 1 or layer == 3 or layer == 4 or layer == 6 or layer == 7 or layer == 9 \
                            or layer == 10 or layer == 12 or layer == 13 or layer == 17 or layer == 16:
                        parameters_to_prune.append((module, 'weight'))


            elif len(self.args.filters) == 3:
                parameters_to_prune = []
                for layer, module in enumerate(model.features):
                    if layer == 1 or layer == 3 or layer == 4 or layer == 6 or layer == 7 or layer == 9 \
                            or layer == 10 or layer == 12 or layer == 13 or layer == 18 or layer == 16\
                            or layer ==19 or layer ==22:
                        parameters_to_prune.append((module, 'weight'))
            #print(parameters_to_prune)
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=global_sparsity,
            )
        return model

    def prunning_model(self, model, global_sparsity):
        with torch.no_grad():
            parameters_to_prune = []
            for layer, (name, module) in enumerate(model._modules.items()):
                # print(layer, (name, module))
                if name == "layer":
                    for layer2, (name2, module2) in enumerate(module._modules.items()):
                        parameters_to_prune.append((module2.conv1, 'weight'))
                        parameters_to_prune.append((module2.conv2, 'weight'))
                        parameters_to_prune.append((module2.bn1, 'weight'))
                        parameters_to_prune.append((module2.bn2, 'weight'))
                if name == "fc":
                    parameters_to_prune.append((module, 'weight'))
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=global_sparsity,
            )



        return model

