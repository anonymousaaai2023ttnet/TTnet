import warnings

from src.nn.evaluateur import Evaluateur
from src.nn.trainer import Trainer
import copy

import torch.nn as nn
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import torch

from src.nn.model_general_ICRL import model_general
from src.utils.config import Config, init_all_for_run
from src.utils.config import str2bool, two_args_str_int, two_args_str_float, str2list, \
    transform_input_filters, transform_input_lr, transform_input_eps

import torchvision
import torchvision.transforms as transforms

from src.utils.count import measure_model
from src.utils.utils import concat

config_general = Config(path="./config/")

if config_general.dataset=="CIFAR10":
    config = Config(path="./config/cifar10/")
elif config_general.dataset=="MNIST":
    config = Config(path="./config/mnist/")
else:
    raise 'PB'

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# initiate the parser


parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=config.general.seed, type=two_args_str_int, choices=[i for i in range(100)])
parser.add_argument("--device", default=config.general.device, choices=["cuda", "cpu"])
parser.add_argument("--device_ids", default=config.general.device_ids, type=str2list)
parser.add_argument("--models_path", default=config.general.models_path)
parser.add_argument("--num_workers", default=config.general.num_workers, type=int)

parser.add_argument("--filters", default=config.model.filters, type=transform_input_filters)
parser.add_argument("--kernelsizes", default=config.model.kernelsizes, type=transform_input_filters)
parser.add_argument("--amplifications", default=config.model.amplifications, type=transform_input_filters)
parser.add_argument("--strides", default=config.model.strides, type=transform_input_filters)
parser.add_argument("--fc", default=config.model.fc, type=int)
parser.add_argument("--nchannel", default=config.model.nchannel, type=int)
parser.add_argument("--groups", default=config.model.groups, type=transform_input_filters)
parser.add_argument("--g_remove_last_bn", default=config.model.g_remove_last_bn)
parser.add_argument("--step_quantization", default=config.model.step_quantization, type=transform_input_eps)

parser.add_argument("--adv_epsilon", default=config.train.adv_epsilon)
parser.add_argument("--batch_size_train", default=config.train.batch_size_train, type=int)
parser.add_argument("--n_epoch", default=config.train.n_epoch, type=two_args_str_int)
parser.add_argument("--loss_type", default=config.train.loss_type, type=two_args_str_int)
parser.add_argument("--optimizer_type", default=config.train.optimizer_type, type=two_args_str_int)
parser.add_argument("--weight_decay", default=config.train.weight_decay, type=two_args_str_float)
parser.add_argument("--lr", default=config.train.lr, type=transform_input_lr)
parser.add_argument("--epochs_lr", default=config.train.epochs_lr, type=transform_input_lr)
parser.add_argument("--clip_grad_norm", default=config.train.clip_grad_norm, type=two_args_str_float)
parser.add_argument("--a_bit_final", default=config.train.a_bit_final, type=two_args_str_float)
parser.add_argument("--l1_coef", default=config.train.l1_coef, type=two_args_str_float)
parser.add_argument("--l1_reg", default=config.train.l1_reg, type=str2bool)


parser.add_argument("--batch_size_test", default=config.eval.batch_size_test, type=two_args_str_int)
parser.add_argument("--pruning", default=config.eval.pruning, type=str2bool)
parser.add_argument("--coef_mul", default=config.eval.coef_mul, type=two_args_str_int)


parser.add_argument("--coef", default=config.eval_with_sat.coef, type=two_args_str_int)

#


args = parser.parse_args()

device = torch.device("cuda:" + str(args.device_ids[0]) if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([
    transforms.ToTensor()])
transform_test = transforms.Compose([
    transforms.ToTensor()])

if config_general.dataset == "MNIST":
    trainset = torchvision.datasets.MNIST("~/datasets/mnist", transform=transform_train, train=True, download=True)
    testset = torchvision.datasets.MNIST("~/datasets/mnist", transform=transform_test, train=False, download=True)
elif config_general.dataset == "CIFAR10":
    if args.nchannel == 1:
        transform_train = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()])
        transform_test = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()])
    elif args.nchannel == 6:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            concat()])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            concat()])
    trainset = torchvision.datasets.CIFAR10("~/datasets/CIFAR10", transform=transform_train, train=True, download=True)
    testset = torchvision.datasets.CIFAR10("~/datasets/CIFAR10", transform=transform_test, train=False, download=True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=(args.batch_size_train), shuffle=True,
                                          num_workers=(args.num_workers))
testloader = torch.utils.data.DataLoader(testset, batch_size=(args.batch_size_test), shuffle=False,
                                         num_workers=(args.num_workers))

dataloaders = {'train': trainloader, "val": testloader}


model = model_general(args)

print(model)

if config_general.dataset == "CIFAR10":
    f, c = measure_model(model, args.nchannel, 32, 32)
else:
    f, c = measure_model(model, 1, 28, 28)

print()
print("Size model")
print("model size %.4f M, ops %.4f M" % (c / 1e6, f / 1e6))
print()

model.to(device)

writer = SummaryWriter(args.models_path)

trainer = Trainer(model, dataloaders, args, writer, device, args.models_path)
print("START TRAINING")
trainer.train()
print()



def main():
    pass

if __name__ == '__main__':
    main()
