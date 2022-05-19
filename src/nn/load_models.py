import copy
import os
import numpy as np
import torch.nn.functional as F
from eevbnn.utils import ModelHelper
import torch
import torchvision.transforms as transforms
from src.nn.model_general_ICRL import model_general
import torchvision
from src.utils.utils import concat

def view(x):
    return x.view(1, x.size(0))


def load_models_binary_prunnig(args, quantized_model_train, path_save_model, device, feature_pos, evaluateur):
    state = torch.load(path_save_model, map_location=torch.device('cpu'))["memory_prunning"]
    _, netprunned = prunning_model(args, evaluateur, quantized_model_train, state, device, feature_pos)
    return netprunned

def load_models_binary( args, path_save_model, device):
    model = (ModelHelper.
             create_with_load(path_save_model + "/last.pth").
             to(device).
             eval())
    model_train = copy.deepcopy(model)
    model.to(device)
    model_train.to(device)

    dataloaders, testset = load_data(args)

    feature_pos = 17 + 6 * (len(args.groups)-2)
    scales = []
    biass = []
    for i, data in enumerate(dataloaders["val"]):
        inputs, labels = data
        # print(inputs.shape)
        _ = model(inputs.to(device))

        var = model_train.features[feature_pos].running_var
        mean = model_train.features[feature_pos].running_mean
        scale, bias = model_train.features[feature_pos]._get_scale_bias(var, mean)
        scales.append(scale.item())
        biass.append(bias.detach().cpu().numpy())
    bias = torch.Tensor(np.mean(biass, axis = 0))
    scale = np.mean(scales)
    #print(scale, bias)
    #coef = args.coef
    coef_scale = scale#.detach().clone().item()
    if coef_scale>1:
        coef = 100 #int(round(100/coef_scale))
    elif 1>coef_scale:
        coef = 100 #int(round(10/coef_scale))
    #print(coef)
    Wbin = model_train.features[feature_pos - 1].weight_bin.data
    biais = F.relu_(-Wbin.view(Wbin.size(0), -1)).sum(dim=1).cpu().clone().detach().numpy()
    Wbin2 = 1.0 * Wbin.cpu() * (coef * scale).round()#).cpu().clone().detach().numpy()
    biais2 = 1.0 * biais * (coef * scale) + 1.0 * view(
        (coef * bias)).cpu().clone().detach().numpy()

    #print(np.unique(Wbin2))

    #Wmask = model_train.features[feature_pos - 1].weight_mask.detach().clone().numpy()
    #Wbin3 = np.zeros((Wbin2.shape[0], Wbin2.shape[1]))
    #for x in range(Wbin2.shape[0]):
    #    for y in range(Wbin2.shape[1]):
    #        if Wmask[x][y] == 1:
    #            Wbin3[x][y] = Wbin2[x][y]

    model_train.features[feature_pos - 1] = torch.nn.Linear(Wbin2.shape[0], Wbin2.shape[1], bias=True)
    model_train.features[feature_pos - 1].weight.data = torch.Tensor(Wbin2).to(device)
    # print(model_train.features[feature_pos - 1].bias.data.shape)
    model_train.features[feature_pos - 1].bias.data = torch.Tensor(biais2).to(device)
    model_train.features = model_train.features[:-1]

    quantized_model_train = copy.deepcopy(model_train)

    return quantized_model_train.eval(), feature_pos


def load_models_real( args, path_save_model, device):
    model = model_general(args).to(device)
    model.load_state_dict(torch.load(path_save_model + "/last.pth", map_location=device)["state_dict"], strict=True)
    model_train = copy.deepcopy(model)
    quantized_model = torch.quantization.quantize_dynamic(
        model_train, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
    )
    quantized_model_train = copy.deepcopy(quantized_model)
    Wint = quantized_model_train.fc.weight().int_repr().detach().cpu().clone().numpy()
    Wintbis = np.zeros((Wint.shape[0], Wint.shape[1]))
    for x in range(Wint.shape[0]):
        for y in range(Wint.shape[1]):
            Wintbis[x][y] = Wint[x][y]
    quantized_model_train.fc = torch.nn.Linear(Wintbis.shape[0], Wintbis.shape[1], bias=False)
    quantized_model_train.fc.weight.data = torch.Tensor(Wintbis).to(device)

    return quantized_model_train.eval()

def load_data(args):
    transform_test = transforms.Compose([
        transforms.ToTensor()])

    if args.dataset == "MNIST":
        testset = torchvision.datasets.MNIST("~/datasets/mnist", transform=transform_test, train=False, download=True)
    elif args.dataset == "CIFAR10":
        if args.nchannel == 1:
            transform_test = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()])
        elif args.nchannel == 6:
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                concat()])
        testset = torchvision.datasets.CIFAR10("~/datasets/CIFAR10", transform=transform_test, train=False,
                                               download=True)
    elif args.dataset == "Tiny":

        data_dir = './eevbnn/data/tiny-imagenet-200/'
        if args.nchannel == 1:
            data_transforms = {
                'train': transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.RandomRotation(20),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                ]),
                'val': transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                ]),
                'test': transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                ])
            }
        elif args.nchannel == 6:
            data_transforms = {
                'train': transforms.Compose([
                    transforms.RandomRotation(20),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                    concat()
                ]),
                'val': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                    concat()
                ]),
                'test': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                    concat()
                ])
            }
        else:
            data_transforms = {
                'train': transforms.Compose([
                    transforms.RandomRotation(20),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                ]),
                'val': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                ]),
                'test': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                ])
            }

        testset = torchvision.datasets.ImageFolder(os.path.join(data_dir, "test"), data_transforms["test"])


    testloader = torch.utils.data.DataLoader(testset, batch_size=(args.batch_size_test), shuffle=False,
                                             num_workers=(args.num_workers))

    dataloaders = {"val": testloader}
    return dataloaders, testset

def load_data_train(args):
    transform_test = transforms.Compose([
        transforms.ToTensor()])

    if args.dataset == "MNIST":
        testset = torchvision.datasets.MNIST("~/datasets/mnist", transform=transform_test, train=True, download=True)
    elif args.dataset == "CIFAR10":
        if args.nchannel == 1:
            transform_test = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()])
        elif args.nchannel == 6:
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                concat()])
        testset = torchvision.datasets.CIFAR10("~/datasets/CIFAR10", transform=transform_test, train=True,
                                               download=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size=(args.batch_size_test), shuffle=False,
                                             num_workers=(args.num_workers))

    dataloaders = {"val": testloader}
    return dataloaders, testset


def prunning_model(config_general, evaluateur, quantized_model_train, prunning, device, feature_pos = None):
    if config_general.type == "binary":
        netprunned = evaluateur.prunning_model_v2(copy.deepcopy(quantized_model_train), prunning)
        W = netprunned.features[feature_pos - 1].weight.detach().cpu().clone().numpy()
        Wmask = netprunned.features[feature_pos - 1].weight_mask.detach().cpu().clone().numpy()
        biaisivi = netprunned.features[feature_pos - 1].bias.detach().cpu().clone().numpy()
    elif config_general.type == "real":
        netprunned = evaluateur.prunning_model(copy.deepcopy(quantized_model_train), prunning)
        W = netprunned.fc.weight.detach().cpu().clone().numpy()
        Wmask = netprunned.fc.weight_mask.detach().cpu().clone().numpy()

    W_prunned = np.zeros((W.shape[0], W.shape[1]))
    for x in range(W.shape[0]):
        for y in range(W.shape[1]):
            if Wmask[x][y] == 1:
                W_prunned[x][y] = W[x][y]

    if config_general.type == "real":
        netprunned.fc = torch.nn.Linear(W.shape[0], W.shape[1], bias=False)
        netprunned.fc.weight.data = torch.Tensor(W_prunned).to(device)
    elif config_general.type == "binary":
        netprunned.features[feature_pos - 1] = torch.nn.Linear(W.shape[0], W.shape[1], bias=True)
        netprunned.features[feature_pos - 1].weight.data = torch.Tensor(W_prunned).to(device)
        netprunned.features[feature_pos - 1].bias.data = torch.Tensor(biaisivi).to(device)

    acc = evaluateur.eval(netprunned, ["val"])

    return acc, netprunned
