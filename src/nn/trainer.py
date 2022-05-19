import copy
import os
import time
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from tqdm import tqdm



class Trainer:

    def __init__(self, model, dataloaders, args, writer, device, path_save_model):
        """
        :param args:
        :param writer:
        :param device:
        :param rng:
        :param path_save_model:
        """
        self.args = args
        self.epochs = self.args.n_epoch
        self.batch_size = self.args.batch_size_train
        self.batch_size_test = self.args.batch_size_test
        self.t = Variable(torch.Tensor([0.5]))
        self.writer = writer
        self.device = device
        self.path_save_model = path_save_model
        self.net = model
        self.get_optimizer(args.lr[0])
        if self.args.loss_type == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.dataloaders = dataloaders
        #images, labels = next(iter(self.dataloaders["train"]))
        #grid = torchvision.utils.make_grid(images)
        #self.writer.add_image('images', grid, 0)
        #self.writer.add_graph(self.net, images.to(self.device))

    def get_optimizer(self, lr):

        if self.args.optimizer_type == "AdamW":
            self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.net.parameters()), lr=lr,
                                               weight_decay=self.args.weight_decay)
        if self.args.optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=lr,
                                              weight_decay=self.args.weight_decay)
        if self.args.optimizer_type == "SGD":
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=lr,
                                             momentum=self.args.momentum_nn)

    def train(self):
        since = time.time()
        flag_change_lr = True

        best_model_wts = copy.deepcopy(self.net.state_dict())
        best_loss = 100
        best_acc = 0.0
        n_batches = self.batch_size
        for epoch in range(self.epochs):
            print('-' * 10)
            print('==> %d/%d epoch, previous best: %.3f' % (epoch + 1, self.epochs, best_acc))
            print('-' * 10)
            # Each epoch has a training and validation phase

            if flag_change_lr:
                if epoch > self.args.epochs_lr:
                    self.get_optimizer(self.args.lr[1])
                    flag_change_lr = False



            for phase in ['train', 'val']:
                if phase == 'train':
                    self.net.train()
                    n_batches = self.batch_size
                if phase == 'val':
                    self.net.eval()
                    n_batches = self.batch_size_test
                running_loss = 0.0
                running_loss1 = 0.0
                running_corrects = 0.0
                nbre_sample = 0
                tk0 = tqdm(self.dataloaders[phase], total=int(len(self.dataloaders[phase])))
                for i, data in enumerate(tk0):
                    inputs, labels = data
                    self.optimizer.zero_grad()
                    # forward + backward + optimize
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.net(inputs.to(self.device))
                        loss = self.criterion(outputs.squeeze(1), labels.to(self.device))
                        L1_reg = torch.tensor(0., requires_grad=phase == 'train')
                        if self.args.l1_reg:
                            for name, param in self.net.named_parameters():
                                if 'weight' in name:
                                    #print(name, self.args.l1_coef, param.norm(1), loss)
                                    L1_reg = L1_reg + param.norm(1)
                            #print(ok)
                            loss = loss + self.args.l1_coef * L1_reg
                            loss1 = self.args.l1_coef * L1_reg

                        if phase == 'train':
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip_grad_norm)
                            self.optimizer.step()
                            #if self.scheduler is not None:
                            #    self.scheduler.step()
                        _, predicted = torch.max(outputs.data, 1)
                        running_corrects += (predicted == labels.to(self.device)).sum().item()
                        running_loss += loss.item() * n_batches
                        if self.args.l1_reg:
                            running_loss1 += loss1.item() * n_batches
                        nbre_sample += labels.size(0)

                epoch_loss = running_loss / nbre_sample
                epoch_loss1 = running_loss1 / nbre_sample
                acc = running_corrects / nbre_sample
                print('{} Loss: {:.4f}'.format(
                    phase, epoch_loss))
                print('{} Acc: {:.4f}'.format(
                    phase, acc))
                for param_group in self.optimizer.param_groups:
                    print("LR value:", param_group['lr'])
                print()
                #self.writer.add_scalar(phase + ' Loss ',
                #                       epoch_loss,
                #                       epoch)
                self.writer.add_scalar(phase + ' Acc ',
                                       acc,
                                       epoch)

                self.writer.add_scalars(phase + ' Loss ', {'Global loss': epoch_loss,
                                               'L1 loss': epoch_loss1,
                                               'Cross-entropy loss': epoch_loss - epoch_loss1}, epoch)


                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss

                    torch.save({'epoch': epoch + 1, 'acc': best_loss, 'state_dict': self.net.state_dict()},
                               os.path.join(self.path_save_model, 'last_loss.pth'))
                if phase == 'val' and acc >= best_acc:
                    best_acc = acc
                    best_model_wts = copy.deepcopy(self.net.state_dict())
                    torch.save({'epoch': epoch + 1, 'acc': best_acc, 'state_dict': self.net.state_dict()},
                               os.path.join(self.path_save_model, 'last.pth'))


            print()



        torch.save({'epoch': epoch + 1, 'acc': best_acc, 'state_dict': self.net.state_dict()},
                   os.path.join(self.path_save_model, 'last_epoch.pth'))
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Loss: {:4f}'.format(best_loss))
        print('Best val Acc: {:4f}'.format(best_acc))
        print()
        # load best model weights
        self.best_acc = best_acc
        self.net.load_state_dict(best_model_wts)
