from copy import deepcopy

import torch
from torch import optim #import optim for our_train
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data


def variable(t: torch.Tensor, use_cuda=False, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    # Here we will store the optimal solution for a given task. 
    def __init__(self, model: nn.Module, dataset: torch.utils.data.DataLoader, gpu: torch.device):

        self.model = model
        self.dataset = dataset
        self.gpu = gpu

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = Variable(p.data).cuda(self.gpu)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = Variable(p.data).cuda(self.gpu)

        self.model.eval()
        for input, target in self.dataset:
            self.model.zero_grad()
            input, target = Variable(input).cuda(self.gpu), Variable(target).cuda(self.gpu)
            output = self.model(input).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss

class splitEWC(object):
    # Here we will store the optimal solution for a given task. 
    def __init__(self, model: nn.Module, dataset: torch.utils.data.DataLoader, cut_idx: int, gpu: torch.device):

        self.model = model
        self.dataset = dataset
        self.cut_idx = cut_idx

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher(gpu)

        for n, p in deepcopy(self.params).items():
            self._means[n] = Variable(p.data).cuda(gpu)

    def _diag_fisher(self, gpu):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = Variable(p.data).cuda(gpu)

        self.model.eval()
        for input, target in self.dataset:
            self.model.zero_grad()
            input, target = Variable(input).cuda(gpu), Variable(target).cuda(gpu)
            output = self.model(input).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)    #自定义step 

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for idx, (n, p) in enumerate(model.named_parameters()):
            if idx < self.cut_idx:
                continue
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        # print('penalty:', loss)
        return loss

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()

def normal_train(model: nn.Module, labels: list, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader, gpu: torch.device):
    model.train()
    model.apply(set_bn_eval)  #冻结BN及其统计数据
    epoch_loss = 0
    for data, target in data_loader:
        data, target = Variable(data).cuda(gpu), Variable(target).cuda(gpu)
        optimizer.zero_grad()
        output = model(data)
        for idx in range(output.size(1)):
            if idx not in labels:
                output[range(len(output)), idx] = 0
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)


def ewc_train(model: nn.Module, labels: list, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader, ewcs: list, lam: float, gpu: torch.device):
    model.train()
    model.apply(set_bn_eval) #冻结BN及其统计数据
    epoch_loss = 0
    for data, target in data_loader:
        data, target = Variable(data).cuda(gpu), Variable(target).cuda(gpu)
        optimizer.zero_grad()
        output = model(data)
        for idx in range(output.size(1)):
            if idx not in labels:
                output[range(len(output)), idx] = 0
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target) 
        # print('loss:', loss.item())
        for ewc in ewcs:
            loss += (lam / 2) * ewc.penalty(model)
            # print('ewc loss:', loss.item())
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)

def our_train(model: nn.Module, labels: list, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader, ewcs: list, lam: float, gpu: torch.device, cut_idx, if_freeze, args_lr):
    model.train()
    model.apply(set_bn_eval) #冻结BN及其统计数据

    nofreezegrad = [] #存储全部不冻结的数据grad
    nofreezegrad2 = []
    #---------------不冻结server&先不加ewc------
    epoch_loss = 0
    for data, target in data_loader:
        data, target = Variable(data).cuda(gpu), Variable(target).cuda(gpu)
        optimizer.zero_grad()
        output = model(data)
        for idx in range(output.size(1)):
            if idx not in labels:
                output[range(len(output)), idx] = 0
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target) 
        # print('loss:', loss.item())
        # for ewc in ewcs:
        #     loss += (lam / 2) * ewc.penalty(model)
        #     # print('ewc loss:', loss.item())
        # epoch_loss += loss.item()

        loss.backward()
        for idx, p in enumerate(model.parameters()):
            # if idx >= cut_idx:
            #     continue
            layer_g_tmp = torch.zeros_like(p.grad.data).cuda(gpu)
            layer_g_tmp += p.grad.data
            nofreezegrad.append(layer_g_tmp)

    #不step,再算一次grad

    #还需要进行loss判断，true：freeze
    #---------------------freeze
    if if_freeze == 1 :
        for idx, param in enumerate(model.parameters()):
            if idx >= cut_idx:
                continue
            param.requires_grad = False

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args_lr) # no need to add: lr=0.1?
    #----------------------


    epoch_loss = 0
    for data, target in data_loader:
        data, target = Variable(data).cuda(gpu), Variable(target).cuda(gpu)
        optimizer.zero_grad()
        output = model(data)
        for idx in range(output.size(1)):
            if idx not in labels:
                output[range(len(output)), idx] = 0
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target) 
        # print('loss:', loss.item())
        # for ewc in ewcs:
        #     loss += (lam / 2) * ewc.penalty(model)
        #     # print('ewc loss:', loss.item())
        epoch_loss += loss.item()

        loss.backward()
        equalresult=[]
        for idx, p in enumerate(model.parameters()):
            # if idx >= cut_idx:
            #     continue
            layer_g_tmp = torch.zeros_like(p.grad.data).cuda(gpu)
            layer_g_tmp += p.grad.data
            nofreezegrad2.append(layer_g_tmp)
            equalresult.append(torch.equal(nofreezegrad[idx], nofreezegrad2[idx]))
        print('torch.equal_result:')
        print(equalresult)        

        optimizer.step()   #optimizer.param_groups : 'params' : .grad ==> 梯度 
                           #91行
                           #for n, p in model.named_parameters():
                           #    p.grad.data ==> 当前网络层梯度数据？
    #-----------------------------解冻
    if if_freeze == 1 :    
        for idx, param in enumerate(model.parameters()):
            if idx >= cut_idx:
                continue
            param.requires_grad = True

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args_lr) # no need to add: lr=0.1?
    #-------------------------------
    return epoch_loss / len(data_loader)

def test_model(model, labels, test_data, gpu):
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for data, target in test_data:
            data, target = Variable(data).cuda(gpu), Variable(target).cuda(gpu)
            output = model(data)
            for idx in range(output.size(1)):
                if idx not in labels:
                    output[range(len(output)), idx] = 0
            # get the index of the max log-probability
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    acc = correct / total
    return acc