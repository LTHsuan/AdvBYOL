"""adversary.py"""
import torch
import numpy as np
import torch.nn as nn
from solo.attack.utils import loss_adv

class FGSM(object):
    def __init__(self, net, p, eps, data_name, target, loss):
        self.net = net
        self.eps = eps
        self.p = p
        self.target = target
        self.data_name = data_name
        self.loss = loss
        #self.device = device
        if self.data_name=="cifar10" and self.target:
            raise AssertionError('cifar10 dont support targeted attack')

    
    def forward(self, images, labels,target_labels):
        # print(len(images))
        # print(images.shape, labels.shape, target_labels.shape)
        batchsize = images.shape[0]
        #images, labels = images.to(self.device), labels.to(self.device)
        #print(target_labels)
        if target_labels is not None:
            target_labels = target_labels
            #target_labels = target_labels.to(self.device)
        advimage = images.clone().detach().requires_grad_(True)
        #advimage = images.clone().detach().requires_grad_(True).to(self.device)
        outputs = self.net(advimage)

        loss = loss_adv(self.loss, outputs, labels, target_labels, self.target) 
        #loss = loss_adv(self.loss, outputs, labels, target_labels, self.target, self.device) 
             
        updatas = torch.autograd.grad(loss, [advimage])[0].detach()
        #print(updatas)

        if self.p == np.inf:
            updatas = updatas.sign()
        else:
            normval = torch.norm(updatas.view(batchsize, -1), self.p, 1)
            updatas = updatas / normval.view(batchsize, 1, 1, 1)
        
        advimage = advimage + updatas*self.eps
        delta = advimage - images

        if self.p==np.inf:
            delta = torch.clamp(delta, -self.eps, self.eps)
        else:
            normVal = torch.norm(delta.view(batchsize, -1), self.p, 1)
            mask = normVal<=self.eps
            scaling = self.eps/normVal
            scaling[mask] = 1
            delta = delta*scaling.view(batchsize, 1, 1, 1)
        advimage = images+delta
        
        advimage = torch.clamp(advimage, 0, 1)

        outputs = self.net(advimage)
        adv_labels = torch.argmax(outputs, axis=-1)
        #print("FGSM:", torch.argmax(outputs, axis=-1),labels)
        test_acc = (torch.argmax(self.net(images), axis=-1)== labels).cpu().numpy()
        success = (adv_labels != labels).cpu().numpy()
        print("FGSM:{}/{}".format(np.sum(success), len(success)))
        print("Original:{}/{}".format(np.sum(test_acc), len(test_acc)))
        return advimage
