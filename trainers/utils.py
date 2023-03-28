'''
- Focal Loss : https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
- Hinge Loss : https://github.com/HaotianMXu/Multiclass_LinearSVM_with_SGD/blob/master/linearSVM.py
'''

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.autograd as autograd
from torchvision.transforms import Normalize
import numpy as np
from clip import clip


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        #if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class multiClassHingeLoss(nn.Module):
    def __init__(self, p=1, margin=1, weight=None, size_average=True):
        super(multiClassHingeLoss, self).__init__()
        self.p=p
        self.margin=margin
        self.weight=weight#weight for each class, size=n_class, variable containing FloatTensor,cuda,reqiures_grad=False
        self.size_average=size_average
    def forward(self, output, y):#output: batchsize*n_class
        output_y=output[torch.arange(0,y.size()[0]).long().cuda(),y.data.cuda()].view(-1,1)#view for transpose
        #margin - output[y] + output[i]
        loss=output-output_y+self.margin#contains i=y
        #remove i=y items
        loss[torch.arange(0,y.size()[0]).long().cuda(),y.data.cuda()]=0
        #max(0,_)
        loss[loss<0]=0
        #^p
        if(self.p!=1):
            loss=torch.pow(loss,self.p)
        #add weight
        if(self.weight is not None):
            loss=loss*self.weight
        #sum up
        loss=torch.sum(loss)
        if(self.size_average):
            loss/=output.size()[0]#output.size()[0]
        return loss


def temperatured_sig(x, tau=1.0):
    return torch.sigmoid(x / (torch.ones_like(x) * tau))

def temperatured_tanh(x, tau=1.0):
    return torch.tanh(x / (torch.ones_like(x) * tau))

CLIP_NORM_FN = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
def clip_normalization(x):
    return CLIP_NORM_FN(x)

def clip_clipping(x):
    #! -inf ~ inf -> CLIP's input RGB range
    if len(x.shape) == 3:
        out = torch.cat([torch.clip(x[0,:,:], min=-1.79226253, max=1.93033625).unsqueeze(0),
                     torch.clip(x[1,:,:], min=-1.75209713, max=2.07488384).unsqueeze(0),
                     torch.clip(x[2,:,:], min=-1.48021977, max=2.14589699).unsqueeze(0)], dim=0)
    else:
        out = torch.cat([torch.clip(x[:,0,:,:], min=-1.79226253, max=1.93033625).unsqueeze(1),
                        torch.clip(x[:,1,:,:], min=-1.75209713, max=2.07488384).unsqueeze(1),
                        torch.clip(x[:,2,:,:], min=-1.48021977, max=2.14589699).unsqueeze(1)], dim=1)
    return out


def vis_p_vis(prompt, o_clip):
    #! set prompt val range to 0 ~ 1 for valid visualization
    if o_clip == 0:           prompt_for_v = torch.sigmoid(prompt)
    elif o_clip in [1, 2, 3]: prompt_for_v = prompt
    elif o_clip in [4, 5]:    prompt_for_v = (prompt + 1)/2
    else: raise ValueError
    return prompt_for_v


#! CLIP image classifier
class ClassificationHead(nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None: self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None: self.bias = torch.nn.Parameter(biases.clone())
        else: self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize: inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head, logit_scale):
        super().__init__()
        self.image_encoder = image_encoder
        self.clf_head = classification_head
        self.logit_scale = logit_scale

    def forward(self, x):
        z = self.image_encoder(x)
        logits = self.logit_scale.exp() * self.clf_head(z)
        return logits