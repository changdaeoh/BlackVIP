"""
    source1(VP): https://github.com/hjbahng/visual_prompting
    source2(AR): https://github.com/savan77/Adversarial-Reprogramming 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter 
from torchvision import transforms
import numpy as np
import os

from transformers import ViTFeatureExtractor, ResNetModel, ViTModel, ViTMAEModel, ViTForImageClassification
import wandb
from PIL import Image
from trainers.utils import temperatured_sig, temperatured_tanh, clip_normalization, clip_clipping, vis_p_vis
from collections import defaultdict

# inverting RGB transform of CLIP
inv_normalize = transforms.Normalize(
                                    mean=[-0.48145466/0.26862954, 
                                          -0.4578275/0.26130258, 
                                          -0.40821073/0.27577711],
                                    std=[1/0.26862954, 
                                         1/0.26130258, 
                                         1/0.27577711]
                                    )

class PadPrompter(nn.Module):
    def __init__(self, args):
        super(PadPrompter, self).__init__()
        self.args = args
        if args.TRAINER.NAME == 'VPWB':
            pad_size = args.TRAINER.VPWB.PROMPT_SIZE
            image_size = args.TRAINER.VPWB.IMAGE_SIZE
        elif args.TRAINER.NAME == 'VPOUR':
            pad_size = args.TRAINER.VPOUR.PROMPT_SIZE
            image_size = args.TRAINER.VPOUR.IMAGE_SIZE
        else: raise ValueError

        self.base_size = image_size - pad_size*2
        self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))

    def forward(self, x):
        n_samples = x.shape[0]
        base = torch.zeros(1, 3, self.base_size, self.base_size).to(x.device)
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])
        x_prompted = x + prompt
        return x_prompted


class PROGRAM(nn.Module):
    def __init__(self, args):
        super(PROGRAM, self).__init__()
        self.frame_size = args.TRAINER.BAR.FRAME_SIZE #! image size (e.g. 224)
        self.image_size = args.INPUT.SIZE[0] #! embedded image size
        self.args = args
        if args.USE_CUDA: self.dev = 'cuda'
        else: self.dev = 'cpu'
        self.W = Parameter(torch.randn(3, self.frame_size, self.frame_size))
        self.mask()
        self.set_mean_std()

    def forward(self, target_data):
        n_samples = target_data.shape[0]
        norm_tdata = inv_normalize(target_data) # => 0 ~ 1
        
        X = norm_tdata.data.new(n_samples, 3, self.frame_size, self.frame_size)
        X[:] = 0
        X[:,:,self.h_lower:self.h_upper,self.w_lower:self.w_upper]=norm_tdata.data.clone()
        P=torch.sigmoid(self.W) * self.M
        X_adv=X+P.repeat(n_samples, 1, 1, 1)
        X_adv_norm=(X_adv-self.mean)/self.std # => CLIP transform
        return X_adv_norm.float()
    
    def mask(self):
        m = torch.ones(3, self.frame_size, self.frame_size)
        x_center, y_center = self.frame_size//2, self.frame_size//2
        self.h_lower = y_center - (self.image_size//2)
        self.h_upper = y_center + (self.image_size//2)	
        self.w_lower = x_center - (self.image_size//2)
        self.w_upper = x_center + (self.image_size//2)
        m[:, self.h_lower:self.h_upper, self.w_lower:self.w_upper] = 0
        self.M = m.to(self.dev)
    
    def set_mean_std(self):
        mean=np.array([0.48145466,0.4578275,0.40821073]).reshape(3,1,1)
        std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(3,1,1)
        self.mean = torch.from_numpy(mean).to(self.dev)
        self.std = torch.from_numpy(std).to(self.dev)


class CoordinatorINIT(nn.Module):
    def __init__(self, args):
        super(CoordinatorINIT, self).__init__()
        self.args = args
        
        act = nn.GELU #if args.TRAINER.BLACKVIP.ACT == 'gelu' else nn.ReLU
        e_out_dim = args.TRAINER.BLACKVIP.E_OUT_DIM
        src_dim = args.TRAINER.BLACKVIP.SRC_DIM

        self.enc = EncoderManual(e_out_dim, act=act, gap=False)
        self.dec = DecoderManual(0, src_dim=e_out_dim, act=act, arch='vit-base')
    
    def forward(self, x):
        z = self.enc(x)
        wrap = self.dec(z)
        return wrap, z


class Coordinator(nn.Module):
    def __init__(self, args):
        super(Coordinator, self).__init__()
        self.args = args        
        self.backbone = args.TRAINER.BLACKVIP.PT_BACKBONE
        act = nn.GELU #if args.TRAINER.BLACKVIP.ACT == 'gelu' else nn.ReLU
        src_dim = args.TRAINER.BLACKVIP.SRC_DIM

        z_dim = 768
        if self.backbone == 'vit-mae-base':   #! SSL-MAE VIT-B (n param: 86M)
            self.enc_pt = ViTForImageClassification.from_pretrained("facebook/vit-mae-base")
        elif self.backbone == 'vit-base':       #! SUP VIT-B
            self.enc_pt = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        elif self.backbone == 'dino-resnet-50': #! SSL-DINO RN50 (n param: 23M)
            self.enc_pt = ResNetModel.from_pretrained("Ramos-Ramos/dino-resnet-50")
            z_dim = 2048
        else: raise ValueError('not implemented')

        self.dec = DecoderManual(z_dim, src_dim, act=act, arch=self.backbone)

    def forward(self, x):
        with torch.no_grad():
            if self.backbone == 'vit-mae-base':
                #! (N, 197, 768) => pick [CLS] => (N, 768)
                out = self.enc_pt(x, output_hidden_states=True)
                z = out.hidden_states[-1][:,0,:]
            elif self.backbone == 'vit-base':
                #! (N, 197, 768) => pick [CLS] => (N, 768)
                out = self.enc_pt(x)
                z = out.last_hidden_state[:,0,:]
            elif self.backbone == 'dino-resnet-50':
                #! (N, 2048, 7, 7) => pool => (N, 2048)
                out_temp = self.enc_pt(x)
                zdim_ = out_temp.last_hidden_state.shape[1]
                out = out_temp.pooler_output.reshape(-1, zdim_)
                z = out
            else: raise ValueError
        
        wrap = self.dec(z)
        return wrap, z


class DecoderManual(nn.Module):
    def __init__(self, i_dim, src_dim, act=nn.GELU, arch='vit-base'):
        super(DecoderManual, self).__init__()
        if i_dim: self.shared_feature = 1
        else:     self.shared_feature = 0
        if self.shared_feature:
            #! start from 7*7*16(784:16) or 7*7*32(1568:800) or 7*7*64(3,136:2368)
            if (src_dim % 49) != 0: raise ValueError('map dim must be devided with 7*7')
            self.p_trigger = torch.nn.Parameter(torch.Tensor(1, src_dim - i_dim))
            torch.nn.init.uniform_(self.p_trigger, a=0.0, b=0.1) # can be tuned
            src_c = src_dim // 49
        else:
            src_c = src_dim
        
        bias_flag = False
        body_seq = []
        
        if arch in ['vit-mae-base', 'vit-base']:
            if src_c >= 64:    g_c = 64
            else:              g_c = src_c
            body_seq              +=  [nn.ConvTranspose2d(src_c, 64, 2, 2, 0, groups=g_c),
                                       nn.ConvTranspose2d(64, 64, kernel_size=1, bias=bias_flag)]
            body_seq              +=  [nn.BatchNorm2d(64), act()]
            body_seq              +=  [nn.ConvTranspose2d(64, 64, 2, 2, 0, groups=64),
                                       nn.ConvTranspose2d(64, 32, kernel_size=1, bias=bias_flag)]
            body_seq              +=  [nn.BatchNorm2d(32), act()]
            body_seq              +=  [nn.ConvTranspose2d(32, 32, 2, 2, 0, groups=32),
                                       nn.ConvTranspose2d(32, 32, kernel_size=1, bias=bias_flag)]
            body_seq              +=  [nn.BatchNorm2d(32), act()]
            body_seq              +=  [nn.ConvTranspose2d(32, 32, 2, 2, 0, groups=32),
                                       nn.ConvTranspose2d(32, 16, kernel_size=1, bias=bias_flag)]
            body_seq              +=  [nn.BatchNorm2d(16), act()]
            body_seq              +=  [nn.ConvTranspose2d(16, 3, 2, 2, 0, bias=bias_flag)]  
        elif arch == 'dino-resnet-50':
            body_seq              +=  [nn.ConvTranspose2d(src_c, 64, 2, 2, 0, groups=32),
                                       nn.ConvTranspose2d(64, 64, kernel_size=1, bias=bias_flag)]
            body_seq              +=  [nn.BatchNorm2d(64), act()]
            body_seq              +=  [nn.ConvTranspose2d(64, 64, 2, 2, 0, groups=64),
                                       nn.ConvTranspose2d(64, 32, kernel_size=1, bias=bias_flag)]
            body_seq              +=  [nn.BatchNorm2d(32), act()]            
            body_seq              +=  [nn.ConvTranspose2d(32, 32, 2, 2, 0, groups=32),
                                       nn.ConvTranspose2d(32, 32, kernel_size=1, bias=bias_flag)]
            body_seq              +=  [nn.BatchNorm2d(32), act()]
            body_seq              +=  [nn.ConvTranspose2d(32, 32, 2, 2, 0, groups=32),
                                       nn.ConvTranspose2d(32, 16, kernel_size=1, bias=bias_flag)]
            body_seq              +=  [nn.BatchNorm2d(16), act()]
            body_seq              +=  [nn.ConvTranspose2d(16, 3, 2, 2, 0, bias=bias_flag)]
        else: raise ValueError('not implemented')
        self.body   = nn.Sequential(*body_seq)

    def forward(self, z):
        if self.shared_feature:
            N = z.shape[0]
            D = self.p_trigger.shape[1]
            p_trigger = self.p_trigger.repeat(N, 1)
            z_cube = torch.cat((z, p_trigger), dim=1)
            z_cube = z_cube.reshape(N, -1, 7, 7)
        else:
            return self.body(z)
        return self.body(z_cube)


class EncoderManual(nn.Module):
    def __init__(self, out_dim, act=nn.GELU, gap=False):
        super(EncoderManual, self).__init__()        
        bias_flag = False
        body_seq = []
        body_seq              +=  [nn.Conv2d(3, 32, 3, 1, 1),
                                    nn.Conv2d(32, 32, 2, 2, 0, bias=bias_flag)]
        body_seq              +=  [nn.BatchNorm2d(32), act()]
        body_seq              +=  [nn.Conv2d(32, 32, 3, 1, 1),
                                    nn.Conv2d(32, 64, 2, 2, 0, bias=bias_flag)]
        body_seq              +=  [nn.BatchNorm2d(64), act()]
        body_seq              +=  [nn.Conv2d(64, 64, 3, 1, 1),
                                    nn.Conv2d(64, 64, 2, 2, 0, bias=bias_flag)]
        body_seq              +=  [nn.BatchNorm2d(64), act()]
        body_seq              +=  [nn.Conv2d(64, 64, 3, 1, 1),
                                    nn.Conv2d(64, 128, 2, 2, 0, bias=bias_flag)]
        body_seq              +=  [nn.BatchNorm2d(128), act()]
        body_seq              +=  [nn.Conv2d(128, 128, 3, 1, 1),
                                   nn.Conv2d(128, out_dim, 2, 2, 0, bias=bias_flag)]
        body_seq              +=  [nn.BatchNorm2d(out_dim), act()]
        if gap:     body_seq  +=  [nn.AdaptiveAvgPool2d((1, 1))]
        self.body   = nn.Sequential(*body_seq)

    def forward(self, x):
        return self.body(x)


def padding(args):
    return PadPrompter(args)

def reprogramming(args):
    return PROGRAM(args)

def coordinator_init(args):
    return CoordinatorINIT(args)

def coordinator(args):
    return Coordinator(args)