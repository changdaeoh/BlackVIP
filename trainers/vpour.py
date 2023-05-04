import os.path as osp
from this import d
import time
import datetime
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
import os
import matplotlib.pyplot as plt
from my_dassl.engine import TRAINER_REGISTRY, TrainerX
from my_dassl.metrics import compute_accuracy
from my_dassl.utils import load_pretrained_weights, load_checkpoint, set_random_seed, AverageMeter, MetricMeter, mkdir_if_missing
from my_dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from trainers import visual_prompters
from trainers.utils import clip_clipping, load_clip_to_cpu
import numpy as np
import pdb
import wandb
from trainers.zsclip import CUSTOM_TEMPLATES

from math import sqrt
_tokenizer = _Tokenizer()


class CustomCLIP(nn.Module):
    '''editted for visual prompting'''
    def __init__(self, cfg, classnames, clip_model, device):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Text Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        self.n_classes = len(classnames)
        self.classnames = classnames
        
        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features.to(device)

        self.prompter = visual_prompters.__dict__[cfg.TRAINER.VPOUR.METHOD](cfg)

    def forward(self, image):
        prompted_images  = self.prompter(image.type(self.dtype))
        image_features = self.image_encoder(prompted_images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits


@TRAINER_REGISTRY.register()
class VPOUR(TrainerX):
    """Visual Prompting in Black-Box setting
    """
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model = clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, self.device)

        #! blackbox setting
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompter, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model.prompter, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompter", self.model.prompter, self.optim, self.sched)

        self.N_params = len(torch.nn.utils.parameters_to_vector(self.model.prompter.parameters()))
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

        #! SPSA hyparams
        self.o, self.c, self.a, self.alpha, self.gamma = cfg.TRAINER.VPOUR.SPSA_PARAMS
        self.opt_type                                  = cfg.TRAINER.VPOUR.OPT_TYPE
        self.b1                                        = cfg.TRAINER.VPOUR.MOMS
        self.sp_avg                                    = cfg.TRAINER.VPOUR.SP_AVG
        self.step = 0                                   
        self.m1 = 0

        self.loss_fn = F.cross_entropy

    def forward_backward(self, batch):
        with torch.no_grad():
            image, label = self.parse_batch_train(batch)
            with autocast(): #! for fast training
                #* SPSA scheduling
                ak = self.a/((self.step + self.o)**self.alpha)
                ck = self.c/(self.step**self.gamma)
                
                # gradient estimation
                w = torch.nn.utils.parameters_to_vector(self.model.prompter.parameters())
                ghat, loss, acc = self.spsa_grad_estimate_bi(w, image, label, ck)
    
                if self.opt_type == 'spsa-gc':
                    if self.step > 1:  self.m1 = self.b1*self.m1 + ghat
                    else:              self.m1 = ghat
                    accum_ghat = ghat + self.b1*self.m1
                elif self.opt_type == 'spsa':
                    accum_ghat = ghat
                else:
                    raise ValueError

                #* param update
                w_new = w - ak * accum_ghat
                torch.nn.utils.vector_to_parameters(w_new, self.model.prompter.parameters())
            
        loss_summary = {"loss": loss,"acc": acc,}
        if self.cfg.use_wandb: wandb.log({'train_ep_acc':acc, 'train_ep_loss':loss.item(), 'gain_seq':ak})
        return loss_summary

    def spsa_grad_estimate_bi(self, w, image, label, ck):
        #* repeat k times and average them for stabilizing
        ghats = []
        for spk in range(self.sp_avg):
            #! Bernoulli {-1, 1}
            # perturb = torch.bernoulli(torch.empty(self.N_params).uniform_(0,1)).cuda()
            # perturb[perturb < 1] = -1
            #! Segmented Uniform [-1, 0.5] U [0.5, 1]
            p_side = (torch.rand(self.N_params).reshape(-1,1) + 1)/2
            samples = torch.cat([p_side,-p_side], dim=1)
            perturb = torch.gather(samples, 1, torch.bernoulli(torch.ones_like(p_side)/2).type(torch.int64)).reshape(-1).cuda()
            del samples; del p_side

            #* two-side Approximated Numerical Gradient
            w_r = w + ck*perturb
            w_l = w - ck*perturb
            torch.nn.utils.vector_to_parameters(w_r, self.model.prompter.parameters())
            output1 = self.model(image)
            torch.nn.utils.vector_to_parameters(w_l, self.model.prompter.parameters())
            output2 = self.model(image)
            loss1 = self.loss_fn(output1, label)
            loss2 = self.loss_fn(output2, label)

            #* parameter update via estimated gradient
            ghat = (loss1 - loss2)/((2*ck)*perturb)
            ghats.append(ghat.reshape(1, -1))
        if self.sp_avg == 1: pass
        else: ghat = torch.cat(ghats, dim=0).mean(dim=0) 
        loss = ((loss1+loss2)/2)
        acc = ((compute_accuracy(output1, label)[0]+
                compute_accuracy(output2, label)[0])/2).item()

        return ghat, loss, acc

    def train(self):
        """Generic training loops."""
        self.before_train()
        set_random_seed(self.cfg.SEED) #! required to reproduce SPSA result
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)
        self.total_length = self.num_batches * self.max_epoch
        self.warmup_length = self.total_length * 0.1

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            self.step += 1
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)
            end = time.time()

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        # if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
        #     curr_result = self.test(split="val")
        #     is_best = curr_result > self.best_result
        #     if is_best:
        #         self.best_result = curr_result
        #         self.save_model(
        #             self.epoch,
        #             self.output_dir,
        #             model_name="model-best.pth.tar"
        #         )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)
        
        # wandb.log({'val_ep_acc':curr_result})

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label