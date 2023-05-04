import os.path as osp
from termios import VLNEXT
import time
import datetime
from math import sqrt 
import os 

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from my_dassl.engine import TRAINER_REGISTRY, TrainerX
from my_dassl.metrics import compute_accuracy
from my_dassl.utils import load_pretrained_weights, load_checkpoint, set_random_seed, AverageMeter, MetricMeter, mkdir_if_missing
from my_dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from trainers.utils import FocalLoss, load_clip_to_cpu
from trainers import visual_prompters
from tqdm import tqdm
import pdb
import wandb
import numpy as np
from trainers.zsclip import CUSTOM_TEMPLATES
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
        
        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features.to(device)

        self.program = visual_prompters.__dict__[cfg.TRAINER.BAR.METHOD](cfg)

    def forward(self, image):
        programmed_image  = self.program(image.type(self.dtype))
        image_features = self.image_encoder(programmed_image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits


@TRAINER_REGISTRY.register()
class BAR(TrainerX):
    """Black-Box Adversarial Reprogramming
    """
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model = clip_model.float()
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, self.device)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.program, cfg.MODEL.INIT_WEIGHTS)

        #! blackbox setting
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model.program.parameters(), cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("program", self.model.program, self.optim, self.sched)

        #! BAR parameters
        self.init_lr, self.min_lr  = cfg.TRAINER.BAR.LRS
        self.sp_avg                = cfg.TRAINER.BAR.SP_AVG
        self.beta                  = cfg.TRAINER.BAR.SMOOTH
        self.sigma                 = cfg.TRAINER.BAR.SIMGA        
        
        self.step = 0                  
        self.step_for_pdecay = 0
        self.N_params = len(torch.nn.utils.parameters_to_vector(self.model.program.parameters()))

        #! BAR's default loss
        self.loss_fn = FocalLoss(cfg.TRAINER.BAR.FOCAL_G)
        self.tot_itert = 0
        self.best_result = -1.0

    def train(self):
        """Generic training loops."""
        self.before_train()
        set_random_seed(self.cfg.SEED) #! required for reproducing
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def after_train(self):
        print("Finish training")
        # all_last_acc = self.test()
        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")
        self.close_writer()

    def forward_backward(self, batch):
        with torch.no_grad():
            image, label = self.parse_batch_train(batch)
            #* learning rate scheduling
            decay_steps = self.total_length * 0.9
            self.step_for_pdecay = min(self.step_for_pdecay, decay_steps)
            ak = (self.init_lr - self.min_lr) * (1 - self.step_for_pdecay / decay_steps) ** (0.9) + self.min_lr
            
            #* prompt parameters
            w = torch.nn.utils.parameters_to_vector(self.model.program.parameters())

            #* Randomized Gradient-Free Minimization
            m, sigma = 0, self.sigma 
            beta = torch.tensor(self.beta).cuda()
            q = torch.tensor(self.sp_avg).cuda()
            d = self.N_params

            output = self.model(image)
            loss_pivot = self.loss_fn(output, label)

            ghat = torch.zeros(d).cuda()
            for _ in range(self.sp_avg):
                # Obtain a random direction vector
                u = torch.normal(m, sigma, size=(d,)).cuda()
                u = u / torch.norm(u, p=2)

                # Forward evaluation 
                w_r = w + beta * u
                torch.nn.utils.vector_to_parameters(w_r, self.model.program.parameters())

                # Gradient estimation
                output_pt = self.model(image)
                loss_pt = self.loss_fn(output_pt, label)
                ghat = ghat + (d / q) * u * (loss_pt - loss_pivot) / beta

            #* param update
            w_new = w - ak * ghat
            torch.nn.utils.vector_to_parameters(w_new, self.model.program.parameters())
            
            loss = loss_pivot
            acc = compute_accuracy(output, label)[0].item()
        loss_summary = {"loss": loss,"acc": acc,}
        if self.cfg.use_wandb: wandb.log({'train_ep_acc':acc, 'train_ep_loss':loss.item(), 'gain_seq':ak})
        return loss_summary

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
            self.step_for_pdecay += 1
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
        # curr_result = 0.0
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