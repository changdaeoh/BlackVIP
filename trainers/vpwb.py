import os.path as osp
import time
import datetime
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from my_dassl.engine import TRAINER_REGISTRY, TrainerX
from my_dassl.metrics import compute_accuracy
from my_dassl.utils import load_pretrained_weights, load_checkpoint, AverageMeter, MetricMeter
from my_dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from trainers.utils import load_clip_to_cpu
from trainers import visual_prompters
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
        
        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features.to(device)
        self.prompter = visual_prompters.__dict__[cfg.TRAINER.VPWB.METHOD](cfg)

    def forward(self, image):
        prompted_images  = self.prompter(image.type(self.dtype))
        image_features = self.image_encoder(prompted_images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits


@TRAINER_REGISTRY.register()
class VPWB(TrainerX):
    """Visual Prompting White-Box sertting
    """
    def check_cfg(self, cfg):
        assert cfg.TRAINER.VPWB.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.VPWB.PREC == "fp32" or cfg.TRAINER.VPWB.PREC == "amp":
            clip_model = clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, self.device)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompter" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompter, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model.prompter, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        
        self.register_model("prompter", self.model.prompter, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.VPWB.PREC == "amp" else None

        self.N_params = len(torch.nn.utils.parameters_to_vector(self.model.prompter.parameters()))
        self.step = 0 
        
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.VPWB.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        acc = compute_accuracy(output, label)[0].item()
        loss_summary = {
            "loss": loss.item(),
            "acc": acc,
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        if self.cfg.use_wandb: wandb.log({'train_ep_acc':acc, 'train_ep_loss':loss.item()})
        return loss_summary


    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

        
    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        #!
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

    def after_train(self):
        print("Finish training")
        # all_last_acc = self.test()
        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")
        self.close_writer()
        
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
        
        #wandb.log({'val_ep_acc':curr_result})