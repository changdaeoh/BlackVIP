import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F

from my_dassl.engine import TRAINER_REGISTRY, TrainerX
from my_dassl.optim import build_optimizer, build_lr_scheduler
from my_dassl.utils import load_checkpoint
from my_dassl.metrics import compute_accuracy

from clip import clip

from trainers.utils import load_clip_to_cpu
import wandb
from trainers.zsclip import CUSTOM_TEMPLATES


class ClassificationHead(torch.nn.Linear):
    def __init__(self, weights, biases=None):
        output_size, input_size = weights.shape     #! (C, D)
        super().__init__(input_size, output_size)   #! (D, C)
        
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone(), requires_grad=True)
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone(), requires_grad=True)
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias), requires_grad=True)


class ImageClassifierCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.dtype = clip_model.dtype
        self.logit_scale = clip_model.logit_scale.exp()

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Text Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        
        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            text_features *= self.logit_scale

        self.image_encoder = clip_model.visual 
        self.classification_head = ClassificationHead(weights=text_features)

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return self.classification_head(image_features)


@TRAINER_REGISTRY.register()
class FTCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model = clip_model.float()
        
        self.model = ImageClassifierCLIP(cfg, classnames, clip_model).to(self.device)

        if cfg.TRAINER.FTCLIP.METHOD == 'lp':
            print("Do Linear Probe")
            self.optim = build_optimizer(self.model.classification_head, cfg.OPTIM)
            self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
            self.register_model("linear_probe", self.model.classification_head, self.optim, self.sched)
        else:
            print("Do Full Fine-Tune")
            self.optim = build_optimizer(self.model, cfg.OPTIM)    
            self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
            self.register_model("full_ft", self.model, self.optim, self.sched)
        
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)
    
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
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
        
        wandb.log({'train_ep_acc':acc, 'train_ep_loss':loss.item()})

        return loss_summary

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