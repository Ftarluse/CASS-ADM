import logging
import os
import pickle
from collections import OrderedDict
import csv
import math
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter

import common
import metrics
from utils import plot_segmentation_images
from scipy.optimize import curve_fit

class CassAdm(torch.nn.Module):
    """Controlled Anomaly Synthesis Strategy with Asymptotic Diffusion Modulation"""
    def __init__(self, device):

        super(CassAdm, self).__init__()
        self.device = device

    def load(
            self,
            backbone,
            layers_to_extract_from,
            device,
            input_shape,
            pretrain_embed_dimension,  # 1536
            target_embed_dimension,  # 1536
            patchsize=3,  # 3
            patchstride=1,

            meta_epochs=1,  # 40
            gan_epochs=1,  # 4
        
            noise_std=0.05, # 0.02
            noise_strategy = "cosine",  # cosine
            max_steps = 20, # 20
            training_steps = 5,
            
            pre_proj=0,  # 1
            proj_layer_type=0,

            add_pos=1, # 1
            pos_trainable = False,
        
            dsc_layers=2,  # 2
            dsc_hidden=None,  # 1024
            dsc_margin=.8,  # .5
            dsc_lr=0.0002,

            gamma_clamp = 0.5
    ):

        self.extranet = common.Extractor(
            backbone=backbone,
            patch_size=patchsize,
            stride=patchstride,
            mid_dim=pretrain_embed_dimension,
            target_dim=target_embed_dimension,
            in_shape=input_shape,
            layers_to_extract_from=layers_to_extract_from,
        )
        
        self.anomaly_segmentor = common.RescaleSegmentor(target_size=input_shape[-2:])
        self.device = device

        if self.device == "cpu":
            self.evaluator = metrics.CPUEvaluator()
        else:
            self.evaluator = metrics.GPUEvaluator()
        
        h, w = self.extranet.detail["Split_shape"]
        
        self.meta_epochs = meta_epochs
        self.gan_epochs = gan_epochs

        self.noise_std = noise_std
        self.noise_strategy = noise_strategy
        self.max_steps = max_steps
        self.training_steps = training_steps
        
        self.dsc_lr = dsc_lr
        self.pre_proj = pre_proj
        
        self.dsc_margin = dsc_margin
        self.add_pos = add_pos

        if self.pre_proj > 0:
            self.pre_projection = common.Projection(target_embed_dimension, target_embed_dimension, pre_proj, proj_layer_type)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.AdamW(self.pre_projection.parameters(), self.dsc_lr * .5)

        if self.add_pos > 0:
            self.add_position = common.PositionEncoder(target_embed_dimension, size=(h, w), learnable=False)
            self.add_position.to(self.device)
            if self.add_position.learnable:
                self.pos_opt = torch.optim.AdamW(self.add_position.parameters(), self.dsc_lr)

        self.gamma = torch.nn.Parameter(torch.full(size=(self.max_steps - 1,), fill_value=0.0, device=self.device))
        self.gamma_opt = torch.optim.Adam([self.gamma], lr=1e-3)
        self.gamma_clamp = gamma_clamp
        self.discriminator = common.Discriminator(target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden, hw = h*w)

        self.discriminator.to(self.device)
        self.dsc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.dsc_lr, weight_decay=1e-5)

        self.model_dir = ""
        self.fin_ep = 0

    def set_model_dir(self, model_dir, dataset_name):

        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.csv_dir = os.path.join(self.ckpt_dir, "result.csv")
        self.weights_dir = os.path.join(self.ckpt_dir, "weights")
        os.makedirs(self.weights_dir, exist_ok=True)

        self.last_pt = os.path.join(self.weights_dir, "last.pth")
        self.image_roc_pt = os.path.join(self.weights_dir, "image_roc.pth")
        self.pixel_roc_pt = os.path.join(self.weights_dir, "pixel_roc.pth")

    def test(self, training_data, test_data, test, save_segmentation_images, evaluate_current_model):

        try:
            csd = torch.load(os.path.join(self.weights_dir, test), weights_only=True)
            self.load_state_dicts(csd)
            i_auroc, p_auroc, aupro = 0.0, 0.0, 0.0
            
            with tqdm.tqdm(total=len(test_data)) as tq:
                if evaluate_current_model:
                    tq.set_description("  >> Compute AUROC <<  ")
                else:
                    tq.set_description("  >> Compute SCORE <<  ")
                    
                aggregator = {"scores": [], "segmentations": [], "features": [], "mask": []}
                for data in test_data:   
                    scores, segmentations, features = self._predict(data["image"])
                    aggregator["scores"].append(scores.cpu().numpy())
                    aggregator["segmentations"].append(segmentations.cpu().numpy())
                    aggregator["features"].append(features.cpu().numpy())
                    aggregator["mask"].append(data["mask"])
                    
                    if evaluate_current_model:
                        self.evaluator.append(
                        image_scores=scores,
                        image_labels=data["is_anomaly"],
                        pixel_scores=segmentations,
                        pixel_masks=data["mask"])
                        tq.update(1)

            segmentations = np.concatenate(aggregator["segmentations"], axis=0)
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0)

            if evaluate_current_model:
                # # aupro
                norm_segmentations = np.zeros_like(segmentations)
                for min_score, max_score in zip(min_scores, max_scores):
                    norm_segmentations += (segmentations - min_score) / max(max_score - min_score, 1e-2)
                norm_segmentations = norm_segmentations / len(scores)
                
                mask = np.concatenate(aggregator["mask"], axis=0).squeeze()
                aupro = round(metrics.compute_pro(mask, norm_segmentations, num_th=200), 4)
    
                test_indicators = self.evaluator.compute()
                i_auroc, p_auroc = test_indicators["image_auroc"], test_indicators["pixel_auroc"]
      
            if save_segmentation_images:
                
                scores = np.concatenate(aggregator["scores"], axis=0)
                min_scores = scores.min(axis=-1).reshape(-1, 1)
                max_scores = scores.max(axis=-1).reshape(-1, 1)
                scores = (scores - min_scores) / (max_scores - min_scores)
                scores = np.mean(scores, axis=0)

                self.save_segmentation_images(test_data, segmentations, scores)
            return i_auroc, p_auroc, aupro
            
        except FileNotFoundError:
            print("Warning: Model file not found. If you want to train the model, please set the test parameter in run.sh to an empty string ('').")
            print(f"Root: {os.path.join(self.weights_dir, test)}")
            return None, None, None
            
    def train(self, training_data, test_data):
        
        if os.path.exists(self.last_pt):
            csd = torch.load(self.last_pt, weights_only=True)
            self.load_state_dicts(csd)
        else:
            self.best_record = (None, None, None)

        self.extranet.eval()
        for i_mepoch in range(self.fin_ep, self.meta_epochs):
            self.fin_ep += 1
            self.head = f"Epoch {i_mepoch}/{self.meta_epochs}  "

            train_indicators = self._train_discriminator(training_data)        
            test_indicators = self._predict_dataloader(test_data)

            indicators = {"epoch": i_mepoch, **train_indicators, **test_indicators}

            if os.path.exists(self.csv_dir):
                with open(self.csv_dir, "a", encoding="utf-8", newline="") as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(indicators.values())
            else:
                with open(self.csv_dir, "w", encoding="utf-8", newline="") as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(indicators.keys())
                    csv_writer.writerow(indicators.values())
        return self.best_record

    
    def _train_discriminator(self, input_data):
        
        _ = self.extranet.eval()

        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()

        i_iter = 0
        train_indicators = {}
        with tqdm.tqdm(total=self.gan_epochs * len(input_data)) as tq:
            tq.set_description(f"{self.head}>> Train  <<  GPU:{torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB")
            for i_epoch in range(self.gan_epochs):
                all_loss = []
                all_p_true = []
                all_p_fake = []
        
                all_gamma = []
                all_noise_std = []
                 
                embeddings_list = []
                for data_item in input_data:
                    self.dsc_opt.zero_grad()
                    if self.pre_proj > 0:
                        self.proj_opt.zero_grad()

                    if self.add_pos > 0:
                        if self.add_position.learnable:
                            self.pos_opt.zero_grad()

                    i_iter += 1
                    img = data_item["image"]
                  
                    img = img.to(torch.float).to(self.device)
                   
                    if self.pre_proj > 0:
                        true_feats = self.pre_projection(self.extranet(img))
                    else:
                        true_feats = self.extranet(img)

                    if self.add_pos > 0:
                        true_feats = self.add_position(true_feats)

                    (b, c), (h, w)= true_feats.size(), self.extranet.patch_maker.ref_num_patches
                    b = b // (h*w)
    
                    
                    th = self.dsc_margin 
                    t = torch.linspace(0,  torch.pi,  self.max_steps).to(self.device)
                    
                    if self.noise_strategy == "cosine":
                        weights = 0.2 + 0.8 / 2 * (1 + torch.cos(t))
                        
                    elif self.noise_strategy == "linear":
                        weights = torch.ones_like(t)
                        
                    beta = self.noise_std**2 * weights / weights.sum()
                    noise_step_std = torch.sqrt(beta)
                    
                    acc_step_std = torch.stack([(noise_step_std[0: step + 1]**2).sum().sqrt() for step in range(self.max_steps)]).to(self.device)
                    fake_feats = true_feats + noise_step_std[0] * torch.randn_like(true_feats) 
                   
                    ano_scores = []
                        
                    for step in range(self.max_steps):
            
                        fin_st = (step + 1) == self.max_steps
                        
                        all_scores = self.discriminator(torch.cat([true_feats, fake_feats]))

                        true_scores, fake_scores = torch.chunk(all_scores, dim=0, chunks=2)
                        ano_scores.append(fake_scores.detach().mean())

                        if (step + 1) % self.training_steps == 0 or fin_st:   
                            true_loss = torch.clip(-true_scores + th, min=0)
                            fake_loss = torch.clip(fake_scores + th, min=0)
                            
                            loss = true_loss.mean() + fake_loss.mean()
                            
                            if not fin_st:
                                loss.backward(retain_graph=True)
                            else:
                                loss.backward()
                                break
                            
                        loss = torch.clip(true_scores - fake_scores, min=0)
                        grad = torch.autograd.grad(loss.mean(), [fake_feats])[0]
                        grad = (grad - grad.mean()) / (grad.std() + 1e-8)
                        grad = - grad.detach()

                        with torch.no_grad():
            
                            sigma = noise_step_std[step+1] 
                            gamma = self.gamma[step] 
                   
                            add_noise = gamma * grad + (1 - gamma**2)**0.5 * torch.randn_like(true_feats)
                            fake_feats.add_(sigma * add_noise) 

                    
                    self.gamma_opt.zero_grad()
                    ano_scores = torch.stack(ano_scores)  

                    inc_scores = ano_scores.median() - ano_scores[1: ]  
                    
                    gamma_grad = - inc_scores
                    self.gamma.grad = gamma_grad / (gamma_grad.norm() + 1e-8)
                       
                    self.gamma_opt.step()
                    
                    with torch.no_grad():
                        self.gamma.clamp_(-self.gamma_clamp, self.gamma_clamp)
                        
                    p_true = (true_scores.detach() >= th).sum() / len(true_scores)
                    p_fake = (fake_scores.detach() < -th).sum() / len(fake_scores)
                
                    noise_std = (fake_feats.detach() - true_feats.detach()).std(dim=1).mean()
                    
                    if self.pre_proj > 0:
                        self.proj_opt.step()

                    if self.add_pos > 0:
                        if self.add_position.learnable:
                            self.pos_opt.step()

                    self.dsc_opt.step()

                    loss = loss.detach().cpu()
                    
                    all_loss.append(loss.item())
                    all_noise_std.append(noise_std.item())
    
                    all_gamma.append(self.gamma.mean().item())
                    all_p_true.append(p_true.cpu().item())
                    all_p_fake.append(p_fake.cpu().item())
              
                    tq.update(1)

                if len(embeddings_list) > 0:
                    self.auto_noise[1] = torch.cat(embeddings_list).std(0).mean(-1)

                all_loss = round(sum(all_loss) / len(input_data), 5)
                all_noise_std = round(sum(all_noise_std) / len(input_data), 3)
       
                all_gamma = round(sum(all_gamma) / len(input_data), 3)
                all_p_true = round(sum(all_p_true) / len(input_data), 3)
                all_p_fake = round(sum(all_p_fake) / len(input_data), 3)
                
                cur_lr = self.dsc_opt.state_dict()['param_groups'][0]['lr']

                tq.set_postfix(
                    loss=all_loss,
                    lr=round(cur_lr, 6),
                    noise_std=all_noise_std,
               
                    gamma= all_gamma,
                    p_true=all_p_true,
                    p_fake=all_p_fake,
                )

                train_indicators["loss"] = train_indicators.get("loss", 0) + all_loss
                train_indicators["noise_std"] = train_indicators.get("noise_std", 0) + all_noise_std
         
                train_indicators["gamma"] = train_indicators.get("gamma", 0) + all_gamma
                train_indicators["p_true"] = train_indicators.get("p_true", 0) + all_p_true
                train_indicators["p_fake"] = train_indicators.get("p_fake", 0) + all_p_fake
            
            for key, value in train_indicators.items():
                if key == "loss":
                    train_indicators[key] = round(value / self.gan_epochs, 5)
                else:
                    train_indicators[key] = round(value / self.gan_epochs, 3)

        return train_indicators

    def update_state_dicts(self):
        
        self.checkpoint = {
            "epoch": self.fin_ep,
            "best_record": self.best_record,
            "gamma": self.gamma.data.detach().cpu() 
        }

        self.checkpoint["discriminator"] =  OrderedDict({k:v.detach().cpu() for k, v in self.discriminator.state_dict().items()})
        if self.pre_proj > 0:
            self.checkpoint["pre_projection"] = OrderedDict({k:v.detach().cpu() for k, v in self.pre_projection.state_dict().items()})
        if self.add_pos > 0:
            self.checkpoint["position_encoding"] = OrderedDict({k:v.detach().cpu() for k, v in self.add_position.state_dict().items()})
            
    def load_state_dicts(self, checkpoint):
        
        self.fin_ep = checkpoint["epoch"]
        self.best_record = checkpoint["best_record"]
       
        if 'discriminator' in checkpoint:
            self.discriminator.load_state_dict(checkpoint['discriminator'])
    
            self.gamma = torch.nn.Parameter(checkpoint['gamma'].to(self.device))
            if "pre_projection" in checkpoint:
                self.pre_projection.load_state_dict(checkpoint["pre_projection"])
            if "position_encoding" in checkpoint:
                self.add_position.load_state_dict(checkpoint["position_encoding"])
        else:
            self.load_state_dict(checkpoint["state_dict"], strict=False)
                
    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    
    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
    
        head, color = " " * len(self.head), "red"
        with tqdm.tqdm(total=len(dataloader)) as tq:
            tq.set_description(f"{head}>> Test   <<  GPU:{torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB")
            for data in dataloader:     

                _scores, _masks, _feats = self._predict(data["image"])

                self.evaluator.append(
                    image_scores=_scores,
                    image_labels=data["is_anomaly"],
                    pixel_scores=_masks,
                    pixel_masks=data["mask"]
                )

                tq.update(1)
            
            test_indicators = self.evaluator.compute()
            i_auroc, p_auroc = test_indicators["image_auroc"], test_indicators["pixel_auroc"]
            aupro = test_indicators["aupro"]
                
            self.update_state_dicts()
            
            if self.best_record == (None, None, None):
                self.best_record = [i_auroc, p_auroc, aupro]
                self.checkpoint["best_record"] = self.best_record
                
            else:
                if i_auroc > self.best_record[0]:
                    self.best_record[0] = i_auroc
                    self.checkpoint["best_record"] = self.best_record
                    
                    torch.save(self.checkpoint, self.image_roc_pt)
                    head, color = "IMAGE AUROC" + " " * (len(head) - 11), "yellow"

                if p_auroc > self.best_record[1]:
                    self.best_record[1] = p_auroc
                    self.best_record[2] = aupro
                    self.checkpoint["best_record"] = self.best_record
                    
                    torch.save(self.checkpoint, self.pixel_roc_pt)
                    head, color = "PIXEL AUROC" + " " * (len(head) - 11), "magenta"

                if i_auroc >= self.best_record[0] and p_auroc >= self.best_record[1]:
                    head, color = "Best Model " + " " * (len(head) - 11), "cyan"

            torch.save(self.checkpoint, self.last_pt)
            
            tq.set_description(colored_text(
                f"{head}>> Test   <<  GPU:{torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB", f"{color}"))
            tq.set_postfix(
                I_AUROC=f"{i_auroc}(MAX:{self.best_record[0]})",
                P_AUROC=f"{p_auroc}(MAX:{self.best_record[1]})",
                AUPRO=f"{aupro}(MAX:{self.best_record[2]})"
            )

        return test_indicators

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.extranet.eval()

        batchsize = images.shape[0]
        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()
        with torch.no_grad():
            features = self.extranet(images)
            h, w = self.extranet.patch_maker.ref_num_patches

            if self.pre_proj > 0:
                features = self.pre_projection(features)
            if self.add_pos > 0:
                features = self.add_position(features)

            patch_scores = image_scores = - self.discriminator(features).reshape(-1, h, w)

            image_scores = torch.max(image_scores.flatten(1), dim=1)[0]
            patch_scores = self.anomaly_segmentor(patch_scores)

            features = features.reshape(batchsize, h, w, -1)

        return image_scores, patch_scores, features

    def save_segmentation_images(self, data, segmentations, scores):
        image_paths = [
            x[2] for x in data.dataset.data_to_iterate
        ]
        mask_paths = [
            x[3] for x in data.dataset.data_to_iterate
        ]

        def image_transform(image):
            in_std = np.array(
                data.dataset.transform_std
            ).reshape(-1, 1, 1)
            in_mean = np.array(
                data.dataset.transform_mean
            ).reshape(-1, 1, 1)
            image = data.dataset.transform_img(image)
            return np.clip(
                (image.numpy() * in_std + in_mean) * 255, 0, 255
            ).astype(np.uint8)

        def mask_transform(mask):
            return data.dataset.transform_mask(mask).numpy()

        plot_segmentation_images(
            f'./output/{data.dataset.classnames_to_use[0]}',
            image_paths,
            segmentations,
            scores,
            mask_paths,
            image_transform=image_transform,
            mask_transform=mask_transform,
        )


from colorama import Fore, Style


def colored_text(text, color="red"):
    colors = {
        'red': Fore.RED,
        'green': Fore.GREEN,
        'yellow': Fore.YELLOW,
        'blue': Fore.BLUE,
        'magenta': Fore.MAGENTA,
        'cyan': Fore.CYAN,
        'white': Fore.WHITE
    }
    return f"{colors.get(color, Fore.WHITE)}{text}{Style.RESET_ALL}"

