"""Anomaly metrics."""
import cv2
import numpy as np
from sklearn import metrics
import torch
from torchmetrics.functional import auroc as auroc_compute, precision_recall_curve, average_precision
from tqdm import tqdm
import pandas as pd
from skimage import measure

class CPUEvaluator:
    def __init__(self, auroc=True, aupro=True, ap=True, best_tpr=True):

        self.auroc = auroc
        self.aupro = aupro
        self.ap = ap
        self.best_tpr = best_tpr

        self.indicators = {}
        self.image_scores = []
        self.image_labels = []
        self.pixel_scores = []
        self.pixel_masks = []

    def reset(self):

        self.indicators = {
            "image_auroc": 0,
            "image_ap": 0,
            "pixel_auroc": 0,
            "pixel_ap": 0,
            "pixel_aupro": 0,

            "best_threshold": 0,
            "best_precision": 0,
            "best_recall": 0
        }

    def append(self, image_scores=None, image_labels=None, pixel_scores=None, pixel_masks=None):
        if pixel_masks is None:
            raise ValueError("pixel_masks not found")
        pixel_scores, pixel_masks = pixel_scores.flatten(1), pixel_masks.flatten(1)

        if pixel_scores is not None:
            self.pixel_scores.append(pixel_scores.cpu().numpy())
            self.pixel_masks.append(pixel_masks.cpu().numpy())

        if image_scores is not None:
            if image_labels is None:
                image_labels = torch.max(pixel_masks, dim=1)[0]
            self.image_scores.append(image_scores.cpu().numpy())
            self.image_labels.append(image_labels.cpu().numpy())

    def compute(self):
        if len(self.image_scores) != 0:
            image_scores, image_labels = np.array(self.image_scores).ravel(), np.array(self.image_labels).ravel()
            min_scores, max_scores = np.min(image_labels), np.max(image_labels)
            image_scores = (image_scores - min_scores) / (max_scores - min_scores + 1e-10)
            self.compute_imagewise_retrieval_metrics(image_scores, image_labels)
            if self.best_tpr:
                self.compute_best_pr_re(image_scores, image_labels)

        if len(self.pixel_scores) != 0:
            pixel_scores, pixel_masks = np.array(self.pixel_scores), np.array(self.pixel_masks)
            h, w = pixel_scores.shape[-2:]
            pixel_scores, pixel_masks = pixel_scores.reshape(-1, h, w), pixel_masks.reshape(-1, h, w)

            min_scores, max_scores = np.min(pixel_scores), np.max(pixel_scores)
            pixel_scores = (pixel_scores - min_scores) / (max_scores - min_scores + 1e-10)
            self.compute_pixelwise_retrieval_metrics(pixel_scores, pixel_masks)
            if self.aupro:
                self.compute_pro(pixel_masks, pixel_scores)

        self.image_scores.clear()
        self.image_labels.clear()
        self.pixel_scores.clear()
        self.pixel_masks.clear()

    def compute_best_pr_re(self,  anomaly_prediction_weights, anomaly_ground_truth_labels):
        """
        Computes the best precision, recall and threshold for a given set of
        anomaly ground truth labels and anomaly prediction weights.
        """
        precision, recall, thresholds = metrics.precision_recall_curve(anomaly_ground_truth_labels,
                                                                       anomaly_prediction_weights)
        f1_scores = 2 * (precision * recall) / (precision + recall)

        best_threshold = thresholds[np.argmax(f1_scores)]
        best_precision = precision[np.argmax(f1_scores)]
        best_recall = recall[np.argmax(f1_scores)]
        self.indicators["best_threshold"] = best_threshold
        self.indicators["best_precision"] = best_precision
        self.indicators["best_recall"] = best_recall

    def compute_imagewise_retrieval_metrics(self, anomaly_prediction_weights, anomaly_ground_truth_labels):

        if self.auroc:
            self.indicators["image_auroc"] = metrics.roc_auc_score(anomaly_ground_truth_labels,
                                                                   anomaly_prediction_weights)
        if self.ap:
            self.indicators["image_ap"] = metrics.average_precision_score(anomaly_ground_truth_labels,
                                                                          anomaly_prediction_weights)

    def compute_pixelwise_retrieval_metrics(self, anomaly_segmentations, ground_truth_masks):
        """
        Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
        and ground truth segmentation masks.
        """

        flat_anomaly_segmentations = anomaly_segmentations.ravel()
        flat_ground_truth_masks = ground_truth_masks.ravel()
        if self.auroc:
            self.indicators["pixel_auroc"] = metrics.roc_auc_score(flat_ground_truth_masks.astype(int),
                                                                   flat_anomaly_segmentations)
        if self.ap:
            self.indicators["pixel_ap"] = metrics.average_precision_score(flat_ground_truth_masks.astype(int),
                                                                          flat_anomaly_segmentations)


class GPUEvaluator:

    def __init__(self, cal_num=-1):

        self.indicators = {}

        self.image_scores = []
        self.image_labels = []
        self.pixel_scores = []
        self.pixel_masks = []

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.cal_num = cal_num
        self.current = 0
        self.count = 0
        self.reset()

    def append(self, image_scores=None, image_labels=None, pixel_scores=None, pixel_masks=None):
        if self.count == 0:
            self.reset()
        self.current += 1
        b = pixel_scores.shape[0]

        if pixel_masks is None:
            raise ValueError("pixel_masks not found")

        if pixel_scores is not None:
            self.pixel_scores += np.split(pixel_scores.float().cpu().numpy(), b, axis=0)
            self.pixel_masks += np.split(pixel_masks.int().cpu().numpy(), b, axis=0)

        if image_scores is not None:
            if image_labels is None:
                image_labels = torch.max(pixel_masks.flatten(1), dim=1)[0]
            self.image_scores += np.split(image_scores.float().cpu().numpy(), b, axis=0)
            self.image_labels += np.split(image_labels.int().cpu().numpy(), b, axis=0)

        if (self.current >= self.cal_num) and self.cal_num > 0:
            self.current = 0
            self.batch_compute()

    def add(self, dict_):
        for key, value in dict_.items():
            self.indicators[key] = self.indicators.get(key,  0) + value

    def reset(self):

        self.indicators = {
            "image_auroc": 0,
            "image_ap": 0,

            "pixel_auroc": 0,
            "pixel_ap": 0,
            "aupro": 0,

            "best_threshold": 0,
            "best_precision": 0,
            "best_recall": 0
        }

    def batch_compute(self):
        # patch core auroc compute
        self.count += 1
        indicators = {}
        if len(self.image_scores) != 0:
            image_scores, image_labels = np.array(self.image_scores), np.array(self.image_labels)

            min_scores, max_scores = np.min(image_labels), np.max(image_labels)
            image_scores = (image_scores - min_scores) / (max_scores - min_scores + 1e-10)

            image_scores, image_labels = (torch.tensor(image_scores).flatten().to(self.device),
                                          torch.tensor(image_labels).flatten().to(self.device))

            image_auroc = auroc_compute(image_scores, image_labels, task="binary")
            image_ap = average_precision(image_scores, image_labels, task="binary")

            indicators["image_auroc"] = image_auroc.item()
            indicators["image_ap"] = image_ap.item()

        if len(self.pixel_scores) != 0:
            pixel_scores, pixel_masks = np.array(self.pixel_scores), np.array(self.pixel_masks)

            min_scores, max_scores = np.min(pixel_scores),  np.max(pixel_scores)
            pixel_scores = (pixel_scores - min_scores) / (max_scores - min_scores + 1e-10)

            pixel_scores, pixel_masks = (torch.tensor(pixel_scores).flatten().to(self.device),
                                         torch.tensor(pixel_masks).flatten().to(self.device))

            pixel_auroc = auroc_compute(pixel_scores, pixel_masks, task="binary")
            pixel_ap = average_precision(pixel_scores, pixel_masks, task="binary")

            precision, recall, thresholds = precision_recall_curve(pixel_scores, pixel_masks, task="binary")
            recall, precision, thresholds = recall.cpu().numpy(), precision.cpu().numpy(), thresholds.cpu().numpy()

            f1_scores = np.divide(
                2 * precision * recall,
                precision + recall,
                out=np.zeros_like(precision),
                where=(precision + recall) != 0,
            )

            f1_argmax = np.argmax(f1_scores)
            best_threshold = thresholds[f1_argmax]
            best_precision = precision[f1_argmax]
            best_recall = recall[f1_argmax]

            indicators["best_threshold"] = best_threshold.item()
            indicators["best_precision"] = best_precision.item()
            indicators["best_recall"] = best_recall.item()

            indicators["pixel_auroc"] = pixel_auroc.item()
            indicators["pixel_ap"] = pixel_ap.item()

        self.add(indicators)
        self.image_scores.clear()
        self.image_labels.clear()
        self.pixel_scores.clear()
        self.pixel_masks.clear()

    def compute(self):
        if self.current != 0:
            self.batch_compute()
            self.current = 0
        self.indicators = {key: round(value / self.count, 4) for key, value in self.indicators.items()}
        self.count = 0
        return self.indicators


    
def compute_pro(masks, amaps, num_th=200):

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    with tqdm(total=num_th) as tq:
        tq.set_description("  >> Compute AUPRO <<  ")
        for th in np.arange(min_th, max_th, delta):
            tq.update(1)
            binary_amaps[amaps <= th] = 0
            binary_amaps[amaps > th] = 1
    
            pros = []
            for binary_amap, mask in zip(binary_amaps, masks):
                binary_amap = cv2.dilate(binary_amap.astype(np.uint8), k)
                for region in measure.regionprops(measure.label(mask)):
                    axes0_ids = region.coords[:, 0]
                    axes1_ids = region.coords[:, 1]
                    tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                    pros.append(tp_pixels / region.area)
    
            inverse_masks = 1 - masks
            fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
            fpr = fp_pixels / inverse_masks.sum()
            
            df.loc[len(df)] = {"pro": np.mean(pros), "fpr": fpr,"threshold": th}

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = metrics.auc(df["fpr"], df["pro"])
    return pro_auc