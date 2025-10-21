import csv
import logging
import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import tqdm
import pandas as pd
LOGGER = logging.getLogger(__name__)


def plot_segmentation_images(
    savefolder,
    image_paths,
    segmentations,
    anomaly_scores=None,
    mask_paths=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
    save_depth=4,
    save_type=1
):
    """Generate anomaly segmentation images.

    Args:
        image_paths: List[str] List of paths to images.
        segmentations: [List[np.ndarray]] Generated anomaly segmentations.
        anomaly_scores: [List[float]] Anomaly scores for each image.
        mask_paths: [List[str]] List of paths to ground truth masks.
        image_transform: [function or lambda] Optional transformation of images.
        mask_transform: [function or lambda] Optional transformation of masks.
        save_depth: [int] Number of path-strings to use for image savenames.
    """
    if mask_paths is None:
        mask_paths = ["-1" for _ in range(len(image_paths))]
    masks_provided = mask_paths[0] != "-1"
    if anomaly_scores is None:
        anomaly_scores = ["-1" for _ in range(len(image_paths))]

    os.makedirs(savefolder, exist_ok=True)

    for image_path, mask_path, anomaly_score, segmentation in tqdm.tqdm(
        zip(image_paths, mask_paths, anomaly_scores, segmentations),
        total=len(image_paths),
        desc="Generating Segmentation Images...",
        leave=False,
    ):
        image = PIL.Image.open(image_path).convert("RGB")
        image = image_transform(image)
        if not isinstance(image, np.ndarray):
            image = image.numpy()

        if masks_provided:
            if mask_path is not None:
                try:
                    mask = PIL.Image.open(mask_path).convert('F')
                except PermissionError:
                    "mvtec_loco"
                    mask = PIL.Image.open(os.path.join(mask_path, "000.png")).convert('L')

                mask = mask_transform(mask)
                mask = np.where(mask > 0, 1, 0).astype(np.float32)
                if not isinstance(mask, np.ndarray):
                    mask = mask.numpy()
            else:
                mask = np.zeros_like(image)

        savename = image_path.replace('/', '\\').split('\\')
        savename = "_".join(savename[-save_depth:])
        savename = os.path.join(savefolder, savename)

        if save_type > 0:

            f, axes = plt.subplots(2, 2)
            axes = axes.flatten()
            for ax in axes:
                ax.axis('off')
            axes[0].imshow(image.transpose(1, 2, 0))
            axes[1].imshow(mask.transpose(1, 2, 0), cmap='gray', interpolation='lanczos')
            axes[2].imshow(segmentation, cmap='magma', interpolation='lanczos')
            axes[3].imshow(segmentation, cmap='jet', interpolation='lanczos')
            f.set_size_inches(6, 6)
            f.tight_layout()

            f.savefig(savename)
            plt.close()
        else:
            os.makedirs(savename, exist_ok=True)
            save_clean_image(image.transpose(1, 2, 0), os.path.join(savename, 'image.png'))
            save_clean_image(mask.transpose(1, 2, 0)[:, :, 0], os.path.join(savename, 'mask.png'))
            save_clean_image(segmentation, os.path.join(savename, 'segmentation.png'), cmap='magma')
 

def save_clean_image(img, save_path, cmap=None):
    if cmap:
        img_norm = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        cmap_func = plt.get_cmap(cmap)
        img_color = (cmap_func(img_norm)[..., :3] * 255).astype(np.uint8)  # RGBA -> RGB
        img_bgr = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)
    else:
        if img.ndim == 3 and img.shape[2] == 3:
            img_bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        else:
            img_bgr = (img * 255).astype(np.uint8) 

    cv2.imwrite(save_path, img_bgr)
    
def create_storage_folder(
    main_folder_path,  group_folder, run_name, mode="iterate"
):
    os.makedirs(main_folder_path, exist_ok=True)
    project_path = os.path.join(main_folder_path)
    os.makedirs(project_path, exist_ok=True)
    save_path = os.path.join(project_path, group_folder, run_name)
    if mode == "iterate":
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(project_path, group_folder + "_" + str(counter))
            counter += 1
        os.makedirs(save_path)
    elif mode == "overwrite":
        os.makedirs(save_path, exist_ok=True)

    return save_path


def set_torch_device(gpu_ids):
    """Returns correct torch.device.

    Args:
        gpu_ids: [list] list of gpu ids. If empty, cpu is used.
    """
    if len(gpu_ids):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def create_tabel(results_path):
    
    class_list = [item for item in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, item))]
  
    data = {"class": [], "train_epoch": [], "best_epoch": [], "image_auroc": [], "pixel_auroc": []}
    for c in class_list:
        try:
            r = pd.read_csv(f"{results_path}/{c}/result.csv")
        except FileNotFoundError:
            continue

        try:
            data["train_epoch"].append(int(r["epoch"].iloc[-1]))
        except IndexError:
            continue

        max_sum_row = r.loc[(r["image_auroc"] + r["pixel_auroc"]).idxmax()]

        iroc = round(max_sum_row["image_auroc"], 4).item()
        proc = round(max_sum_row["pixel_auroc"], 4).item()
        best_epoch = int(max_sum_row["epoch"])

        data["image_auroc"].append(iroc)
        data["pixel_auroc"].append(proc)
        data["best_epoch"].append(best_epoch)
        data["class"].append(c)
  
    l = len(data["class"])
    data["class"].append("mean")
    data["train_epoch"].append(int(sum(data["train_epoch"]) / l))
    data["best_epoch"].append(int(sum(data["best_epoch"]) / l))
    data["image_auroc"].append(round(sum(data["image_auroc"]) / l, 4))
    data["pixel_auroc"].append(round(sum(data["pixel_auroc"]) / l, 4))

    df = pd.DataFrame(data)
    df.to_csv(f"{results_path}/all_results.csv", index=False)
    

def compute_and_store_final_results(
    results_path,
    results,
    row_names=None,
    column_names=[
        "Instance AUROC",
        "Full Pixel AUROC",
        "Full PRO",
        "Anomaly Pixel AUROC",
        "Anomaly PRO",
    ],
):
    """Store computed results as CSV file.

    Args:
        results_path: [str] Where to store result csv.
        results: [List[List]] List of lists containing results per datasets,
                 with results[i][0] == 'dataset_name' and results[i][1:6] =
                 [instance_auroc, full_pixelwisew_auroc, full_pro,
                 anomaly-only_pw_auroc, anomaly-only_pro]
    """
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."

    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        mean_metrics[result_key] = np.mean([x[i] for x in results])
            
    savename = os.path.join(results_path, "test_results.csv")
    with open(savename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ["Mean"] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {"mean_{0}".format(key): item for key, item in mean_metrics.items()}
    return mean_metrics
