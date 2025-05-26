
import logging
import os
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import click
import torch

sys.path.append("src")
import backbones
import CassAdm
import utils


LOGGER = logging.getLogger(__name__)

_DATASETS = {
    "mvtec": ["datasets.mvtec", "MVTecDataset"],
}


@click.group(chain=True)
@click.option("--results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_project", type=str, default="project")
@click.option("--run_name", type=str, default="test")
@click.option("--test", type=str, is_flag="")
@click.option("--save_segmentation_images",  type=int, default=0, show_default=True)
def main(**kwargs):
    pass


@main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    seed,
    log_project,
    run_name,
    test,
    save_segmentation_images
):
    methods = {key: item for (key, item) in methods}

    run_save_path = utils.create_storage_folder(
        results_path, log_project, run_name, mode="overwrite"
    )

    list_of_dataloaders = methods["get_dataloaders"](seed)
    device = utils.set_torch_device(gpu)

    result_collect = []
    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        utils.fix_seeds(seed, device)

        dataset_name = dataloaders["training"].name

        imagesize = dataloaders["training"].dataset.imagesize
        CassAdm_list = methods["get_CassAdm"](imagesize, device)
        os.makedirs(run_save_path, exist_ok=True)
       
        for i, CassAdm in enumerate(CassAdm_list):
            
            if CassAdm.extranet.backbone.seed is not None:
                utils.fix_seeds(CassAdm.backbone.seed, device)
            LOGGER.info(
                "Training models ({}/{})".format(i + 1, len(CassAdm_list))
            )
            if len(CassAdm_list) != 1:
                CassAdm.set_model_dir(os.path.join(run_save_path, f"{i}"), dataset_name)
            else:
                CassAdm.set_model_dir(run_save_path, dataset_name)
            
            if test == "":
                i_auroc, p_auroc, pro_auroc = CassAdm.train(dataloaders["training"], dataloaders["testing"])
            else:
                i_auroc, p_auroc, pro_auroc = CassAdm.test(dataloaders["training"], dataloaders["testing"], test, save_segmentation_images)

            result_collect.append({
                        "dataset_name": dataset_name,
                        "instance_auroc": i_auroc, 
                        "full_pixel_auroc": p_auroc,
                        "anomaly_pixel_auroc": pro_auroc
            })

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:3.3f}".format(key, item)) 
            LOGGER.info("\n\n-----\n") 
 
            if test == "":
                utils.create_tabel(run_save_path)
            else:
                result_metric_names = list(result_collect[-1].keys())[1:]
                result_dataset_names = [results["dataset_name"] for results in result_collect]
                result_scores = [list(results.values())[1:] for results in result_collect]
                utils.compute_and_store_final_results(
                    run_save_path,
                    result_scores,
                    column_names=result_metric_names,
                    row_names=result_dataset_names,
                )
                            
@main.command("net")
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--patchsize", type=int, default=3)
@click.option("--meta_epochs", type=int, default=1)
@click.option("--gan_epochs", type=int, default=1)
@click.option("--noise_std", type=float, default=0.05)
@click.option("--noise_strategy", type=str, default="cosine")
@click.option("--max_steps", type=int, default=20)
@click.option("--training_steps", type=int, default=5)
@click.option("--pre_proj", type=int, default=0)
@click.option("--proj_layer_type", type=int, default=0)
@click.option("--add_pos", type=int, default=1)
@click.option("--pos_trainable", type=bool, default=False)
@click.option("--dsc_layers", type=int, default=2)
@click.option("--dsc_hidden", type=int, default=None)
@click.option("--dsc_margin", type=float, default=0.8)
@click.option("--dsc_lr", type=float, default=0.0002)
@click.option("--gamma_clamp", type=float, default=0.5)

def net(
    backbone_names,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    patchsize,
    meta_epochs,
    gan_epochs, 
    noise_std,
    noise_strategy,
    max_steps,
    training_steps,
    pre_proj,
    proj_layer_type,
    add_pos,
    pos_trainable,
    dsc_layers, 
    dsc_hidden,
    dsc_margin,
    dsc_lr,
    gamma_clamp
    
):
    backbone_names = list(backbone_names)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    def get_CassAdm(input_shape, device):
        CassAdms = []
        for backbone_name, layers_to_extract_from in zip(
            backbone_names, layers_to_extract_from_coll
        ):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            backbone = backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            CassAdm_inst = CassAdm.CassAdm(device)
            CassAdm_inst.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                
                meta_epochs=meta_epochs,
                gan_epochs=gan_epochs,
                
                noise_std=noise_std,
                noise_strategy=noise_strategy,
                
                max_steps=max_steps,
                training_steps=training_steps,
                
                pre_proj=pre_proj,
                proj_layer_type=proj_layer_type,

                add_pos=add_pos,
                pos_trainable=pos_trainable,
                
                dsc_layers=dsc_layers,
                dsc_hidden=dsc_hidden,
                dsc_margin=dsc_margin,
                dsc_lr=dsc_lr,
                gamma_clamp=gamma_clamp
            )
            CassAdms.append(CassAdm_inst)
        return CassAdms

    return ("get_CassAdm", get_CassAdm)


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--train_val_split", type=float, default=1, show_default=True)
@click.option("--batch_size", default=2, type=int, show_default=True)
@click.option("--num_workers", default=2, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--rotate_degrees", default=0, type=int)
@click.option("--translate", default=0, type=float)
@click.option("--scale", default=0.0, type=float)
@click.option("--brightness", default=0.0, type=float)
@click.option("--contrast", default=0.0, type=float)
@click.option("--saturation", default=0.0, type=float)
@click.option("--gray", default=0.0, type=float)
@click.option("--hflip", default=0.0, type=float)
@click.option("--vflip", default=0.0, type=float)
@click.option("--augment", is_flag=True)
def dataset(
    name,
    data_path,
    subdatasets,
    train_val_split,
    batch_size,
    resize,
    imagesize,
    num_workers,
    rotate_degrees,
    translate,
    scale,
    brightness,
    contrast,
    saturation,
    gray,
    hflip,
    vflip,
    augment,
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed):
        dataloaders = []
        for subdataset in subdatasets:
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                train_val_split=train_val_split,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
                seed=seed,
                rotate_degrees=rotate_degrees,
                translate=translate,
                brightness_factor=brightness,
                contrast_factor=contrast,
                saturation_factor=saturation,
                gray_p=gray,
                h_flip_p=hflip,
                v_flip_p=vflip,
                scale=scale,
                augment=augment,
            )

            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )
            
            LOGGER.info(f"Dataset: train={len(train_dataset)} test={len(test_dataset)}")

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                prefetch_factor=2,
                pin_memory=True,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=2,
                pin_memory=True,
            )

            train_dataloader.name = name
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset

            if train_val_split < 1:
                val_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    classname=subdataset,
                    resize=resize,
                    train_val_split=train_val_split,
                    imagesize=imagesize,
                    split=dataset_library.DatasetSplit.VAL,
                    seed=seed,
                )

                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    prefetch_factor=4,
                    pin_memory=True,
                )
            else:
                val_dataloader = None
            dataloader_dict = {
                "training": train_dataloader,
                "validation": val_dataloader,
                "testing": test_dataloader,
            }

            dataloaders.append(dataloader_dict)
        return dataloaders

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
 
