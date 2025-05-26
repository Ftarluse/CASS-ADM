datapath=/root/MVTEC_AD
datasets=('wood' 'transistor' 'tile' 'screw' 'pill' 'capsule' 'carpet' 'grid' 'zipper' 'cable' 'toothbrush' 'metal_nut' 'bottle' 'hazelnut' 'leather')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

python3 main.py \
--gpu 0 \
--seed 0 \
--log_project MVTecAD_Results \
--results_path results \
--run_name 0.02 \
--test "" \
--save_segmentation_images 0 \
net \
-b wideresnet50 \
-le layer2 \
-le layer3 \
--pretrain_embed_dimension 1536 \
--target_embed_dimension 1536 \
--patchsize 3 \
--meta_epochs 200 \
--gan_epochs 1 \
--noise_std 0.02 \
--max_steps 20 \
--training_steps 5 \
--add_pos 1 \
--dsc_hidden 1024 \
--dsc_layers 2 \
--dsc_margin .5 \
--pre_proj 1 \
--gamma_clamp 0.5 \
dataset \
--batch_size 8 \
--resize 329 \
--imagesize 288 "${dataset_flags[@]}" mvtec $datapath
