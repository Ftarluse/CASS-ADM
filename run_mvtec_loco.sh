datapath=D:/Dataset/MVTEC_LOCO
augpath=D:/Dataset/dtd/images
datasets=('breakfast_box' 'juice_bottle' 'pushpins' 'screw_bag' 'splicing_connectors')
datasets=('pushpins')

dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

python main.py \
--gpu 0 \
--seed 0 \
--log_project MVTecAD_Results \
--results_path results \
--run_name mvtec_loco_00 \
--test "pixel_roc.pth" \
--evaluate_current_model 0 \
--save_segmentation_images 1 \
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
datasets \
--batch_size 8 \
--resize 288 \
--imagesize 288 "${dataset_flags[@]}" mvtec_loco $datapath $augpath
