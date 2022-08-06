eval "$(conda shell.bash hook)"
conda activate LAMA

# export model_path=/home/wph/model-hub/LostGAN/netGv1_coco128.pth
# export CUDA_VISIBLE_DEVICES='0'
# export dir_name=lostganv1_coco_128

# export model_path=/home/wph/model-hub/layout2img/G_app_coco.pth
# export CUDA_VISIBLE_DEVICES='4'
# export dir_name=cal2im_coco_128

export model_path=/home/wph/model-hub/SCL-GAN/G_coco128_tem_6_50.pth
export CUDA_VISIBLE_DEVICES='6'
export dir_name=scl_coco_128_2

# export model_path=/home/wph/model-hub/LostGAN/netGv2_coco128.pth
# export CUDA_VISIBLE_DEVICES='2'
# export dir_name=lostganv2_coco_128

cd /home/wph/LAMA &&
python test.py --dataset coco --model_path $model_path --img_size 128 -N --cropped_size 32 --sample_path samples/$dir_name/ --gpu $CUDA_VISIBLE_DEVICES &&

cd /home/wph/pytorch_image_classification &&
mkdir $dir_name &&
cd $dir_name 
ln -s /home/wph/datasets/coco/val_128_cropped_32/ val &&
ln -s /home/wph/LAMA/samples/$dir_name/coco128_repeat5_thres2.0_cropped_32/ train
cd .. &&


sed -i '/macs/d' train.py 
sed -i '/n_params/d' train.py &&
python train.py --config configs/cifar/resnet.yaml dataset.name ImageNet dataset.dataset_dir $dir_name train.output_dir experiments/$dir_name/ dataset.n_classes 184
