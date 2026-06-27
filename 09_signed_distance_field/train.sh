rm -rf ./runs/demo01

# NeuS风格，CDF式，表面更锐利
#   --alpha_mode neus --s_val_init 5.0

# volsdf风格，训练更稳定
#   --alpha_mode volsdf --s_val_init 10.0

python train.py \
    --init_radius 0.8 \
    --alpha_mode volsdf \
    --s_val_init 10.0 \
    --eikonal_weight 0.1 \
    --n_samples 128 \
    --batch_size 512 \
    --n_iters 100000 \
    --warm_up_end 5000 \
    --lr_alpha 0.05 \
    --display_int 1000 \
    --exp_dir ./runs/demo01 \
    --device mps