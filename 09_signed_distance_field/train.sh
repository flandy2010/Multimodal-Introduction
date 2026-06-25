rm -rf ./runs/demo01

python train.py \
    --init_radius 0.8 \
    --s_val_init 10.0 \
    --eikonal_weight 0.05 \
    --n_samples 128 \
    --n_iters 20000 \
    --display_int 250 \
    --exp_dir ./runs/demo01 \
    --device mps