rm -rf ./runs/demo01

python train.py \
    --init_radius 0.8 \
    --s_val_init 50.0 \
    --eikonal_weight 0.1 \
    --n_samples 64 \
    --n_iters 20000 \
    --display_int 250 \
    --exp_dir ./runs/demo01 \
    --device cuda