rm -rf ./runs/demo01

python train.py \
    --init_radius 0.8 \
    --s_val_init 10.0 \
    --eikonal_weight 0.05 \
    --n_samples 128 \
    --batch_size 512 \
    --n_iters 100000 \
    --display_int 1000 \
    --exp_dir ./runs/demo01 \
    --device mps \
    --normal_mode autograd   # P800/旧卡用 finite_diff；H20/新卡用 autograd
