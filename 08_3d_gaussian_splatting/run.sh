rm -rf ./runs/h20_flowers_sh3

python train.py \
    --data_path ../data/360_extra_scenes/flowers \
    --exp_dir ./runs/h20_flowers_sh3 \
    --factor 4 \
    --num_points 50000 \
    --n_iters 30000 \
    --lr 1e-2 \
    --sh_degree 3 \
    --grad_threshold 0.0002 \
    --display_int 500 \
    --device cuda