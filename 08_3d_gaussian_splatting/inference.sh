DATA_PATH=../data/360_extra_scenes/bonsai
EXP_DIR=./runs/h20_v5/

python inference.py \
  --data_path $DATA_PATH \
  --model_path $EXP_DIR/gs_final.pth \
  --output_gif $EXP_DIR/walkthrough.gif \
  --factor 2 \
  --n_frames 120 \
  --fps 20 \
  --tile_size 32 \
  --device cuda
