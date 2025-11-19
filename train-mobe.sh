# MODEL_DIR=/home/user1/workspace/bobchenyx/Qwen
MODEL_DIR=/shared/user67/workspace/bobchenyx/Qwen
MODEL_NAME=Qwen3-30B-A3B-Instruct-2507

# python -u train.py --index_path $MODEL_DIR/$MODEL_NAME/model.safetensors.index.json \
#       --base_dir $MODEL_DIR/$MODEL_NAME \
#       --save_path $MODEL_DIR/MoBE/$MODEL_NAME \
#       --num_hidden_layers 48 \
#       --num_matrices 128 \
#       --rows_per_matrix 768 \
#       --cols 2048 \
#       --num_epochs 10000 \
#       --batch_size 32 \
#       --num_batches 4 \
#       --learning_rate 0.07 \
#       --num_B 64 \
#       --truncation 768 \
#       --start_layer 0 \
#       --end_layer 48 \
#       --matrix_type "gate_proj" \
#       --activation 'silu' 2>&1 | tee logs/train_mobe_$MODEL_NAME.log

python -u train.py --index_path $MODEL_DIR/$MODEL_NAME/model.safetensors.index.json \
      --base_dir $MODEL_DIR/$MODEL_NAME \
      --save_path $MODEL_DIR/MoBE/$MODEL_NAME \
      --num_hidden_layers 48 \
      --num_matrices 128 \
      --rows_per_matrix 768 \
      --cols 2048 \
      --num_epochs 10000 \
      --batch_size 32 \
      --num_batches 4 \
      --learning_rate 0.07 \
      --num_B 64 \
      --truncation 768 \
      --start_layer 0 \
      --end_layer 48 \
      --matrix_type "up_proj" \
      --activation 'silu' 2>&1 | tee logs/train_mobe_$MODEL_NAME-up_proj.log


# python train.py --index_path /root/DeepSeek-V3-0324/model.safetensors.index.json \
#       --base_dir /root/DeepSeek-V3-0324 \
#       --save_path /root/MoBE/DeepSeek-V3-0324 \
#       --num_hidden_layers 61 \
#       --num_matrices 256 \
#       --rows_per_matrix 2048 \
#       --cols 7168 \
#       --num_epochs 10000 \
#       --batch_size 32 \
#       --num_batches 8 \
#       --learning_rate 0.07 \
#       --num_B 64 \
#       --truncation 2048 \
#       --start_layer 3 \
#       --end_layer 61 \
#       --matrix_type "gate_proj" \
#       --activation 'tanh'