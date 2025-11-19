# MODEL_DIR=/home/user1/workspace/bobchenyx/Qwen
MODEL_DIR=/shared/user67/workspace/bobchenyx/Qwen
MODEL_NAME=Qwen3-30B-A3B-Instruct-2507

python -u get_mobe.py --base_model $MODEL_DIR/$MODEL_NAME \
      --mobe_dir $MODEL_DIR/MoBE/$MODEL_NAME \
      --save_dir $MODEL_DIR/MoBE/$MODEL_NAME-MoBE \
      --num_B 64 \
      --num_experts 128 \
      --start_layer 0 \
      --end_layer 48 \
      --dtype bfloat16 \
      --activation 'silu' 2>&1 | tee logs/get_mobe_$MODEL_NAME.log


# python get_mobe.py --base_model /root/DeepSeek-V3-0324 \
#       --mobe_dir /root/MoBE/DeepSeek-V3-0324 \
#       --save_dir /root/DeepSeek-V3-0324-MoBE \
#       --num_B 64 \
#       --num_experts 256 \
#       --start_layer 3 \
#       --end_layer 61 \
#       --dtype bfloat16 \
#       --activation 'tanh' 