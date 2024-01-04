export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=3
export TOKENIZERS_PARALLELISM=false

#MASTER_PORT=$(shuf -n 1 -i 60000-65535)
#torchrun --nproc_per_node=2 --master_port $MASTER_PORT train.py \
#  --model_name_or_path ~/degeneration/model-zoo/hf-repo/Llama-2-7b-hf/ \
#  --learning_rate 2.0e-05 \
#  --per_device_train_batch_size 1

python train.py \
  --model_name_or_path ~/degeneration/model-zoo/hf-repo/Llama-2-7b-hf/ \
  --learning_rate 2.0e-06 \
  --per_device_train_batch_size 1 \
  --lr_scheduler_type constant
