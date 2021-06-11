MODEL_PATH=gpt2
DATA_PATH=data/passage/

CUDA_VISIBLE_DEVICES=0 python fine_tune.py \
--data_dir $DATA_PATH \
--model_name_or_path ${MODEL_PATH} \
--output_dir outputs/ \
--do_train \
--max_seq_length 384 \
--train_batch_size 16
--gradient_accumulation_steps 4 \
--overwrite_output_dir \

