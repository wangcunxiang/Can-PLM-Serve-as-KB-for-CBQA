MODEL_PATH=outputs/saved_model/
DATA_PATH=data/GPT-2_topic_train_16_test_4_original/question_answer/

CUDA_VISIBLE_DEVICES=1 python fine_tune.py \
--data_dir $DATA_PATH \
--model_name_or_path ${MODEL_PATH} \
--output_dir $MODEL_PATH \
--do_train \
--max_seq_length 32 \
--train_batch_size 64 \
--gradient_accumulation_steps 4 \
--overwrite_output_dir \

