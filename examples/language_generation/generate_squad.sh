MODEL_PATH=outputs/saved_model/saved_model/

CUDA_VISIBLE_DEVICES=1 python generate.py \
--data_dir data/GPT-2_topic_train_16_test_4_original/question_answer/ \
--model_name_or_path $MODEL_PATH \
--output_dir $MODEL_PATH/predictions
