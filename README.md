# Can-PLM-Serve-as-KB-for-CBQA

This folder contains the original code and data for ACL2021 paper *<Can Generative Pre-trained Language Models Serve asKnowledge Bases for Closed-book QA?>*

Paper link:https://arxiv.org/abs/2106.01561

## Setup
```bash
git clone https://github.com/wangcunxiang/Can-PLM-Serve-as-KB-for-CBQA.git && cd simpletransformers
pip install -e ./
pip install -r requirements-dev.txt
pip install transformers==3.5.0
```

## dataset
Download dataset from https://drive.google.com/file/d/1K2uw9WXct6kA8i6_taJWeETGj2OdqgC1/view?usp=sharing

Our used data is from open source datasets, including [NaturalQuestions](https://ai.google.com/research/NaturalQuestions/), [TriviaQA](http://nlp.cs.washington.edu/triviaqa/), [WebQuestion](https://worksheets.codalab.org/worksheets/0xba659fe363cb46e7a505c5b6a774dc8a) and [SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer/). We split the development set of the
NaturalQuestions, TriviaQA and SQuAD2.0 and the test set of WebQuestion into two subsets to serve as a new development set and a new test set(Dec 2020). And
we extract several subsets from SQuAD2.0 to serve as our new datasets,Among which articles are used as pre-training data, and questions and answers are used as QA dataset(Dec 2020). There is no additional data collection process.

You can read Dataset.md for details of datasets.

## command



Followings are simple example commands, you can read the description of each command argument for more details.
### LM-tuning
Example training command for LM-tuning

```bash
MODEL_PATH=facebook/bart-large
DATA_PATH=/* YOUR_DATA_PATH */
OUTPUT_DIR=/* YOUR_OUTPUT_DIRECTORY */

CUDA_VISIBLE_DEVICES=0 python train_generate_qa.py \
--data_dir $DATA_PATH \
--model_type seq2seq \
--model_name_or_path $MODEL_PATH \
--output_dir $OUTPUT_DIR \
--do_train \
--max_seq_length 512 \
--max_length 512 \
--train_batch_size 1 \
--gradient_accumulation_steps 4 \
--mask_ratio 0.3 \
--save_step 500 \
--overwrite_output_dir \
--num_train_epochs 200 \
--predict_on_valid \
--evaluation_metric passage
```
Example testing command for LM-tuning
```bash
MODEL_PATH=/* YOUR_LM-TUNED_MODEL_PATH */
DATA_PATH=/* YOUR_DATA_PATH */
PREDICTION_DIR=/* YOUR_PREDICTION_DIRECTORY */

CUDA_VISIBLE_DEVICES=0 python train_generate_qa.py \
--data_dir $DATA_PATH \
--model_type seq2seq \
--model_name_or_path $MODEL_PATH \
--prediction_dir $PREDICTION_DIR \
--do_predict \
--max_seq_length 512 \
--max_length 512 \
--train_batch_size 1 \
--gradient_accumulation_steps 8 \
--save_step 500 \
--overwrite_output_dir \
--evaluation_metric passage
```

### QA-tuning
Example training command for QA-tuning
```bash
MODEL_PATH=/* YOUR_LM-TUNED_MODEL_PATH */
DATA_PATH=/* YOUR_DATA_PATH */
OUTPUT_DIR=/* YOUR_OUTPUT_DIRECTORY */
MAX_SEQ_LENGTH=64
MAX_LENGTH=64
TRAIN_BATCH_SIZE=8

CUDA_VISIBLE_DEVICES=1 python train_generate_qa.py \
--model_type seq2seq \
--data_dir ${DATASET} \
--model_name_or_path $MODEL_PATH \
--output_dir $OUTPUT_DIR \
--do_train \
--max_seq_length $MAX_SEQ_LENGTH \
--max_length $MAX_LENGTH \
--train_batch_size $TRAIN_BATCH_SIZE \
--save_step 500 \
--num_train_epochs 30 \
--dataloader_num_workers 0 \
--overwrite_output_dir \
--predict_on_valid \
--gradient_accumulation_steps 4 \
--evaluation_metric qa
```

Example testing command for QA-tuning
```bash
MODEL_PATH=/* YOUR_LM-TUNED_MODEL_PATH */
DATA_PATH=/* YOUR_DATA_PATH */
OUTPUT_DIR=/* YOUR_OUTPUT_DIRECTORY */
PREDICTION_DIR=/* YOUR_PREDICTION_DIRECTORY */
MAX_SEQ_LENGTH=64
MAX_LENGTH=64
TRAIN_BATCH_SIZE=8

CUDA_VISIBLE_DEVICES=1 python train_generate_qa.py \
--model_type seq2seq \
--data_dir ${DATASET} \
--model_name_or_path $MODEL_PATH \
--prediction_dir $PREDICTION_DIR \
--max_seq_length $MAX_SEQ_LENGTH \
--max_length $MAX_LENGTH \
--train_batch_size $TRAIN_BATCH_SIZE \
--save_step 500 \
--num_train_epochs 30 \
--dataloader_num_workers 0 \
--overwrite_output_dir \
--do_predict \
--gradient_accumulation_steps 4 \
--evaluation_metric qa
```

