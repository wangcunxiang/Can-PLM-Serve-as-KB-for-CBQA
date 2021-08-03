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
### Frequently Asked Questions and Answers

It will continue to be updated on this page.  So please feel free to ask us questions through any channels.

Q0: Can you describe this work briefly?
A0: Some works have proved PLMs can contain knowledge in their parameters. We want to know whether PLMs can learn knowledge through pretraining and whether PLMs can use their internal knowledge to solve problems. So we first continue pretrain (LM-finetune) the PLM with some passages and ask the PLM to recite the passages. Then, we QA-finetune the model and ask it to answer questions which relates the passages. The results show that the PLMs cannot memorize much knowledge through pretraining and it is weak of them to use internal knowledge to answer questions after finetuning.

Q1: What is your conclusion?
A1: The results show that the PLMs cannot memorize much knowledge through pretraining and it is weak of them to use internal knowledge to answer questions after finetuning. So, it is difficult to use PLMs as KBs in current Pre-training -$>$ Fine-tuning paradigm.

Q2: Why you randomly mask tokens during LM-finetuning while mask specific tokens during reciting?
A2: For LM-finetuning, we want to explore whether PLMs can learn knowledge from pretraining. So, we set LM-finetuning the same with original pretraining process, which randomly mask tokens. For reciting, we want to link this process to the subsequent QA process, therefore, we can naturally compare the reciting accuracy and QA accuracy.

Q3: Why don't you split train/dev/test set during knowledge memorization experiments?
A3: We don't think it makes sense to ask the model to learn some passages and then ask it to recite others.

Q4: If you do XXX things (e.g. Using more refined mask policy) in knowledge memorization/question answering, it will improve the accuracy in reciting/QA, why do not you do these?
A4: Our approach is based on the most classic Pre-training -$>$ Fine-tuning paradigm. We suppose it is more valuable to use the most popular and standard paradigm when researching on this question. In addition, some methods may improve accuracy on both tasks, but we suppose it will affect the conclusion too much, after all, current results are far away from indicating strong abilities of knowledge memorizing and question answering.

Q5: Does this paper concludes that PLMs cannot serve as knowledge bases?
A5: Yes and no. If you simply use the Pre-training -$>$ Fine-tuning paradigm, we suppose it will not work. However, if you optimize the paradigm, we suppose it is still promising to research on this topic because PLMs can indeed store and utilize knowledge. 

Q6: Can the conclusion of this paper also apply to other downstream tasks? Why you choose Closed-book QA as the representative task?
A6: We suppose the conclusion is very likely the same with other downstream tasks. We choose Closed-book QA is because it is most direct and suitable task for explore why how much knowledge can models have, we have also considered other tasks, but none of them is as suitable as Closed-book QA.


### Citation
If you find this paper useful, you can cite

```
@inproceedings{wang-etal-2021-generative,
    title = "Can Generative Pre-trained Language Models Serve As Knowledge Bases for Closed-book {QA}?",
    author = "Wang, Cunxiang  and
      Liu, Pai  and
      Zhang, Yue",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.251",
    doi = "10.18653/v1/2021.acl-long.251",
    pages = "3241--3251",
    abstract = "Recent work has investigated the interesting question using pre-trained language models (PLMs) as knowledge bases for answering open questions. However, existing work is limited in using small benchmarks with high test-train overlaps. We construct a new dataset of closed-book QA using SQuAD, and investigate the performance of BART. Experiments show that it is challenging for BART to remember training facts in high precision, and also challenging to answer closed-book questions even if relevant knowledge is retained. Some promising directions are found, including decoupling the knowledge memorizing process and the QA finetune process, forcing the model to recall relevant knowledge when question answering.",
}

```
