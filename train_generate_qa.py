import logging

import argparse
import os

import sklearn

from simpletransformers.seq2seq import Seq2SeqModel
from simpletransformers.t5 import T5Model

from data_reader.data_reader import read_data_source_target

def main():

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the source and target files for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type, choose from [seq2seq, T5]",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )


    # Other parameters
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the valid set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run prediction on the test set.")
    parser.add_argument("--predict_on_valid", action="store_true", help="Whether to run prediction on the valid set. If yes, it will only run predication on valid set.")
    parser.add_argument("--init_model_weights", action="store_true", help="Whether to initialize the model weights")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Whether to overwrite on the existing output dir")
    parser.add_argument("--use_multiprocessed_decoding", action="store_true",
                        help="Whether to use multiprocess when decoding")
    parser.add_argument("--save_model_every_epoch", action="store_true",
                        help="Whether to save model every epoch during training")
    parser.add_argument("--predict_during_training", action="store_true",
                        help="Whether to predict after each checkpoint-saving during training")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to evaluate after each checkpoint-saving during training")
    parser.add_argument(
        "--output_dir",
        default='output_dir/', type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--save_step",
        default=0, type=int,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=16, type=int,
        help="Size of each train batch",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=16, type=int,
        help="Size of each eval/predict batch",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1, type=int,
        help="gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        default=4e-5, type=float,
        help="learning rate",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=100, type=int,
        help="Number of train epochs",
    )
    parser.add_argument(
        "--max_seq_length",
        default=None, type=int,
        help="Max input seq length",
    )
    parser.add_argument(
        "--max_length",
        default=None, type=int,
        help="Max output seq length",
    )
    parser.add_argument(
        "--prediction_dir",
        default=None, type=str,
        help="The output directory where the predictions results will be written.",
    )
    parser.add_argument(
        "--prediction_suffix",
        default=None, type=str,
        help=" The supplementary suffix of prediction results name.",
    )
    parser.add_argument(
        "--mask_ratio",
        default=0.0, type=float,
        help="the proportion of masked words in the source",
    )
    parser.add_argument(
        "--mask_length",
        default="span-poisson", type=str,
        choices=['subword', 'word', 'span-poisson'],
        help="when masking words, the length of mask segments",
    )
    parser.add_argument(
        '--replace_length', default=-1, type=int,
        help='when masking N tokens, replace with 0, 1, or N tokens (use -1 for N)'
    )
    parser.add_argument(
        '--poisson_lambda',
        default=3.0, type=float,
        help='randomly shuffle sentences for this proportion of inputs'
    )
    parser.add_argument(
        '--dataloader_num_workers', default=0, type=int,
        help='the number of cpus used in collecting data in dataloader, '
             'note that if it is large than cpu number, the program may be stuck'
    )
    parser.add_argument(
        '--evaluation_metric', default='qa', type=str,
        help='if pretrain passages, use \'passage\', else use \'qa\''
    )


    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    if args.do_train == True:
        train_df = read_data_source_target(args.data_dir + "train.source", args.data_dir + "train.target")
    else:
        train_df = None

    if args.do_eval == True or args.evaluate_during_training == True:
        eval_df = read_data_source_target(args.data_dir + "valid.source", args.data_dir + "valid.target")
    else:
        eval_df = None

    if args.do_predict == True or args.predict_during_training == True:
        if args.predict_on_valid == True:
            test_df = read_data_source_target(args.data_dir + "valid.source", args.data_dir + "valid.target")
        else:
            test_df = read_data_source_target(args.data_dir + "test.source", args.data_dir + "test.target")
    else:
        test_df = None

    model_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": args.overwrite_output_dir,
        "init_model_weights": args.init_model_weights,
        "max_seq_length": args.max_seq_length,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "save_eval_checkpoints": False,
        "save_model_every_epoch": args.save_model_every_epoch,
        "save_steps": args.save_step,
        "evaluate_during_training": args.evaluate_during_training,
        "evaluate_generated_text": True,
        "evaluate_during_training_verbose": True,
        "predict_during_training": args.predict_during_training,
        "use_multiprocessing": False,
        "output_dir": args.output_dir,
        "max_length": args.max_length,
        "manual_seed": 4,
        "mask_ratio": args.mask_ratio,
        "mask_length": args.mask_length,
        "replace_length": args.replace_length,
        "poisson_lambda": args.poisson_lambda,
        "fp16":False,
        "truncation":True,
        "dataloader_num_workers":args.dataloader_num_workers,
        "use_multiprocessed_decoding":args.use_multiprocessed_decoding,
        "evaluation_metric": args.evaluation_metric,
        "predict_on_valid": args.predict_on_valid
    }

    # Initialize model
    if args.model_type == 'seq2seq':
        model = Seq2SeqModel(
            encoder_decoder_type="bart",
            encoder_decoder_name=args.model_name_or_path,
            args=model_args,
        )
    elif args.model_type == 't5':
        model = T5Model(
            model_name=args.model_name_or_path,
            args=model_args,
        )
    else:
        raise ValueError(
            "The {} model is not supported now".format(args.model_type)
        )

    # Train the model
    if args.do_train == True:
        model.train_model(train_data=train_df, eval_data=eval_df, test_data=test_df, output_dir=args.output_dir)

    # Evaluate the model
    if args.do_eval == True:
        results = model.eval_model(eval_data=eval_df)
        print(results)

    # Use the model for prediction
    if args.do_predict == True:
        print(model.predict(pred_data=test_df, output_dir=args.prediction_dir, suffix=args.prediction_suffix))

if __name__ == '__main__':
    main()