import logging
import os
import argparse
from simpletransformers.language_modeling import LanguageModelingModel

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

def main():
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
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )

    # Other parameters
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Whether to overwrite on the existing output dir")
    parser.add_argument(
        "--output_dir",
        default='output_dir/', type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=None, type=int,
        help="Max input seq length",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=100, type=int,
        help="Number of train epochs",
    )
    parser.add_argument(
        "--train_batch_size",
        default=16, type=int,
        help="Size of each train batch",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1, type=int,
        help="gradient accumulation steps",
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

    train_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "block_size": 128,
        "max_seq_length": args.max_seq_length,
        "learning_rate": 5e-6,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "mlm": False,
        "fp16": False,
        "output_dir": args.output_dir,
        "dataset_type": "line_by_line",
    }

    model = LanguageModelingModel(model_type = "gpt2", model_name = args.model_name_or_path, args=train_args)

    model.train_model(args.data_dir+"train.txt", eval_file=args.data_dir+"test.txt")

    model.eval_model(args.data_dir+"test.txt")

if __name__ == '__main__':
    main()