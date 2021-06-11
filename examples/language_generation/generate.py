import logging
import os
import argparse
from simpletransformers.language_generation import LanguageGenerationModel

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
    parser.add_argument(
        "--output_dir",
        default='output_dir/', type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    args = parser.parse_args()

    model = LanguageGenerationModel(model_type = "gpt2", model_name = args.model_name_or_path, args={"max_length": 64})

    fsource = open(args.data_dir+'test.source', 'r', encoding='utf8')
    prompts = [sent.strip()+'\t' for sent in fsource.readlines()]

    ftarget = open(args.data_dir+'test.target', 'r', encoding='utf8')
    targets = [sent.strip()+'\t' for sent in ftarget.readlines()]

    foutput = open(args.output_dir+'test.hypo', 'w', encoding='utf8', newline='\n')

    assert len(prompts) == len(targets)
    case_number = len(prompts)
    correct_number = 0
    for i, prompt in enumerate(prompts):
        # Generate text using the model. Verbose set to False to prevent logging generated sequences.
        generated = model.generate(prompt, verbose=False)

        print("=============================================================================")
        print(generated[0])
        generated[0] = generated[0].split('\t')[1].strip('!')
        targets[i] = targets[i].strip()
        print(targets[i])
        print(generated[0])
        print("=============================================================================")
        foutput.write(generated[0])
        if generated[0] == targets[i]:
            correct_number += 1
    print('correct number = {}, case number = {}'.format(correct_number, case_number))

if __name__ == '__main__':
    main()