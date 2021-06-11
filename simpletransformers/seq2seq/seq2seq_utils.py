import logging
import os
import pickle
from multiprocessing import Pool
from typing import Tuple

import pandas as pd
import torch
import math
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def preprocess_data(data):
    input_text, target_text, encoder_tokenizer, decoder_tokenizer, args = data

    input_text = encoder_tokenizer.encode(
        input_text, max_length=args.max_seq_length, padding='max_length', return_tensors="pt", truncation=True
    )

    target_text = decoder_tokenizer.encode(
        target_text, max_length=args.max_seq_length, padding='max_length', return_tensors="pt", truncation=True
    )
    return (torch.flatten(input_text), torch.flatten(target_text))


class Seq2SeqDataset(Dataset):
    def __init__(self, encoder_tokenizer, decoder_tokenizer, args, data, mode):
        cached_features_file = os.path.join(
            args.cache_dir, args.model_name + "_cached_" + str(args.max_seq_length) + str(len(data))
        )

        if os.path.exists(cached_features_file) and (
            (not args.reprocess_input_data and not args.no_cache)
            or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
        ):
            logger.info(" Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(" Creating features from dataset file at %s", args.cache_dir)

            data = [
                (input_text, target_text, encoder_tokenizer, decoder_tokenizer, args)
                for input_text, target_text in zip(data["input_text"], data["target_text"])
            ]

            if args.use_multiprocessing:
                with Pool(args.process_count) as p:
                    self.examples = list(
                        tqdm(
                            p.imap(preprocess_data, data, chunksize=args.multiprocessing_chunksize),
                            total=len(data),
                            disable=args.silent,
                        )
                    )
            else:
                self.examples = [preprocess_data(d) for d in tqdm(data, disable=args.silent)]

            logger.info(" Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def preprocess_data_bart(data):
    input_text, target_text, tokenizer, args = data

    input_ids = tokenizer.batch_encode_plus(
        [input_text], max_length=args.max_seq_length, padding='max_length', return_tensors="pt", truncation=True
    )

    target_ids = tokenizer.batch_encode_plus(
        [target_text], max_length=args.max_length, padding='max_length', return_tensors="pt", truncation=True
    )

    return {
        "source_ids": input_ids["input_ids"].squeeze(),
        "source_mask": input_ids["attention_mask"].squeeze(),
        "target_ids": target_ids["input_ids"].squeeze(),
    }

def preprocess_bart_data_mask_signal(data):
    input_text, target_text, tokenizer, args = data

    input_ids = tokenizer.batch_encode_plus(
        [input_text], max_length=args.max_seq_length, padding='max_length', return_tensors="pt", truncation=True
    )

    target_ids = tokenizer.batch_encode_plus(
        [target_text], max_length=args.max_length, padding='max_length', return_tensors="pt", truncation=True
    )

    source_ids = input_ids["input_ids"].squeeze()
    target_ids = target_ids["input_ids"].squeeze()

    source_tokens = [tokenizer._convert_id_to_token(id.item()) for id in source_ids]
    target_tokens = [tokenizer._convert_id_to_token(id.item()) for id in target_ids]

    source_word_mask_signal = get_word_mask_signal(source_tokens)
    target_word_mask_signal = get_word_mask_signal(target_tokens)

    return {
        "source_word_mask_signal": torch.tensor(source_word_mask_signal),
        "target_word_mask_signal": torch.tensor(target_word_mask_signal),
    }



def get_word_mask_signal(token_list):
    # is_first_real_word = True

    def convert(str_):
        if str_ == '<s>' or str_ == '<pad>' or str_ == '</s>':
            return 0
        else:
            if str_.startswith('Ġ'):
                return 1
            else:
                return 2
    word_mask_signal = [convert(str_) for str_ in token_list]
    # for str_ in token_list:
    #     if str_ == "<s>":
    #         word_mask_signal.append(0)
    #     elif str_ == "<pad>":
    #         word_mask_signal.append(0)
    #     elif str_ == "</s>":
    #         word_mask_signal.append(0)
    #     else:
    #         if str_.startswith('Ġ'):
    #             word_mask_signal.append(1)
    #         else:
    #             if is_first_real_word:
    #                 word_mask_signal.append(1)
    #                 is_first_real_word = False
    #             else:
    #                 word_mask_signal.append(2)
    return word_mask_signal

class SimpleSummarizationDataset(Dataset):
    def __init__(self, tokenizer, args, data, mode):
        self.tokenizer = tokenizer
        self.mask_ratio = args.mask_ratio
        self.replace_length = args.replace_length
        self.mask_idx = tokenizer.mask_token_id # int, the idx of <mask> token
        self.pad_idx = tokenizer.pad_token_id  # int, the idx of <pad> token
        self.stop_idx = [tokenizer.encoder[token] for token in ['.', ';', '</s>'] ] # [int], the idx of stop tokens '.' and ';' and '</s>'

        if self.replace_length not in [-1, 0, 1]:
            raise ValueError(f'invalid arg: replace_length={self.replace_length}')
        if args.mask_length not in ['subword', 'word', 'span-poisson']:
            raise ValueError(f'invalid arg: mask-length={args.mask_length}')
        if args.mask_length == 'subword' and args.replace_length not in [0, 1]:
            raise ValueError(f'if using subwords, use replace-length=1 or 0')

        self.mask_span_distribution = None
        if args.mask_length == 'span-poisson':
            _lambda = args.poisson_lambda

            lambda_to_the_k = 1
            e_to_the_minus_lambda = math.exp(-_lambda)
            k_factorial = 1
            ps = []
            for k in range(0, 128):
                ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
                lambda_to_the_k *= _lambda
                k_factorial *= (k + 1)
                if ps[-1] < 0.0000001:
                    break
            ps = torch.FloatTensor(ps)
            self.mask_span_distribution = torch.distributions.Categorical(ps)

        cached_features_file = os.path.join(
            args.cache_dir, args.model_name + "_cached_" + str(args.max_seq_length) + str(len(data))
        )

        if os.path.exists(cached_features_file) and (
            (not args.reprocess_input_data and not args.no_cache)
            or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
        ):
            logger.info(" Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(" Creating features from dataset file at %s", args.cache_dir)

            data = [
                (input_text, target_text, tokenizer, args)
                for input_text, target_text in zip(data["input_text"], data["target_text"])
            ]

            if args.use_multiprocessing:
                with Pool(args.process_count) as p:
                    self.examples = list(
                        tqdm(
                            p.imap(preprocess_data_bart, data, chunksize=args.multiprocessing_chunksize),
                            total=len(data),
                            disable=args.silent,
                        )
                    )
            else:

                self.examples = [preprocess_data_bart(d) for d in tqdm(data, disable=args.silent)]
                logger.info("Examples created")
                if self.mask_ratio > 0:
                    self.examples_types = [preprocess_bart_data_mask_signal(d) for d in tqdm(data, disable=args.silent)]  # 0 indicates start/end/pad token, 1 indicates the start of one word, 2 indicates the subsequent part of one word
                    logger.info("Examples types created")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        data = self.examples[index]
        if self.mask_ratio > 0:
            source = data["source_ids"]
            source_type = self.examples_types[index]["source_word_mask_signal"]
            data["source_ids"] = self.add_whole_word_mask(source, source_type, self.mask_ratio)
            data["source_mask"] = (data["source_ids"] != 1).long()
        return data

    def add_whole_word_mask(self, source, source_type, p=1.0):
        is_word_start = (source_type == 1).long()  # in source_type, 1 indicates the start of one word
        num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
        num_inserts = 0
        if num_to_mask == 0:
            return source

        if self.mask_span_distribution is not None:
            lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))

            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1] < num_to_mask:
                lengths = torch.cat([lengths, self.mask_span_distribution.sample(sample_shape=(num_to_mask,))], dim=0)
                cum_length = torch.cumsum(lengths, 0)

            # Trim to masking budget
            i = 0
            while cum_length[i] < num_to_mask:
                i += 1
            lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
            num_to_mask = i + 1

            # Handle 0-length mask (inserts) separately
            lengths = lengths[lengths > 0]
            num_inserts = num_to_mask - lengths.size(0)
            num_to_mask -= num_inserts
            if num_to_mask == 0:
                return self.add_insertion_noise(source, num_inserts / source.size(0))

            lengths = lengths[:num_to_mask]
            num_to_mask = lengths.size(0) # in some extreme cases, num_to_mask < lengths.size(0)

            assert (lengths > 0).all()
        else:
            lengths = torch.ones((num_to_mask,)).long()

        word_starts = is_word_start.nonzero(as_tuple=False)
        indices = word_starts[torch.randperm(word_starts.size(0))[:num_to_mask]].squeeze(1)

        end_of_sent = (source_type > 0).nonzero(as_tuple=False)[-1] + 1 # (source_type >0)[-1] is the last meaningful token
        assert end_of_sent not in indices
        assert (end_of_sent > indices).all()
        assert end_of_sent < source.size(0)
        to_keep = torch.ones(source_type.size(0), dtype=torch.bool)
        is_word_start[end_of_sent] = 255 # acts as a long length, so spans don't go over the end of doc
        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            source[indices] = self.mask_idx

        if self.mask_span_distribution is not None:
            assert len(lengths.size()) == 1
            assert lengths.size() == indices.size()
            lengths -= is_word_start[indices].long()
            uncompleted = lengths >= 0
            lengths = lengths[uncompleted]
            while indices.size(0) > 0:
                assert lengths.size() == indices.size()
                lengths -= is_word_start[indices + 1].long()
                uncompleted = lengths >= 0
                indices = indices[uncompleted] + 1
                lengths = lengths[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx

        else:
            # A bit faster when all lengths are 1
            while indices.size(0) > 0:
                uncompleted = is_word_start[indices + 1] == 0
                indices = indices[uncompleted] + 1
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx

                assert end_of_sent not in indices

        source = source[to_keep]
        if num_inserts > 0:
            source = self.add_insertion_noise(source, num_inserts / source.size(0))

        if source.size(0) < source_type.size(0):
            source = torch.cat((source, torch.tensor([self.pad_idx] * (source_type.size(0) -source.size(0)), dtype=source.dtype)), dim=0)
        else:
            source = source[:source_type.size(0)]


        return source

    def add_insertion_noise(self, tokens, p):
        if p == 0.0:
            return tokens

        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))

        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
        noise_mask[noise_indices] = 1
        result = torch.LongTensor(n + len(tokens)).fill_(-1)

        result[noise_indices] = self.mask_idx

        result[~noise_mask] = tokens

        assert (result >= 0).all()
        return result