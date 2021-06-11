from sacrebleu import corpus_bleu
from transformers import BartTokenizer
from tqdm import tqdm
import numpy as np
import json

def tokenize(input, Tokenizer):
    # input_ids = Tokenizer.batch_encode_plus(input)['input_ids'][0]
    # input_strs = [Tokenizer.decoder[id] for id in input_ids]
    # return ' '.join(input_strs)
    return input
def load_triplets(fin):
    '''
    input:
    (Repeated 3 lines)
    question
        output
        target
    '''
    with open(fin,encoding='utf8') as f:
        # to remove the head lines of file
        f.readline()
        f.readline()
        f.readline()
        contents = f.readlines()
        triplets = []
        for i, sent in enumerate(contents):
            if '\t' not in sent:
                if '\t' in contents[i + 1] and '\t' in contents[i + 2]:
                    # question:'', output:'', target:['']
                    triplets.append((sent.strip(), contents[i + 1].strip(), contents[i + 2].strip().split('\t')))
    return triplets

def load_s2t_t2s(f0, f1):
    '''Output: s2t: {[]}; train_t2s: {[]}'''
    with open(f0, "r",encoding='utf8') as f0, open(f1, "r",encoding='utf8') as f1:
        s2t = {}
        t2s = {}
        source = []
        target = []
        for s, ts in zip(f0, f1):
            s = s.strip()
            source.append(s)
            ts = ts.strip().split('\t')
            target += [t for t in ts]
            s2t[s.strip()] = ts
            for t in ts:
                t = t.strip()
                if t2s.__contains__(t):
                    t2s[t].append(s)
                else:
                    t2s[t] = [s,]
    return source, list(set(target)), s2t, t2s

def main():
    Tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    test_triplets = load_triplets('../results/facebook_bart-large_WebQuestion_splitted_48_24_128/predictions_35000.txt')
    train_source, train_target, train_s2t, train_t2s = load_s2t_t2s('../dataset/WebQuestion/original/train.source', '../dataset/WebQuestion/original/train.target')

    train_ques_set_ = train_t2s.values()
    train_ques_set = [tokenize(ques, Tokenizer) for ques in train_ques_set_]
    test_triplets_iters = tqdm(test_triplets)
    # test_ques2train_ques_bleu = {ques_ : {tr_s: corpus_bleu(sys_stream = tokenize(ques_, Tokenizer), ref_streams=tokenize(tr_s, Tokenizer)).score \
    #                                       for tr_s in train_source}  for ques_, op_, tg_ in test_triplets_iters}

    with open('test_ques2train_ques_bleu.json', 'r', encoding='utf8') as fin:
        test_ques2train_ques_bleu = json.load(fin)
    test_ques2train_ques_sort = {ques_:[(k, v) for k,v in sorted(test_ques2train_ques_bleu[ques_].items(), key=lambda d: d[1], reverse=True)] \
                                 for ques_, op_, tg_ in test_triplets_iters}
    tmp = ([(key, test_ques2train_ques_sort[key][0], test_ques2train_ques_sort[key][1], test_ques2train_ques_sort[key][2]) for key in test_ques2train_ques_sort.keys()])
    for item in tmp:
        print(item)
    # test_tg2train_tg_bleu = {tg_ : {tr_t: corpus_bleu(sys_stream = tokenize(tg_, Tokenizer), ref_streams=tokenize(tr_t, Tokenizer)).score \
    #                                       for tr_t in train_target}  for ques_, op_, tgs_ in test_triplets_iters for tg_ in tgs_}

    with open('test_tg2train_tg_bleu.json', 'r', encoding='utf8') as fin:
        test_tg2train_tg_bleu = json.load(fin)
    test_tg2train_tg_sort = {tg_:{v:k for k,v in sorted(test_tg2train_tg_bleu[tg_].items(), key=lambda d: d[1], reverse=True)} \
                                 for ques_, op_, tgs_ in test_triplets_iters for tg_ in tgs_}

    for ques_, op_, tg_ in test_triplets_iters:
        if train_t2s.__contains__(op_):
            cor_train_ques = train_t2s[op_]
            print([test_ques2train_ques_sort[ques] for ques in cor_train_ques])
        else:
            print([test_tg2train_tg_sort[tg_][i] for i in range(3)])


if __name__ == '__main__':
    main()