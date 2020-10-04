# -*- coding: utf-8 -*-

'''
   Read data from JSON files,
   in the meantime, we do preprocess like capitalize the first character of a sentence or normalize digits
'''
import os

import json
from collections import Counter

import numpy as np
import argparse

from io_utils import read_yaml, read_lines, read_json_lines, load_embedding_dict, save_pickle
from str_utils import capitalize_first_char, normalize_tok, normalize_sent, collapse_role_type
from vocab import Vocab

from actions import Actions

joint_config = read_yaml('joint_config.yaml')

parser = argparse.ArgumentParser(description = 'this is a description')
parser.add_argument('--seed', '-s', required = False, type = int, default=joint_config['random_seed'])
args = parser.parse_args()
joint_config['random_seed'] = args.seed
print('seed:',joint_config['random_seed'])

np.random.seed(joint_config['random_seed'])

data_config = read_yaml('data_config.yaml')

data_dir = data_config['data_dir']
ace05_event_dir = data_config['ace05_event_dir']
embedding_dir = data_config['embedding_dir']
embedding_file = data_config['embedding_file']
embedding_type = data_config['embedding_type']

normalize_digits = data_config['normalize_digits']
lower_case = data_config['lower_case']

vocab_dir = data_config['vocab_dir']
token_vocab_file= os.path.join(vocab_dir, data_config['token_vocab_file'])
char_vocab_file= os.path.join(vocab_dir, data_config['char_vocab_file'])
ent_type_vocab_file= os.path.join(vocab_dir, data_config['ent_type_vocab_file'])
ent_ref_vocab_file= os.path.join(vocab_dir, data_config['ent_ref_vocab_file']) # co-reference
tri_type_vocab_file= os.path.join(vocab_dir, data_config['tri_type_vocab_file'])
arg_type_vocab_file= os.path.join(vocab_dir, data_config['arg_type_vocab_file'])
action_vocab_file = os.path.join(vocab_dir, data_config['action_vocab_file'])
pos_vocab_file = os.path.join(vocab_dir, data_config['pos_vocab_file'])

pickle_dir = data_config['pickle_dir']
vec_npy_file = data_config['vec_npy']
inst_pl_file = data_config['inst_pl_file']

train_list = read_json_lines(os.path.join(ace05_event_dir, 'train_nlp_ner.json'))
dev_list = read_json_lines(os.path.join(ace05_event_dir, 'dev_nlp_ner.json'))
test_list = read_json_lines(os.path.join(ace05_event_dir, 'test_nlp_ner.json'))

print('Sentence size Train: %d, Dev: %d, Test: %d'%(len(train_list), len(dev_list), len(test_list)))

embedd_dict, embedd_dim = None, None

def read_embedding():
    global  embedd_dict, embedd_dim
    embedd_dict, embedd_dim = load_embedding_dict(embedding_type,
                                                  os.path.join(embedding_dir, embedding_file),
                                                  normalize_digits=normalize_digits)
    print('Embedding type %s, file %s'%(embedding_type, embedding_file))
    #print('Embedding vocab size: %d, dim: %d'%(len(embedd_dict), embedd_dim))

read_embedding()


def build_vocab():
    token_list = []
    char_list = []
    tri_type_list = []
    ent_type_list = []
    ent_ref_list = []
    arg_type_list = []
    actions_list = []
    pos_list = []

    tri_word_set = []
    for inst in train_list:
        words = inst['nlp_words']
        tris = inst['Triggers'] # (idx, event_type)
        ents = inst['Entities'] # (start, end, coarse_type, ref_type)
        args = inst['Arguments'] # (ent_start, ent_end, trigger_idx, argument_type)
        pos_list.extend(inst['nlp_pos'])

        for word in words:
            word = normalize_tok(word, lower_case, normalize_digits)
            token_list.append(word)
            char_list.extend(list(word))

        for tri in tris:
            tri_type_list.append(tri[1].lower())
            tri_word_set.append((words[tri[0]]))

        for ent in ents:
            ent_type_list.append(ent[2])
            ent_ref_list.append(ent[3])

        collapsed_args = []
        for arg in args:
            collapsed_type = collapse_role_type(arg[3]).lower()
            arg_type_list.append(collapsed_type)
            collapsed_args.append([arg[0], arg[1], arg[2], collapsed_type])

        actions = Actions.make_oracle(words,tris,ents,collapsed_args)
        actions_list.extend(actions)

    train_token_set = set(token_list)

    dev_oo_train_but_in_glove = 0
    for inst in dev_list:
        words = inst['nlp_words']
        tris = inst['Triggers']  # (idx, event_type)
        ents = inst['Entities']  # (start, end, coarse_type, ref_type)
        args = inst['Arguments']  # (ent_start, ent_end, trigger_idx, argument_type)
        pos_list.extend(inst['nlp_pos'])

        for word in words:
            word = normalize_tok(word, lower_case, normalize_digits)
            if embedd_dict is not None and (word in embedd_dict or word.lower() in embedd_dict):
                token_list.append(word)
                char_list.extend(list(word))
                if word not in train_token_set:
                    dev_oo_train_but_in_glove += 1

        for tri in tris:
            tri_type_list.append(tri[1].lower())
            tri_word_set.append((words[tri[0]]))

        for ent in ents:
            ent_type_list.append(ent[2])
            ent_ref_list.append(ent[3])

        collapsed_args = []
        for arg in args:
            collapsed_type = collapse_role_type(arg[3]).lower()
            arg_type_list.append(collapsed_type)
            collapsed_args.append([arg[0], arg[1], arg[2], collapsed_type])

        actions = Actions.make_oracle(words, tris, ents, collapsed_args)
        actions_list.extend(actions)

    test_oo_train_but_in_glove = 0
    for inst in test_list:
        words = inst['nlp_words']
        tris = inst['Triggers']  # (idx, event_type)
        ents = inst['Entities']  # (start, end, coarse_type, ref_type)
        args = inst['Arguments']  # (ent_start, ent_end, trigger_idx, argument_type)
        pos_list.extend(inst['nlp_pos'])

        for word in words:
            word = normalize_tok(word, lower_case, normalize_digits)
            if embedd_dict is not None and (word in embedd_dict or word.lower() in embedd_dict):
                token_list.append(word)
                char_list.extend(list(word))
                if word not in train_token_set:
                    test_oo_train_but_in_glove += 1

        for tri in tris:
            tri_type_list.append(tri[1].lower())
            #tri_word_set.append((words[tri[0]]))

        for ent in ents:
            ent_type_list.append(ent[2])
            ent_ref_list.append(ent[3])

        collapsed_args = []
        for arg in args:
            collapsed_type = collapse_role_type(arg[3]).lower()
            arg_type_list.append(collapsed_type)
            collapsed_args.append([arg[0], arg[1], arg[2], collapsed_type])

        actions = Actions.make_oracle(words, tris, ents, collapsed_args)
        actions_list.extend(actions)

    print('dev_oo_train_but_in_glove : ', dev_oo_train_but_in_glove)
    print('test_oo_train_but_in_glove : ', test_oo_train_but_in_glove)

    print('--------token_vocab---------------')
    token_vocab = Vocab()
    token_vocab.add_spec_toks(unk_tok=True, pad_tok=False)
    token_vocab.add_counter(Counter(token_list))
    token_vocab.save(token_vocab_file)
    print(token_vocab)

    print('--------char_vocab---------------')
    char_vocab = Vocab()
    char_vocab.add_spec_toks(unk_tok=True, pad_tok=False)
    char_vocab.add_counter(Counter(char_list))
    char_vocab.save(char_vocab_file)
    print(char_vocab)

    print('--------ent_type_vocab---------------')
    ent_type_vocab = Vocab()
    ent_type_vocab.add_spec_toks(pad_tok=False, unk_tok=False)
    ent_type_vocab.add_counter(Counter(ent_type_list))
    ent_type_vocab.save(ent_type_vocab_file)
    print(ent_type_vocab)

    print('--------ent_ref_vocab---------------')
    ent_ref_vocab = Vocab()
    ent_ref_vocab.add_spec_toks(pad_tok=False, unk_tok=False)
    ent_ref_vocab.add_counter(Counter(ent_ref_list))
    ent_ref_vocab.save(ent_ref_vocab_file)
    print(ent_ref_vocab)

    print('--------tri_type_vocab---------------')
    tri_type_vocab = Vocab()
    tri_type_vocab.add_spec_toks(pad_tok=False, unk_tok=False, null_tok=True)
    tri_type_vocab.add_counter(Counter(tri_type_list))
    tri_type_vocab.save(tri_type_vocab_file)
    print(tri_type_vocab)

    print('--------arg_type_vocab---------------')
    arg_type_vocab = Vocab()
    arg_type_vocab.add_spec_toks(pad_tok=False, unk_tok=False, null_tok=True)
    arg_type_vocab.add_counter(Counter(arg_type_list))
    arg_type_vocab.save(arg_type_vocab_file)
    print(arg_type_vocab)

    print('--------action_vocab---------------')
    action_vocab = Vocab()
    action_vocab.add_spec_toks(pad_tok=False, unk_tok=False)
    action_vocab.add_counter(Counter(actions_list))
    action_vocab.save(action_vocab_file)
    print(action_vocab)

    print('--------pos_vocab---------------')
    pos_vocab = Vocab()
    pos_vocab.add_spec_toks(pad_tok=False, unk_tok=False)
    pos_vocab.add_counter(Counter(pos_list))
    pos_vocab.save(pos_vocab_file)
    print(pos_vocab)




def construct_instance(inst_list, token_vocab, char_vocab, ent_type_vocab,
                       ent_ref_vocab, tri_type_vocab, arg_type_vocab, action_vocab,
                       pos_vocab, is_train=True):
    word_num = 0
    processed_inst_list = []
    sample_sent_total = 2000
    sample_sent_num = 0
    for inst in inst_list:
        words = inst['nlp_words']
        tris = inst['Triggers'] # (idx, event_type)
        ents = inst['Entities'] # (start, end, coarse_type, ref_type)
        args = inst['Arguments'] # (ent_start, ent_end, trigger_idx, argument_type)
        pos = inst['nlp_pos']
        deps = inst['nlp_deps']

        # if is_train and len(tris) == 0:
        #     if len(ents) > 0 and sample_sent_num < sample_sent_total:
        #         sample_sent_num += 1
        #     else:
        #         continue

        # Empirically filter out sentences where event size is 0 or entity size less than 3 (for traning)
        if is_train and len(tris) == 0 and len(ents) < 3: continue

        words_processed = []
        word_indices = []
        char_indices = []
        for word in words:
            word = normalize_tok(word, lower_case, normalize_digits)
            words_processed.append(word)
            word_idx = token_vocab.get_index(word)
            word_indices.append(word_idx)
            char_indices.append([char_vocab.get_index(c) for c in word])

        del inst['Sent']
        inst['words'] = words_processed
        inst['word_indices'] = word_indices
        inst['char_indices'] = char_indices

        inst['pos_indices'] = [pos_vocab.get_index(p) for p in pos]

        inst['tri_indices'] = [[tri[0], tri_type_vocab.get_index(tri[1].lower())] for tri in tris]

        inst['ent_indices'] = [[ent[0], ent[1], ent_type_vocab.get_index(ent[2]),
                                ent_ref_vocab.get_index(ent[3])] for ent in ents]

        collapsed_args = []
        for arg in args:
            collapsed_type = collapse_role_type(arg[3]).lower()
            collapsed_args.append([arg[0], arg[1], arg[2], collapsed_type])
        inst['Arguments'] = collapsed_args

        inst['arg_indices'] = [[arg[0], arg[1], arg[2], arg_type_vocab.get_index(arg[3])]
                               for arg in collapsed_args]

        actions = Actions.make_oracle(words, tris, ents, collapsed_args)
        inst['actions'] = actions
        inst['action_indices'] = [action_vocab.get_index(act) for act in actions]

        inst['sent_range'] = list(range(word_num, word_num + len(words)))
        word_num += len(words)
        processed_inst_list.append(inst)

    return processed_inst_list



def pickle_data():
    token_vocab = Vocab.load(token_vocab_file)
    char_vocab = Vocab.load(char_vocab_file)
    ent_type_vocab = Vocab.load(ent_type_vocab_file)
    ent_ref_vocab = Vocab.load(ent_ref_vocab_file)
    tri_type_vocab = Vocab.load(tri_type_vocab_file)
    arg_type_vocab = Vocab.load(arg_type_vocab_file)
    action_vocab = Vocab.load(action_vocab_file)
    pos_vocab = Vocab.load(pos_vocab_file)

    processed_train = construct_instance(train_list, token_vocab, char_vocab, ent_type_vocab, ent_ref_vocab, tri_type_vocab,
                       arg_type_vocab, action_vocab, pos_vocab)
    processed_dev = construct_instance(dev_list, token_vocab, char_vocab, ent_type_vocab, ent_ref_vocab, tri_type_vocab,
                       arg_type_vocab, action_vocab, pos_vocab, False)
    processed_test = construct_instance(test_list, token_vocab, char_vocab, ent_type_vocab, ent_ref_vocab, tri_type_vocab,
                       arg_type_vocab, action_vocab, pos_vocab, False)

    print('Saving pickle to ', inst_pl_file)
    print('Saving sent size Train: %d, Dev: %d, Test:%d'%(len(processed_train),len(processed_dev),len(processed_test)))
    save_pickle(inst_pl_file, [processed_train, processed_dev, processed_test, token_vocab, char_vocab, ent_type_vocab,
                               ent_ref_vocab, tri_type_vocab, arg_type_vocab, action_vocab,
                               pos_vocab])

    scale = np.sqrt(3.0 / embedd_dim)
    vocab_dict = token_vocab.tok2idx
    table = np.empty([len(vocab_dict), embedd_dim], dtype=np.float32)
    oov = 0
    for word, index in vocab_dict.items():
        if word in embedd_dict:
            embedding = embedd_dict[word]
        elif word.lower() in embedd_dict:
            embedding = embedd_dict[word.lower()]
        else:
            embedding = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
            oov += 1
        table[index, :] = embedding

    np.save(vec_npy_file, table)
    print('pretrained embedding oov: %d' % oov)
    print()



if __name__ == '__main__':
    build_vocab()
    pickle_data()
