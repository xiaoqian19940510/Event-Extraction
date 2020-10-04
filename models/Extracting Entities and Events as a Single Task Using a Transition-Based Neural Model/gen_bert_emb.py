from flair.data import Sentence
from flair.models import SequenceTagger

from flair.embeddings import CharLMEmbeddings, StackedEmbeddings, BertEmbeddings
import os
import pickle

import numpy as np
from io_utils import read_yaml, read_lines, read_json_lines

data_config = read_yaml('data_config.yaml')

data_dir = data_config['data_dir']
ace05_event_dir = data_config['ace05_event_dir']

train_list = read_json_lines(os.path.join(ace05_event_dir, 'train_nlp_ner.json'))
dev_list = read_json_lines(os.path.join(ace05_event_dir, 'dev_nlp_ner.json'))
test_list = read_json_lines(os.path.join(ace05_event_dir, 'test_nlp_ner.json'))

train_sent_file = data_config['train_sent_file']

bert = BertEmbeddings(layers='-1', bert_model_or_path='bert-base-uncased').to('cuda:0')

def save_bert(inst_list, filter_tri=True, name='train'):
    sents = []
    sent_lens = []
    for inst in inst_list:
        words, trigger_list, ent_list, arg_list = inst['nlp_words'], inst['Triggers'], inst['Entities'], inst['Arguments']
        # Empirically filter out sentences where event size is 0 or entity size less than 3 (for traning)
        if len(trigger_list) == 0 and len(ent_list) < 3 and filter_tri: continue
        sents.append(words)
        sent_lens.append(len(words))

    total_word_nums = sum(sent_lens)
    input_table = np.empty((total_word_nums,768 * 1))
    acc_len = 0
    for i, words in enumerate(sents):
        if i % 100 ==0:
            print('progress: %d, %d'%(i, len(sents)))
        sent_len = sent_lens[i]
        flair_sent = Sentence(' '.join(words))
        bert.embed(flair_sent)
        for j, token in enumerate(flair_sent):
            start = acc_len + j
            input_table[start, :] = token.embedding.cpu().detach().numpy()
        acc_len += sent_len

    bert_fname = data_config['train_sent_file'] if name == 'train' else \
                data_config['dev_sent_file'] if name == 'dev' else data_config['test_sent_file']
    np.save(bert_fname, input_table)

    print('total_word_nums:', total_word_nums)
    #print(len(sent_lens))


if __name__ == "__main__":
    save_bert(train_list, name='train')
    save_bert(dev_list,filter_tri=False, name='dev')
    save_bert(test_list,filter_tri=False, name='test')
