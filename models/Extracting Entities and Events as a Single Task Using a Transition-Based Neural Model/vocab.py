import pickle
from collections import Counter

class Vocab(object):
    PAD = '*PAD*'
    UNK = '*UNK*'
    NULL = '*NULL*'
    START = '*START*'
    END = '*END*'
    ROOT = '*ROOT*'

    def __init__(self):

        self.tok2idx = {}
        self.idx2count = {}
        self.idx2tok = {}
        self.special_token_size = 0
        self.singleton_size = 0
        self.singleton_max_count = 0
        self.min_count = 0
        self.max_count = 0

    def __len__(self):
        return len(self.tok2idx)

    def items(self):
        for k, v in self.tok2idx.items():
            yield k, v

    def __getitem__(self, item):
        return self.tok2idx[item]

    def keys(self):
        return self.tok2idx.keys()

    def vals(self):
        return self.tok2idx.values()

    def add_counter(self,
                    counter,
                    min_count = 1,
                    max_count = 1e7,
                    singleton_max_count=1,
                    update_count = False
                    ):
        '''
        :param counter:
        :param min_count:
        :param max_count:
        :param singleton_max_count: int, we treat a token as a singleton
            when 0 < token count <= singleton_max_count,
            this is in favor of some UNK replace strategies
        :return:
        '''
        self.min_count = min_count
        self.max_count = max_count
        self.singleton_max_count = singleton_max_count

        for tok, count in counter.most_common(n=int(max_count)):
            if count >= self.min_count:
                self.add_token(tok, count, update_count)


    def add_spec_toks(self,
                      pad_tok = True,
                      unk_tok = True,
                      start_tok = False,
                      end_tok = False,
                      root_tok = False,
                      null_tok = False):
        if pad_tok:
            self.add_token(Vocab.PAD)
            self.special_token_size += 1

        if unk_tok:
            self.add_token(Vocab.UNK)
            self.special_token_size += 1

        if start_tok:
            self.add_token(Vocab.START)
            self.special_token_size += 1

        if end_tok:
            self.add_token(Vocab.END)
            self.special_token_size += 1

        if root_tok:
            self.add_token(Vocab.ROOT)
            self.special_token_size += 1

        if null_tok:
            self.add_token(Vocab.NULL)
            self.special_token_size += 1


    def add_token(self, token, count = 1, update_count=False):
        idx = self.tok2idx.get(token, None)
        if idx is None:
            idx = len(self.tok2idx)
            self.tok2idx[token] = idx
            self.idx2count[idx] = count
            if count <= self.singleton_max_count:
                self.singleton_size += 1
        elif update_count:
            new_count = self.idx2count[idx] + count
            self.idx2count[idx] = new_count
            if new_count > self.singleton_max_count:
                self.singleton_size -= 1

        return idx


    def get_vocab_size(self):
        return len(self.tok2idx)


    def get_vocab_size_without_spec(self):
        return len(self.tok2idx) - self.special_token_size


    def get_index(self, token, default_value = '*UNK*'):
        idx = self.tok2idx.get(token, None)
        if idx is None:
            if default_value:
                return self.tok2idx[default_value]
            else:
                raise RuntimeError('Token %s not found'%token)
        else:
            return idx

    def __iter__(self):
        for tok, idx in self.tok2idx.items():
            yield idx, tok


    def get_token(self, index):
        if len(self.idx2tok) == 0:
            for tok, idx in self.tok2idx.items():
                self.idx2tok[idx] = tok

        return self.idx2tok[index]

    def get_token_set(self):
        return self.tok2idx.keys()


    def recount_singleton_size(self):
        singleton_size = 0
        for count in self.idx2count.values():
            if count <= self.singleton_max_count:
                singleton_size += 1
        self.singleton_max_count = singleton_size


    def get_singleton_size(self, re_count=False):
        if re_count:
            self.recount_singleton_size()
        return self.singleton_size


    def is_singleton(self, token_or_index):
        if isinstance(token_or_index, str):
            idx = self.tok2idx.get(token_or_index, None)
            if idx is None:
                # we treat OOV as singletons
                return True
        elif isinstance(token_or_index, int):
            idx = token_or_index
        else:
            raise TypeError('Unknown type %s'%(type(token_or_index)))

        count = self.idx2count[idx]
        return count <= self.singleton_max_count


    def __contains__(self, token):
        return token in self.tok2idx


    def __str__(self):
        spec_tok_size_str = 'special_token_size\t' + str(self.special_token_size) + '\n'
        tok_size_str = 'token_size\t' + str(self.get_vocab_size_without_spec()) + '\n'
        singleton_size_str = 'singleton_size\t' + str(self.singleton_size) + '\n'
        singleton_max_count_str = 'singleton_max_count\t' + str(self.singleton_max_count) + '\n'
        return spec_tok_size_str + tok_size_str + singleton_size_str + singleton_max_count_str


    def save(self, file_path, format='text'):
        '''
        :param format: 'pickle' or 'text'
        :return:
        '''

        if format == 'pickle':
            with open(file_path, 'wb') as file:
                pickle.dump(self, file)

        elif format == 'text':
            with open(file_path, 'w') as file:
                file.write(str(self))
                # write tokens by index increase order
                tok2idx_list = sorted(list(self.tok2idx.items()), key=lambda x:x[1])
                for tok, idx in tok2idx_list:
                    count = self.idx2count[idx]
                    file.write(str(idx)+ '\t' + tok + '\t' + str(count) +'\n')
        else:
            raise RuntimeError('Unknown save format')


    @staticmethod
    def load(file_path, format='text'):

        if format == 'pickle':
            with open(file_path, 'rb') as file:
                return pickle.load(file)

        elif format == 'text':
            with open(file_path, 'r') as file:
                vocab = Vocab()
                lines = list(file.readlines())
                spec_tok_size = int(lines[0].split('\t')[1])
                tok_size = int(lines[1].split('\t')[1])
                singleton_size = int(lines[2].split('\t')[1])
                singleton_max_count = int(lines[3].split('\t')[1])

                vocab.special_token_size = spec_tok_size
                vocab.singleton_size = singleton_size
                vocab.singleton_max_count = singleton_max_count

                offset = 4 # skip head information
                for i in range(offset, offset + tok_size + spec_tok_size):
                    line_arr = lines[i].strip().split('\t')
                    idx, tok, count = int(line_arr[0]), line_arr[1], int(line_arr[2])
                    vocab.tok2idx[tok] = idx
                    vocab.idx2count[idx] = count
                return vocab

        else:
            raise RuntimeError('Unknown load format')
