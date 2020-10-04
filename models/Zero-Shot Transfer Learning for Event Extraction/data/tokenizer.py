#encoding=utf-8
import os
import jieba
import nltk
import re
import itertools
import unicodedata as ud


class Tokenizer(object):
    def __init__(self, seg_option="linebreak", tok_option="unitok"):
        self.segmenters = {'linebreak': self.seg_linebreak,
                           'nltk': self.seg_nltk,
                           'cmn': self.seg_cmn,
                           'edl_spanish': self.seg_edl_spanish,
                           'edl_cmn': self.seg_edl_cmn,
                           'nltk+linebreak': self.seg_nltk_linebreak,
                           'tigrinya': self.seg_tigrinya
                           }
        self.tokenizers = {'unitok': self.tok_unitok,
                           'unitok_cut': self.tok_unitok_cut,
                           'regexp': self.tok_regexp,
                           'nltk_wordpunct': self.tok_nltk_wordpunct,
                           'space': self.tok_space,
                           'char': self.tok_char,
                           'jieba': self.tok_jieba,
                           }

        self.root_dir = os.path.dirname(os.path.abspath(__file__))

        self.seg_option = seg_option
        self.tok_option = tok_option

        # initialize jieba cn tok
        if tok_option == 'jieba':
            jieba.initialize()

    def run_segmenter(self, plain_text):
        # right strip plain text
        plain_text = plain_text.rstrip()

        # run segmenter
        sents = self.segmenters[self.seg_option](plain_text)

        sents = [s for s in sents if s.strip()]

        return sents

    def run_tokenizer(self, sents):
        # right strip each sent
        for i in range(len(sents)):
            sents[i] = sents[i].rstrip()

        # run tokenizer
        tokenized_sents = self.tokenizers[self.tok_option](sents)

        for i, s in enumerate(tokenized_sents):
            s = [t for t in s if t.strip()]
            tokenized_sents[i] = s

        return tokenized_sents

    #
    # segmenters
    #
    def seg_linebreak(self, plain_text):
        """
        use "\n" as delimiter
        :param plain_text:
        :return:
        """
        result = [item.strip() for item in plain_text.split('\n') if item.strip()]

        return result

    def seg_nltk(self, plain_text):
        """
        use nltk default segmenter
        :param plain_text:
        :return:
        """
        result = [item.strip() for item in nltk.sent_tokenize(plain_text)]

        return result

    def seg_nltk_linebreak(self, plain_text):
        """
        use nltk segmenter and then use "\n" as delimiter to re-segment.
        :param plain_text:
        :return:
        """
        nltk_result = '\n'.join(self.seg_nltk(plain_text))
        linebreak_result = self.seg_linebreak(nltk_result)

        return linebreak_result

    def seg_cmn(self, plain_text):
        """
        use Chinese punctuation as delimiter
        :param plain_text:
        :return:
        """
        res = []
        sent_end_char = [u'。', u'！', u'？']
        current_sent = ''
        for i, char in enumerate(list(plain_text)):
            if char in sent_end_char or i == len(list(plain_text)) - 1:
                res.append(current_sent + char)
                current_sent = ''
            else:
                current_sent += char

        return [item.strip() for item in res]

    def seg_edl(self, plain_text, seg_option):
        # replace \n with ' ' because of the fix line length of edl data
        # plain_text = plain_text.replace('\n', ' ')

        # do sentence segmentation
        if seg_option == 'edl_spanish':
            # use nltk sent tokenization for spanish
            tmp_seg = nltk.sent_tokenize(plain_text)
        if seg_option == 'edl_cmn':
            # use naive sent tokenization for chinese
            tmp_seg = self.seg_cmn(plain_text)

        # recover \n after xml tag
        recovered_tmp_seg = []
        for sent in tmp_seg:
            sent = sent.replace('> ', '>\n').replace(' <', '\n<')
            sent = sent.split('\n')
            recovered_tmp_seg += [item.strip() for item in sent]

        return recovered_tmp_seg

    def seg_edl_spanish(self, plain_text):
        return self.seg_edl(plain_text, 'edl_spanish')

    def seg_edl_cmn(self, plain_text):
        return self.seg_edl(plain_text, 'edl_cmn')

    def seg_tigrinya(self, plain_text):
        result = [item.strip() for item in plain_text.split('\n') if
                  item.strip()]

        updated_result = []
        for r in result:
            if '።' in r:
                sents = []
                start = 0
                for i, char in enumerate(r):
                    if char == '።':
                        sents.append(r[start:i+1])
                        start = i + 1
                updated_result += sents
            else:
                updated_result.append(r)

        return updated_result

    #
    # tokenizers
    #
    def tok_unitok(self, sents):
        res = []
        for s in sents:
            s = unitok_tokenize(s).split()
            res.append(s)

        return res

    def tok_unitok_cut(self, sents):
        res = []
        num_sent_cut = 0
        for s in sents:
            s = unitok_tokenize(s).split()
            if len(s) > 80:
                sub_sents = [item.split() for item in nltk.sent_tokenize(' '.join(s))]
                assert sum([len(item) for item in sub_sents]) == len(s)

                # sub_sent = [list(group) for k, group in
                #             itertools.groupby(s, lambda x: x == ".") if not k]
                res += sub_sents
                if len(sub_sents) > 1:
                    num_sent_cut += 1
            else:
                res.append(s)
        print('%d sentences longer than 80 and cut by delimiter ".".')
        return res

    def tok_regexp(self, sents):
        result = []
        for s in sents:
            tokenizer = nltk.tokenize.RegexpTokenizer('\w+|\$[\d\.]+|\S+')
            tokenization_out = tokenizer.tokenize(s)
            result.append(tokenization_out)

        return result

    def tok_nltk_wordpunct(self, sents):
        result = []
        for s in sents:
            tokenizer = nltk.tokenize.WordPunctTokenizer()
            tokenization_out = tokenizer.tokenize(s)
            result.append(tokenization_out)
        return result

    def tok_space(self, sents):
        result = []
        for s in sents:
            tokenization_out = s.split(' ')
            result.append(tokenization_out)
        return result

    def tok_char(self, sents):
        result = []
        for s in sents:
            tokenization_out = list(s)
            result.append(tokenization_out)
        return result

    def tok_jieba(self, sents):
        result = []
        for s in sents:
            raw_tokenization_out = list(jieba.cut(s))
            result.append(raw_tokenization_out)
        return result


# by Jon May
def unitok_tokenize(data):
    toks = []
    for offset, char in enumerate(data):
        cc = ud.category(char)
        # separate text by punctuation or symbol
        if char in ['ʼ', '’', '‘', '´', '′', "'"]:  # do not tokenize oromo apostrophe
            toks.append(char)
        elif cc.startswith("P") or cc.startswith("S") \
                or char in ['።', '፡']:  # Tigrinya period and comma
            toks.append(' ')
            toks.append(char)
            toks.append(' ')
        else:
            toks.append(char)

    toks = [item for item in ''.join(toks).split() if item]

    return ' '.join(toks)