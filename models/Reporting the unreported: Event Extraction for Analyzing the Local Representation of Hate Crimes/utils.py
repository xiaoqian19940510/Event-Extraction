from nltk.corpus import stopwords
import re
import numpy as np
import nltk.tokenize as tokenizer
from collections import Counter
import operator
from tqdm import tqdm
from nltk import sent_tokenize

def get_vocabs(df):
    data = [tokenizer.TreebankWordTokenizer().tokenize(sent) for sent in df]
    dictionary = Counter([word.lower() for sent in data for word in sent])
    words, counts = zip(*sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True))
    vocab = list(words[:10000]) + ["<unk>", "<pad>"]
    print("vocab size:", len(vocab))
    return vocab

def read_embedding(vocab):
    # reads an embedding file and return a dictionary of word: vector
    with open("embeddings/glove.300.txt", 'r') as file:
        vectors = dict()
        for line in file:
            tokens = line.split()
            vec = np.array(tokens[len(tokens) - 300:], dtype=np.float32)
            token = "".join(tokens[:len(tokens) - 300])
            vectors[token] = vec
    unk_embedding = np.random.rand(300) * 2. - 1.
    embedding = dict()
    for v in vocab:
        try:
            embedding[v] = vectors[v]
        except Exception:
            # if the word is not in the embeddings, use the random vector
            embedding[v] = unk_embedding
    return np.array(list(embedding.values()))

def bag_to_ids(dic, bag, max_length):
    i_bag = list()
    max_len = min(max(len(sent) for sent in bag), max_length)
    lengths = list()
    for sent in bag[:min(200, len(bag))]:
        i_sent = list()
        for word in sent[:min(max_len, len(sent))]:
            try:
                i_sent.append(dic[word.lower()])
            except Exception:
                i_sent.append(dic["<unk>"])
        lengths.append(len(i_sent))
        while len(i_sent) < max_len:
            i_sent.append(dic["<pad>"])
        i_bag.append(i_sent)
    return i_bag, len(bag), lengths

def stop_words(sent):
    stop_words = set(stopwords.words('english'))
    stop_words_exp = re.compile(r"({})\s+".format('|'.join(stop_words)))
    try:
        new_sent = stop_words_exp.sub(' ', sent)
    except TypeError:
        print(sent)
        new_sent = []
    return new_sent


def clean(sent):
    sent = stop_words(sent)
    return sent

def TrainToBags(df, vocab, test=False, max_length=300):
    dictionary = {word: idx for idx, word in enumerate(vocab)}
    bags = list()
    print("Cleaning data ...")
    with tqdm(total=df.shape[0]) as counter:
        for idx, row in df.iterrows():
            words = [tokenizer.TreebankWordTokenizer().tokenize(sent) for sent in sent_tokenize(row["text"])]
            bag, sentences, lengths = bag_to_ids(dictionary, words, max_length)
            if test:
                bags.append({"article": bag,
                             "lengths": lengths,
                             "sent_lengths": sentences})
            else:
                bags.append({"article": bag,
                             "lengths": lengths,
                             "labels": row["labels"],
                             "target_label": row["target"],
                             "action_label": row["action"],
                             "sent_lengths": sentences})
            counter.update(1)
    return bags

def BatchIt(bags, batch_size, vocab, unlabeled=False):
    batches = list()
    for idx in range(len(bags) // batch_size + 1):
        batch = bags[idx * batch_size: min((idx + 1) * batch_size, len(bags))]
        try:
            max_bag = max([len(bag["article"]) for bag in batch])
            max_len = max([len(sent) for bag in batch for sent in bag["article"]])
        except Exception:
            continue
        for bag in batch:
            padding = [vocab.index("<pad>") for i in range(max_len)]
            sub_pad = [vocab.index("<pad>") for i in range(max_len - len(bag["article"][0]))]
            for sent in bag["article"][:bag["sent_lengths"]]:
                sent.extend(sub_pad)
            if len(bag["article"]) > bag["sent_lengths"]:
                bag["article"][bag["sent_lengths"]].extend(sub_pad)
            while len(bag["article"]) < max_bag:
                bag["article"].append(padding)
                bag["lengths"].append(0)
        batches.append(batch)
    return batches
