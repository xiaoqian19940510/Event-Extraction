import pandas as pd
from utils import *
import pickle
from sklearn.model_selection import train_test_split

def GenerateUnlabeled(vocabs, batch_size, dataset):
    print("Loading the unlabeled dataset")
    patch = pd.read_csv("Data/patch_more3.csv")

    print("Unlabeled set shape:", patch.shape)

    print("Converting unlabeled dataset to batches")

    test = TrainToBags(patch, vocabs, True)
    print("Saving the unlabeled dataset into patch.pkl")

    pickle.dump(test, open("Data/" + dataset + "/patch.pkl", "wb"))
    return test

def GenerateTrain(batch_size, dataset):
    df = pd.read_csv("Data/" + dataset + "/train_" + dataset + ".csv")
    print("Train set includes", df.shape[0], "annotated data points")

    print("Learning vocabs")
    vocabs = get_vocabs(df["text"].tolist())

    print("Converting articles to bags of sentences")
    bags = TrainToBags(df, vocabs)

    print("Splitting into train and dev set")
    train, dev_test = train_test_split(bags, test_size=0.3, random_state=33)
    test, dev = train_test_split(dev_test, test_size=0.33, random_state=33)

    print("Loading pretrained word embeddings")
    embedding = read_embedding(vocabs)

    print("All datasets are saved in data.pkl")
    pickle.dump((train, dev, test, vocabs, embedding), open("Data/" + dataset + "/data.pkl", "wb"))
    return (train, dev, test, vocabs, embedding)
