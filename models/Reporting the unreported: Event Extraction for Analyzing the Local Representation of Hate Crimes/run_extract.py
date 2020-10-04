import json
import os
from Entity import Entity
from Hi_Attn import *
import argparse


def _load_data(params, args):
    if os.path.isfile("Data/" + args.dataset + "/data.pkl"):
        print("Loading data.pkl for analyzing", args.dataset)
        data = pickle.load(open("Data/" + args.dataset + "/data.pkl", "rb"))
    else:
        print("data.pkl not found. Run run_detect.py first")
        exit(1)

    print("Loading train and test sets")
    return data


def _load_unlabeled(params, args, vocabs):
    if args.goal != "train":
        if os.path.isfile("Data/" + args.dataset + "/predict.pkl"):
            print("Loading unlabeled patch batches")
            unlabeled_batches = pickle.load(open("Data/" + args.dataset + "/predict.pkl", "rb"))
        else:
            print("Batching unlabeled patch data")
            unlabeled_batches = GenerateUnlabeled(vocabs, params["batch_size"], args.dataset)
    else:
        unlabeled_batches = []
    return unlabeled_batches


def _extract(params, data):
    train_batches, dev_batches, test_batches, vocabs, embedding = data
    hate_train_batches = [train for train in train_batches if train["labels"] == 1]
    hate_dev_batches = [dev for dev in dev_batches if dev["labels"] == 1]
    hate_test_batches = [test for test in test_batches if test["labels"] == 1]

    t_weights = np.array([1 - (Counter([train["target_label"] for train in hate_train_batches])[i] /
                               len(hate_train_batches)) for i in range(8)])
    a_weights = np.array([1 - (Counter([train["action_label"] for train in hate_train_batches])[i] /
                               len(hate_train_batches)) for i in range(4)])
    entity = Entity(params, vocabs, embedding)
    entity.build()
    if args.goal == "train":
        entity.run_model(BatchIt(hate_train_batches, params["batch_size"], vocabs),
                         BatchIt(hate_dev_batches, params["batch_size"], vocabs),
                         BatchIt(hate_test_batches, params["batch_size"], vocabs),
                         (t_weights, a_weights))
    elif args.goal == "predict":
        unlabeled_batches = _load_unlabeled(params, args, vocabs)
        target, action = entity.predict(unlabeled_batches, (t_weights, a_weights))
        pickle.dump(target, open("Data/" + args.dataset + "/targets.pkl", "wb"))
        pickle.dump(action, open("Data/" + args.dataset + "/actions.pkl", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--goal", help="Goal can be either train or predict")
    parser.add_argument("--params", help="Path to the params file, a json file "
                                         "that contains model parameters")
    args = parser.parse_args()

    try:
        params = json.load(open(args.params, "r"))
    except Exception:
        print("Error in reading from the provided path, loading the default"
              "parameters instead")
        params = json.load(open("params.json", "r"))

    params["dataset"] = "hate"
    data = _load_data(params, args)
    _extract(params, data)
