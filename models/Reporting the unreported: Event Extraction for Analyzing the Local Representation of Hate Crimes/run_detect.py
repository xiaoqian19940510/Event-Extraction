import json
import os
from MICNN import *
from Hi_Attn import *
import argparse

def _load_data(params, args):
    if os.path.isfile("Data/" + args.dataset + "/data.pkl"):
        print("Loading data.pkl for analyzing", args.dataset)
        data = pickle.load(open("Data/" + args.dataset + "/data.pkl", "rb"))
    else:
        print("data.pkl not found. Generating bags ..")
        data = GenerateTrain(params["batch_size"], args.dataset)

    print("Loading train and test sets")
    return data

def _load_unlabeled(params, args, vocabs):
    if args.goal != "train":
        if os.path.isfile("Data/" + args.dataset + "/patch.pkl"):
            print("Loading unlabeled patch batches")
            unlabeled_batches = pickle.load(open("Data/" + args.dataset + "/patch.pkl", "rb"))
        else:
            print("Batching unlabeled patch data")
            unlabeled_batches = GenerateUnlabeled(vocabs, params["batch_size"], args.dataset)
    else:
        unlabeled_batches = []
    return unlabeled_batches

def _detect(params, data):
    train_batches, dev_batches, test_batches, vocabs, embedding = data
    unlabeled_batches = _load_unlabeled(params, args, vocabs)

    if args.model == "MICNN":
        model = MICNN(params, vocabs, embedding)
    elif args.model == "ATTN":
        model = Hi_Attn(params, vocabs, embedding)

    model.build()

    if args.goal == "train":
        train_sent, dev_sent, test_sent = model.run_model(
            BatchIt(train_batches, params["batch_size"], vocabs),
            BatchIt(dev_batches, params["batch_size"], vocabs),
            BatchIt(test_batches, params["batch_size"], vocabs))
        best_sent = [train_sent, test_sent, dev_sent]
        if train_sent:
            for j, articles in enumerate([train_batches, test_batches, dev_batches]):
                for i in range(len(articles)):
                    articles[i]["best_sent"] = best_sent[j][i]

            pickle.dump((train_batches, dev_batches, test_batches, vocabs, embedding),
                        open("Data/" + args.dataset + "/data.pkl", "wb"))

    elif args.goal == "predict":
        hate_pred, indice_pred = model.predict(unlabeled_batches)
        pickle.dump(hate_pred, open("Data/" + args.dataset + "/labels.pkl", "wb"))
        hate_batches = list()
        for i in range(len(unlabeled_batches)):
            if hate_pred[i] == 1:
                if args.model == "MICNN":
                    hate_batches.append(unlabeled_batches[i] + (hate_pred[i], indice_pred[i], i))
                else:
                    hate_batches.append(unlabeled_batches[i] + (hate_pred[i], i))
        pickle.dump(hate_batches, open("Data/" + args.dataset + "/predict.pkl", "wb"))

    elif args.goal == "active":
        probabilities = model.predict(unlabeled_batches, active_learning=True)
        pickle.dump(probabilities, open("probability.pkl", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", help = "Model is either MICNN or ATTN")
    parser.add_argument("--goal", help = "Goal can be either train or predict")
    parser.add_argument("--dataset", help="Dataset is either hate, homicide or kidnap")
    parser.add_argument("--params", help = "Path to the params file, a json file "
                                           "that contains model parameters")

    args = parser.parse_args()
    
    try:
        params = json.load(open(args.params, "r"))
    except Exception:
        print("Error in reading from the provided path, loading the default"
              "parameters instead")
        params = json.load(open("params.json", "r"))

    params["dataset"] = args.dataset
    data = _load_data(params, args)
    _detect(params, data)
