import scipy
from nltk.metrics import spearman, spearman_correlation
import numpy as np
from collections import OrderedDict
from web.utils import batched
import pandas as pd

def get_word_embedding(word, word_embedding, vocab, default):
    '''
    Get word embeddings.

    parameters:
    - word: query word.
    - word_embedding: expected to be a matrix with size num_words*embed_size, which
    stores all the embedding.
    - vocab: corresponding vocubulary of all words.
    - default: vector that used to represent words out of vocabulary

    return:
    - word embeddings
    '''

    if word not in vocab:
        return default
    else:
        return word_embedding[vocab[word]]


def evaluate_similarity(word_embedding, vocab, x, y):
    '''
    Calculate the Spearman correlation between cosine similarity of the model
    and human rated similarity of word pairs

    parameters:
    - word_embedding: expected to be a matrix with size num_words*embed_size, which
    stores all the embedding.
    - vocab: corresponding vocubulary of all words.
    - x: an array that stores all the pairs of similar words, shape: (n_sample, 2).
    - y: an array that stores all the human ratings, shape: (n_sample, ).

    return:
    - Spearman correlation cosine similarity of the model and human rated similarity
    '''

    missing_words = 0
    mean_vector = np.mean(word_embedding, axis=0, keepdims=True)
    for query in x:
        for query_word in query:
            if query_word not in vocab:
                missing_words += 1
    if missing_words > 0:
        print("Missing {} words. Will replace them with mean vector".format(missing_words))

    mean_vector = np.mean(word_embedding, axis=0, keepdims=True)
    A = np.vstack([get_word_embedding(word, word_embedding, vocab, mean_vector)
                   for word in x[:, 0]])
    B = np.vstack([get_word_embedding(word, word_embedding, vocab, mean_vector)
                   for word in x[:, 1]])
    scores = np.array([v1.dot(v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                       for v1, v2 in zip(A, B)])

    return scipy.stats.spearmanr(scores, y).correlation

    # rank1 = [(i,np.where(scores.argsort()==i)[0].item()) for i in range(len(scores))]
    # rank2 = [(i,np.where(y.argsort()==i)[0].item()) for i in range(len(y))]
    # return spearman_correlation(rank1, rank2)



def solve_analogy(word_embeddings, vocab, X, batch_size=100, method='add'):
    word_list = vocab.get_itos()
    word_id = vocab.get_stoi()
    mean_vector = np.mean(word_embeddings, axis=0)
    output = []

    missing_words = 0
    for query in X:
        for query_word in query:
            if query_word not in word_id:
                missing_words += 1
    if missing_words > 0:
        print("Missing {} words. Will replace them with mean vector".format(missing_words))

    for id_batch, batch in enumerate(batched(range(len(X)), batch_size)):
        ids = list(batch)
        X_b = X[ids]
        if id_batch % np.floor(len(X) / (10. * batch_size)) == 0:
            print("Processing {}/{} batch".format(int(np.ceil(ids[1] / float(batch_size))),
                                                  int(np.ceil(X.shape[0] / float(batch_size)))))

        A = np.vstack(get_word_embedding(word, word_embeddings, vocab, mean_vector) for word in X_b[:, 0])
        B = np.vstack(get_word_embedding(word, word_embeddings, vocab, mean_vector) for word in X_b[:, 1])
        C = np.vstack(get_word_embedding(word, word_embeddings, vocab, mean_vector) for word in X_b[:, 2])

        if method == "add":
            D = np.dot(word_embeddings, (B - A + C).T)
        elif method == "mul":
            D_A = np.log((1.0 + np.dot(word_embeddings, A.T)) / 2.0 + 1e-5)
            D_B = np.log((1.0 + np.dot(word_embeddings, B.T)) / 2.0 + 1e-5)
            D_C = np.log((1.0 + np.dot(word_embeddings, C.T)) / 2.0 + 1e-5)
            D = D_B - D_A + D_C
        else:
            raise RuntimeError("Unrecognized method parameter")

        # Remove words that were originally in the query
        for id, row in enumerate(X_b):
            D[[vocab([r])[0] for r in row if r in word_list], id] = np.finfo(np.float32).min

        output.append([word_list[id] for id in D.argmax(axis=0)])

    return np.array([item for sublist in output for item in sublist])



def evaluate_analogy(word_embeddings, vocab, X, y, method='add', batch_size=100, category=None):
    y_pred = solve_analogy(word_embeddings, vocab, X, batch_size=batch_size, method=method)

    if category is not None:
        results = OrderedDict({"all": np.mean(y_pred == y)})
        count = OrderedDict({"all": len(y_pred)})
        correct = OrderedDict({"all": np.sum(y_pred == y)})
        for cat in set(category):
            results[cat] = np.mean(y_pred[category == cat] == y[category == cat])
            count[cat] = np.sum(category == cat)
            correct[cat] = np.sum(y_pred[category == cat] == y[category == cat])

        return pd.concat([pd.Series(results, name="accuracy"),
                          pd.Series(correct, name="correct"),
                          pd.Series(count, name="count")],
                         axis=1)
    else:
        return np.mean(y_pred == y)