import numpy as np
import dynet as dy
import nn


def cat(xs, dim=-1):
    head_shape, batch_size = xs[0].dim()
    if dim > len(head_shape):
        raise RuntimeError('Bad dim %d for shape %s, you can '
                           'use -1 to indicate last dim '
                           % (dim, str(head_shape)))
    if dim == -1:
        dim = len(head_shape)
    return dy.concatenate(xs, d=dim)



def expand_dims(x, dim=-1):
    head_shape, batch_size = x.dim()
    if dim > len(head_shape):
        raise RuntimeError('Bad dim %d for shape %s, you can '
                           'use -1 to indicate last dim '
                           %(dim, str(head_shape)))
    if dim == -1:
        dim = len(head_shape)
    ex_shape = list(head_shape)
    ex_shape.insert(dim, 1)
    return dy.reshape(x, tuple(ex_shape))


def layer_norm(xs):
    head_shape, batch_size = xs[0].dim()
    g = dy.ones(head_shape)
    b = dy.zeros(head_shape)
    return [dy.layer_norm(x, g, b) for x in xs]


def squeeze(x, dim=None):
    head_shape, batch_size = x.dim()
    if dim is None:
        sq_shape = [d for d in head_shape if d != 1]
    else:
        if dim > len(head_shape):
            raise RuntimeError('Bad dim %d for shape %s, you can '
                               'use -1 to indicate last dim. Hint: '
                               'you can not squeeze batch dim due to dynet mechanism'
                               % (dim, str(head_shape)))
        if head_shape[dim] != 1:
            raise RuntimeError('You can not squeeze dim %d for shape %s' % (dim, str(head_shape)))
        sq_shape = list(head_shape)
        sq_shape.pop(dim)
    return dy.reshape(x , tuple(sq_shape))

def sum(x, dim=None, include_batch_dim=False):
    if isinstance(x, list):
        return dy.esum(x)
    head_shape, batch_size = x.dim()
    if dim is None:
        x =  dy.sum_elems(x)
        if include_batch_dim and batch_size > 1:
            return dy.sum_batches(x)
        else:
            return x
    else:
        if dim == -1:
            dim = len(head_shape) - 1
        return dy.sum_dim(x, d=[dim], b=include_batch_dim)

def mean(x, dim=None, include_batch_dim=False):
    if isinstance(x, list):
        return dy.average(x)
    head_shape, batch_size = x.dim()
    if dim is None:
        # warning: dynet only implement 2 or lower dims for mean_elems
        x =  dy.mean_elems(x)
        if include_batch_dim and batch_size > 1:
            return dy.mean_batches(x)
        else:
            return x
    else:
        if dim == -1:
            dim = len(head_shape) - 1
        return dy.mean_dim(x, d=[dim], b=include_batch_dim)

def split(x, dim=1):
    head_shape, batch_size = x.dim()
    res = []
    if dim == 0:
        for i in range(head_shape[0]):
            res.append(dy.select_rows(x, [i]))
    elif dim == 1:

        for i in range(head_shape[1]):
            res.append(dy.select_cols(x, [i]))
    return res

def pick_mat(x, row_idx, col_idx):
    return x[row_idx][col_idx]

def logsumexp_dim(x, dim=0):
    return dy.logsumexp_dim(x, d=dim)

# def logsumexp(x):
#     return dy.logsumexp(x)

def log_sum_exp(scores, n_tags):
    npval = scores.npvalue()
    argmax_score = np.argmax(npval)
    max_score_expr = dy.pick(scores, argmax_score)
    max_score_expr_broadcast = dy.concatenate([max_score_expr] * n_tags)
    return max_score_expr + dy.log(dy.sum_cols(dy.transpose(dy.exp(scores - max_score_expr_broadcast))))

def dropout_list(rep_list, dp_rate):
    return [dy.dropout(rep, dp_rate) for rep in rep_list]

def dropout_dim_list(rep_list, dp_rate, dim=0):
    return [dy.dropout_dim(rep, dim, dp_rate) for rep in rep_list]

def cat_list(rep_list_a, rep_list_b, dim=0):
    return [dy.concatenate([rep_a, rep_b], d=dim) for rep_a, rep_b in zip(rep_list_a, rep_list_b)]

def add_list(rep_list_a, rep_list_b):
    return [rep_a + rep_b for rep_a, rep_b in zip(rep_list_a, rep_list_b)]

def sum_list(rep_list_a, rep_list_b):
    return [rep_a+rep_b for rep_a, rep_b in zip(rep_list_a, rep_list_b)]

def binary_cross_entropy(x, y):
    max_val = nn.relu(-x)
    loss = x - dy.cmult(x, y) + max_val + dy.log(dy.exp(-max_val) + dy.exp(-x - max_val))
    return nn.mean(loss)

def max_np(np_vec):
    np_vec = np_vec.flatten()
    return np.max(np_vec), np.argmax(np_vec)
