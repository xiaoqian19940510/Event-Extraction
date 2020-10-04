import dynet as dy
import numpy as np
from dy_utils import ParamManager as pm
# dy.renew_cg(immediate_compute = True, check_validity = True)


def sigmoid(x):
    return dy.logistic(x)

def tanh(x):
    return dy.tanh(x)

def penalized_tanh(x):
    alpha = 0.25
    tanh_x = dy.tanh(x)
    return dy.bmax(tanh_x, alpha*tanh_x)

def tanh_list(rep_list):
    return [dy.tanh(x) for x in rep_list]

def relu(x):
    return dy.rectify(x)

def relu_list(rep_list):
    return [dy.rectify(x) for x in rep_list]

def cube(x):
    return dy.cube(x)

def selu(x):
    return dy.selu(x)

def elu(x, alpha=1.0):
    return dy.elu(x)

def log_sigmoid(x):
    return dy.log_sigmoid(x)

def softmax(x, dim=0):
    # warning: dynet only implement 2d and 1d softmax
    if dim == -1:
        dim = len(x.dim()[0]) - 1
    return dy.softmax(x, d=dim)

def log_softmax(x):
    # warning: dynet only implement 2d and 1d log_softmax
    return dy.log_softmax(x)


class Linear(object):
    def __init__(self, n_in, n_out, bias=True, activation='linear', model=None, init_w=None):
        if model is None:
            model = pm.global_collection()

        if init_w is not None:
            self.W = model.parameters_from_numpy(init_w)
        else:
            self.W = model.add_parameters((n_out, n_in), init='glorot', name='linearW')
        self.bias = bias
        self.act = activation
        if bias:
            self.b = model.add_parameters((n_out), init=0, name='linearBias')

    def __call__(self, input):
        if isinstance(input, list):
            return [self._compute(x) for x in input]
        else:
            return self._compute(input)

    def _compute(self, input):
        if not self.bias:
            output = self.W * input
        else:
            output = dy.affine_transform([self.b, self.W, input])

        if self.act == 'linear':
            return output
        elif self.act == 'sigmoid':
            return sigmoid(output)
        elif self.act == 'tanh':
            return tanh(output)
        elif self.act == 'ptanh':
            return penalized_tanh(output)
        elif self.act == 'relu':
            return relu(output)
        elif self.act == 'elu':
            return elu(output)
        raise ValueError('Unknown activation function :'+self.act)


class Embedding(object):

    def __init__(self, n_vocab, n_dim, init_weight=None, trainable=True, model=None, name='embed'):
        if model is None:
            model = pm.global_collection()
        self.trainable = trainable
        if init_weight is not None:
            self.embed = model.lookup_parameters_from_numpy(init_weight, name=name)
        else:
            self.embed = model.add_lookup_parameters((n_vocab, n_dim), name=name)

    def __call__(self, input):
        output = [dy.lookup(self.embed, x, update=self.trainable) for x in input]
        # output = dy.lookup_batch(self.embed, input, update=self.trainable)

        return output

    def __getitem__(self, item):
        return dy.lookup(self.embed, item, update=self.trainable)


class MultiLayerLSTM(object):

    def __init__(self, n_in, n_hidden, n_layer=1, bidirectional=False, lstm_params=None, model=None, dropout_x=0., dropout_h=0.):
        if model is None:
            model = pm.global_collection()
        self.bidirectional = bidirectional
        self.n_layer = n_layer
        rnn_builder_factory = LSTMCell
        self.fwd_builders = [rnn_builder_factory(model,n_in, n_hidden, dropout_x, dropout_h)]
        if bidirectional:
            self.bwd_builders = [rnn_builder_factory(model,n_in, n_hidden, dropout_x, dropout_h)]

        hidden_input_dim = n_hidden * 2 if bidirectional else n_hidden
        for _ in range(n_layer - 1):
            self.fwd_builders.append(rnn_builder_factory(model,hidden_input_dim, n_hidden))
            if bidirectional:
                self.bwd_builders.append(rnn_builder_factory(model, hidden_input_dim, n_hidden))

        if lstm_params is not None:
            self._init_param(lstm_params)

    def get_cells(self):
        return self.fwd_builders + self.bwd_builders

    def init_sequence(self, test=False):
        for fwd in self.fwd_builders:
            fwd.init_sequence(test)

        if self.bidirectional:
            for bwd in self.bwd_builders:
                bwd.init_sequence(test)

    def _init_param(self, lstm_params):
        '''
        :param lstm_params: (forward_param_list, backward_param_list),
                            forward_param_list: [[layer1], [layer2], ...]
                            layer1: (Wx, Wh, bais)
        :return:
        '''
        def _set_param_value(builder, Wx, Wh, bais):
            model_Wx, model_Wh, model_bias = builder.get_parameters()[0]
            model_Wx.set_value(Wx)
            model_Wh.set_value(Wh)
            bais = np.zeros_like(bais)
            model_bias.set_value(bais)

        if self.bidirectional:
            lstm_params_fw, lstm_params_bw = lstm_params
            for i in range(self.n_layer):
                fwd_params, bwd_params = lstm_params_fw[i], lstm_params_bw[i]
                builder_fw = self.fwd_builders[i]
                builder_bw = self.bwd_builders[i]
                builder_fw.init_params( fwd_params[0], fwd_params[1], fwd_params[2])
                builder_bw.init_params(bwd_params[0], bwd_params[1], bwd_params[2])
        else:
            lstm_params_fw = lstm_params
            for i in range(self.n_layer):
                fwd_params = lstm_params_fw[i]
                builder_fw = self.fwd_builders[i]
                builder_fw.init_params(fwd_params[0], fwd_params[1], fwd_params[2])


    def transduce(self, input, hx=None, cx=None):
        layer_rep = input
        if self.bidirectional:
            fs, bs = None, None
            for fw, bw in zip(self.fwd_builders, self.bwd_builders):
                fs, _ = fw.transduce(layer_rep, hx, cx)
                bs, _ = bw.transduce(reversed(layer_rep))
                layer_rep = [dy.concatenate([f, b]) for f, b in zip(fs, reversed(bs))]

            return layer_rep, (fs, bs)
        else:
            for fw in self.fwd_builders:
                layer_rep, _ = fw.transduce(layer_rep, hx, cx)
            return layer_rep


    def last_step(self, input, hx=None, cx=None, separated_fw_bw=False):
        layer_rep = input
        if self.bidirectional:
            for fw, bw in zip(self.fwd_builders[:-1], self.bwd_builders[:-1]):
                fs, _ = fw.transduce(layer_rep, hx, cx)
                bs, _ = bw.transduce(reversed(layer_rep))
                layer_rep = [dy.concatenate([f, b]) for f, b in zip(fs, reversed(bs))]

            fw, bw = self.fwd_builders[-1], self.bwd_builders[-1]
            fs, fc = fw.transduce(layer_rep, hx, cx)
            bs, bc = bw.transduce(reversed(layer_rep))
            layer_rep = [dy.concatenate([f, b]) for f, b in zip(fs, reversed(bs))]

            last_rep = dy.concatenate([fs[-1], bs[-1]])
            last_c = dy.concatenate([fc[-1],bc[-1]])

            if not separated_fw_bw:
                return layer_rep, (last_rep, last_c)
            else:
                return  (fw, bw), (last_rep, last_c)
        else:
            for fw in self.fwd_builders:
                layer_rep, _ = fw.transduce(layer_rep, hx, cx)
            return layer_rep, (hx, cx)



class LSTMCell:

    def __init__(self, model, n_in, n_hidden, dropout_x=0., dropout_h=0.):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.dropout_x = dropout_x
        self.dropout_h = dropout_h

        self.weight_ih = model.add_parameters((n_hidden * 4, n_in), name='lstmIH')

        self.weight_hh = model.add_parameters((n_hidden * 4, n_hidden), name='lstmHH')

        self.bias= model.add_parameters((n_hidden * 4), init=0, name='lstmBias')


    def init_params(self, weight_ih=None, weight_hh=None, bias=None):
        if weight_ih is not None:
            self.weight_ih.set_value(weight_ih)

        if weight_hh is not None:
            self.weight_hh.set_value(weight_hh)

        if bias is not None:
            self.bias.set_value(bias)

    def transduce(self, input, hx=None, cx=None):
        hx = hx if hx is not None else dy.zeros((self.n_hidden))
        cx = cx if cx is not None else dy.zeros((self.n_hidden))
        output = []
        cells = []
        for x in input:
            hx, cx = self.step(x, hx, cx)
            output.append(hx)
            cells.append(cx)

        return output, cells

    def init_sequence(self, test=False):
        self.test = test
        if not test:
            self.dropout_mask_x = dy.dropout(
                dy.ones((self.n_in,)), self.dropout_x
            )
            self.dropout_mask_h = dy.dropout(
                dy.ones((self.n_hidden,)), self.dropout_h
            )


    def step(self, x, hx, cx):
        if not self.test:
            if self.dropout_x > 0:
                x = dy.cmult(self.dropout_mask_x, x)
            if self.dropout_h > 0:
                hx = dy.cmult(self.dropout_mask_h, hx)

        gates = dy.affine_transform([self.bias, self.weight_ih, x, self.weight_hh, hx])
        i = dy.pickrange(gates, 0, self.n_hidden)
        f = dy.pickrange(gates, self.n_hidden, self.n_hidden * 2)
        g = dy.pickrange(gates, self.n_hidden * 2, self.n_hidden * 3)
        o = dy.pickrange(gates, self.n_hidden * 3, self.n_hidden * 4)

        i, f, g, o = dy.logistic(i), dy.logistic(f), dy.tanh(g), dy.logistic(o)
        cy = dy.cmult(f, cx) + dy.cmult(i, g)
        hy = dy.cmult(o, dy.tanh(cy))
        return hy, cy



class LayerNorm:

    def __init__(self, n_hid, model=None):
        if model is None:
            model = pm.global_collection()
        self.p_g = model.add_parameters(dim=n_hid, init=dy.ConstInitializer(1.0))
        self.p_b = model.add_parameters(dim=n_hid, init=dy.ConstInitializer(0.0))

    def transform(self, x):
        g = self.p_g
        b = self.p_b
        return dy.layer_norm(x, g, b)

    def __call__(self, input):
        if isinstance(input, list):
            return [self.transform(x) for x in input]
        else:
            return self.transform(input)



class StackLSTM(object):

    def __init__(self, input_size, hidden_size, dropout_x=0., dropout_h=0.):
        super(StackLSTM, self).__init__()
        self.hidden_size = hidden_size
        model = pm.global_collection()
        self.cell = LSTMCell(model, input_size, hidden_size, dropout_x, dropout_h)
        self.empty_embedding = model.add_parameters((self.hidden_size,), name='stackGuardEmb')
        self.states = []
        self.indices = []

        #self.linear = nn.Linear(input_size, hidden_size, activation='tanh')

    def init_sequence(self, test=False):
        self.cell.init_sequence(test)

    def get_reverse_hx(self):
        rev_hx = []
        for i in range(len(self.states) - 1, -1, -1):
            rev_hx.append(self.states[i][0])
        return rev_hx

    def iter(self):
        for (hx, cx), idx in zip(self.states, self.indices):
            yield  hx, idx

    def push(self, input, idx):
        '''
        :param input:
        :param idx: word idx in buffer or action_id in vocab
        :return:
        '''
        if len(self.states) == 0:
            init_h, init_c = dy.zeros((self.hidden_size)), dy.zeros((self.hidden_size))
            hx, cx = self.cell.step(input, init_h, init_c)
        else:
            pre_hx, pre_cx = self.states[-1]
            hx, cx = self.cell.step(input, pre_hx, pre_cx)

        self.states.append((hx, cx))

        #self.states.append((self.linear(input), None))
        self.indices.append(idx)

    def pop(self):
        if len(self.states) == 0:
            raise RuntimeError('Empty states')
        hx, cx = self.states.pop()
        idx = self.indices.pop()
        return hx, idx

    def last_state(self):
        return self.states[-1][0], self.indices[-1]

    def all_h(self):
        return [s[0] for s in self.states]

    def clear(self):
        self.states.clear()
        self.indices.clear()


    def embedding(self):
        if len(self.states) == 0:
            hx = self.empty_embedding
            # hx = dy.zeros((self.hidden_size,))
        else:
            hx, cx = self.states[-1]

        return hx

    def is_empty(self):
        return len(self.states) == 0

    def idx_range(self):
        return self.indices[0], self.indices[-1]

    def last_idx(self):
        return self.indices[-1]

    def __getitem__(self, item):
        hx, cx = self.states[item]
        idx = self.indices[item]

        return hx, idx

    def __len__(self):
        return len(self.states)

    def __str__(self):
        return str(len(self.states)) + ':' + str(len(self.indices))


class Buffer(object):

    def __init__(self, bi_rnn_dim, hidden_state_list):
        '''
        :param state_tensor: list of (n_dim)
        '''

        self.hidden_states = hidden_state_list
        self.seq_len = len(hidden_state_list)
        self.idx = 0

    def pop(self):
        if self.idx == self.seq_len:
            raise RuntimeError('Empty buffer')
        hx = self.hidden_states[self.idx]
        cur_idx = self.idx
        self.idx += 1
        return hx, cur_idx

    def last_state(self):
        return self.hidden_states[self.idx], self.idx

    def buffer_idx(self):
        return self.idx

    def hidden_embedding(self):
        return self.hidden_states[self.idx]

    def hidden_idx_embedding(self, idx):
        return self.hidden_states[idx]


    def is_empty(self):
        return (self.seq_len - self.idx) == 0

    def move_pointer(self, idx):
        self.idx = idx

    def move_back(self):
        self.idx -= 1

    def __len__(self):
        return  self.seq_len - self.idx


class LambdaVar(object):
    TRIGGER = 't'
    ENTITY = 'e'
    OTHERS = 'o'

    def __init__(self, bi_rnn_dim):
        self.var = None
        self.idx = -1
        self.model = pm.global_collection()
        self.bi_rnn_dim = bi_rnn_dim
        self.lmda_empty_embedding = self.model.add_parameters((bi_rnn_dim,))
        self.lambda_type = LambdaVar.OTHERS

    def push(self, embedding, idx, lambda_type):
        self.var = embedding
        self.idx = idx
        self.lambda_type = lambda_type

    def pop(self):
        var, idx =  self.var, self.idx
        self.var, self.idx = None, -1
        self.lambda_type = LambdaVar.OTHERS
        return var, idx

    def clear(self):
        self.var, self.idx = None, -1

    def is_empty(self):
        return self.var is None

    def is_trigger(self):
        return self.lambda_type == LambdaVar.TRIGGER

    def is_entity(self):
        return self.lambda_type == LambdaVar.ENTITY

    def embedding(self):
        # return dy.zeros(self.bi_rnn_dim) if self.var is None else self.var
        return self.lmda_empty_embedding if self.var is None else self.var
