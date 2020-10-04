import numpy
import time
import sys
import subprocess
import os
import random
import cPickle
import copy

import theano
from theano import tensor as T
from collections import OrderedDict, defaultdict
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from theano.updates import OrderedUpdates
import theano.tensor.shared_randomstreams

#########################SOME UTILITIES########################


def randomMatrix(r, c, scale=0.2):
    #W_bound = numpy.sqrt(6. / (r + c))
    W_bound = 1.
    return scale * numpy.random.uniform(low=-W_bound, high=W_bound,\
                   size=(r, c)).astype(theano.config.floatX)

def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(theano.config.floatX)

def _slice(_x, n, dim):
    return _x[:,n*dim:(n+1)*dim]
    
def zeroVec(shape):
    return T.zeros(shape, dtype=theano.config.floatX)

###############################################################

##########################Optimization function################

def adadelta(ips,cost,fupdate,names,parameters,gradients,lr,norm_lim,rho=0.95,eps=1e-6):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in zip(names, parameters)]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rup2'%k) for k, p in zip(names, parameters)]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2'%k) for k, p in zip(names, parameters)]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, gradients)]
    rg2up = [(rg2, rho * rg2 + (1. - rho) * (g ** 2)) for rg2, g in zip(running_grads2, gradients)]
    
    update_map = fupdate if fupdate else OrderedUpdates()
    for kk, vv in zgup: update_map[kk] = vv
    for kk, vv in rg2up: update_map[kk] = vv
    
    f_grad_shared = theano.function(ips, cost, updates=update_map, on_unused_input='ignore')

    updir = [-T.sqrt(ru2 + eps) / T.sqrt(rg2 + eps) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, rho * ru2 + (1. - rho) * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(parameters, updir)]
    
    if norm_lim > 0:
        param_up = clipGradient(param_up, norm_lim, names)
        
    #update_map = fupdate if fupdate else OrderedUpdates()
    #for kk, vv in ru2up: update_map[kk] = vv
    #for kk, vv in param_up: update_map[kk] = vv

    f_param_update = theano.function([lr], [], updates=ru2up+param_up, on_unused_input='ignore')

    return f_grad_shared, f_param_update

def sgd(ips,cost,fupdate,names,parameters,gradients,lr,norm_lim):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) for k, p in zip(names, parameters)]
    gsup = [(gs, g) for gs, g in zip(gshared, gradients)]
    
    update_map = fupdate if fupdate else OrderedUpdates()
    for kk, vv in gsup: update_map[kk] = vv

    f_grad_shared = theano.function(ips, cost, updates=update_map, on_unused_input='ignore')

    pup = [(p, p - lr * g) for p, g in zip(parameters, gshared)]
    
    if norm_lim > 0:
        pup = clipGradient(pup, norm_lim, names)
        
    #update_map = fupdate if fupdate else OrderedUpdates()
    #for kk, vv in pup: update_map[kk] = vv
    
    f_param_update = theano.function([lr], [], updates=pup, on_unused_input='ignore')

    return f_grad_shared, f_param_update

def clipGradient(updates, norm, names):
    id = -1
    res = []
    for p, g in updates:
        id += 1
        if not names[id].startswith('word') and p.get_value(borrow=True).ndim == 2:
            col_norms = T.sqrt(T.sum(T.sqr(g), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm))
            scale = desired_norms / (1e-7 + col_norms)
            g = g * scale
            
        res += [(p, g)]
    return res          

###############################################################

def _dropout_from_layer(rng, layers, p):
    if p <= 0: return layers
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    res = []
    for layer in layers:
        mask = srng.binomial(n=1, p=1-p, size=layer.shape)
        # The cast is important because
        # int * float32 = float64 which pulls things off the gpu
        output = layer * T.cast(mask, theano.config.floatX)
        res += [output]
    return res

def generateBinomial(shape, rng, p):
    if p <= 0: p = 0
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    total = numpy.array(list(shape)).prod()
    mask = srng.binomial(n=1, p=1-p, size=(total,))
    return T.cast(mask.reshape(shape), dtype=theano.config.floatX)

###############################Models###############################

def getConcatenation(embDict, vars, features, features_dim):

    xs = []

    for ed in features:
        if features[ed] == 0:
            var = vars[ed].T
            xs += [embDict[ed][T.cast(var.flatten(), dtype='int32')].reshape((var.shape[0], var.shape[1], features_dim[ed]))]
        elif features[ed] == 1:
            xs += [vars[ed].dimshuffle(1,0,2)]

    if len(xs) == 1:
        basex = xs[0]
    else:
        basex = T.cast(T.concatenate(xs, axis=2), dtype=theano.config.floatX)

    return basex

def getInverseConcatenation(embDict, vars, features, features_dim):
        
    ixs = []

    for ed in features:
        if features[ed] == 0:
            var = vars[ed].T[::-1]
            ixs += [embDict[ed][T.cast(var.flatten(), dtype='int32')].reshape((var.shape[0], var.shape[1], features_dim[ed]))]
        elif features[ed] == 1:
            ixs += [vars[ed].dimshuffle(1,0,2)[::-1]]                

    if len(ixs) == 1:
        ibasex = ixs[0]
    else:
        ibasex = T.cast(T.concatenate(ixs, axis=2), dtype=theano.config.floatX)
    
    return ibasex
    
def rnn_ff(inps, dim, hidden, batSize, prefix, params, names):
    Wx  = theano.shared(randomMatrix(dim, hidden))
    Wh  = theano.shared(randomMatrix(hidden, hidden))
    bh  = theano.shared(numpy.zeros(hidden, dtype=theano.config.floatX))
    #model.container['bi_h0']  = theano.shared(numpy.zeros(model.container['nh'], dtype=theano.config.floatX))

    # bundle
    params += [ Wx, Wh, bh ] #, model.container['bi_h0']
    names += [ prefix + '_Wx', prefix + '_Wh', prefix + '_bh' ] #, 'bi_h0'

    def recurrence(x_t, h_tm1):
        h_t = T.nnet.sigmoid(T.dot(x_t, Wx) + T.dot(h_tm1, Wh) + bh)
        return h_t

    h, _  = theano.scan(fn=recurrence, \
            sequences=inps, outputs_info=[T.alloc(0., batSize, hidden)], n_steps=inps.shape[0])
    
    return h
    
def rnn_gru(inps, dim, hidden, batSize, prefix, params, names):
    Wc = theano.shared(numpy.concatenate([randomMatrix(dim, hidden), randomMatrix(dim, hidden)], axis=1))

    bc = theano.shared(numpy.zeros(2 * hidden, dtype=theano.config.floatX))

    U = theano.shared(numpy.concatenate([ortho_weight(hidden), ortho_weight(hidden)], axis=1))
    Wx = theano.shared(randomMatrix(dim, hidden))

    Ux = theano.shared(ortho_weight(hidden))

    bx = theano.shared(numpy.zeros(hidden, dtype=theano.config.floatX))

    #model.container['bi_h0'] = theano.shared(numpy.zeros(model.container['nh'], dtype=theano.config.floatX))

    # bundle
    params += [ Wc, bc, U, Wx, Ux, bx ] #, model.container['bi_h0']
    names += [ prefix + '_Wc', prefix + '_bc', prefix + '_U', prefix + '_Wx', prefix + '_Ux', prefix + '_bx' ] #, 'bi_h0'
    
    def recurrence(x_t, h_tm1):
        preact = T.dot(h_tm1, U)
        preact += T.dot(x_t, Wc) + bc

        r_t = T.nnet.sigmoid(_slice(preact, 0, hidden))
        u_t = T.nnet.sigmoid(_slice(preact, 1, hidden))

        preactx = T.dot(h_tm1, Ux)
        preactx = preactx * r_t
        preactx = preactx + T.dot(x_t, Wx) + bx

        h_t = T.tanh(preactx)

        h_t = u_t * h_tm1 + (1. - u_t) * h_t

        return h_t

    h, _  = theano.scan(fn=recurrence, \
            sequences=inps, outputs_info=[T.alloc(0., batSize, hidden)], n_steps=inps.shape[0])
    
    return h
    
def ffBidirectCore(inps, iinps, dim, hidden, batSize, prefix, iprefix, params, names):

    bi_h = rnn_ff(inps, dim, hidden, batSize, prefix, params, names)
    
    ibi_h = rnn_ff(iinps, dim, hidden, batSize, iprefix, params, names)

    _ibi_h = ibi_h[::-1]
    
    bi_rep = T.cast(T.concatenate([ bi_h, _ibi_h ], axis=2), dtype=theano.config.floatX) #.dimshuffle(1,0,2)

    return bi_rep
    
def gruBidirectCore(inps, iinps, dim, hidden, batSize, prefix, iprefix, params, names):

    bi_h = rnn_gru(inps, dim, hidden, batSize, prefix, params, names)
    
    ibi_h = rnn_gru(iinps, dim, hidden, batSize, iprefix, params, names)

    _ibi_h = ibi_h[::-1]

    bi_rep = T.cast(T.concatenate([ bi_h, _ibi_h ], axis=2), dtype=theano.config.floatX) #.dimshuffle(1,0,2)

    return bi_rep

def ffForward(embDict, vars, features, features_dim, dimIn, hidden, batch, prefix, params, names):
    ix = getConcatenation(embDict, vars, features, features_dim)
    
    i_h = rnn_ff(ix, dimIn, hidden, batch, prefix, params, names)
    
    rep = i_h
    #rep = T.cast(i_h.dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    return rep

def ffBackward(embDict, vars, features, features_dim, dimIn, hidden, batch, iprefix, params, names):
    iix = getInverseConcatenation(embDict, vars, features, features_dim)
    
    ii_h = rnn_ff(iix, dimIn, hidden, batch, iprefix, params, names)
    
    _ii_h = ii_h[::-1]
    
    rep = _ii_h
    #rep = T.cast(_ii_h.dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    return rep

def ffBiDirect(embDict, vars, features, features_dim, dimIn, hidden, batch, prefix, params, names):
    bix = getConcatenation(embDict, vars, features, features_dim)
    ibix = getInverseConcatenation(embDict, vars, features, features_dim)
    
    return ffBidirectCore(bix, ibix, dimIn, hidden, batch, prefix + '_ffbi', prefix + '_ffibi', params, names)
    
def gruForward(embDict, vars, features, features_dim, dimIn, hidden, batch, prefix, params, names):
    ix = getConcatenation(embDict, vars, features, features_dim)
    
    i_h = rnn_gru(ix, dimIn, hidden, batch, prefix, params, names)
    
    rep = i_h
    #rep = T.cast(i_h.dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    return rep

def gruBackward(embDict, vars, features, features_dim, dimIn, hidden, batch, iprefix, params, names):
    iix = getInverseConcatenation(embDict, vars, features, features_dim)
    
    ii_h = rnn_gru(iix, dimIn, hidden, batch, iprefix, params, names)
    
    _ii_h = ii_h[::-1]
    
    rep = _ii_h
    #rep = T.cast(_ii_h.dimshuffle(1,0,2), dtype=theano.config.floatX)
    
    return rep

def gruBiDirect(embDict, vars, features, features_dim, dimIn, hidden, batch, prefix, params, names):
    bix = getConcatenation(embDict, vars, features, features_dim)
    ibix = getInverseConcatenation(embDict, vars, features, features_dim)
    
    return gruBidirectCore(bix, ibix, dimIn, hidden, batch, prefix + '_grubi', prefix + '_gruibi', params, names)
    
###############################CONVOLUTIONAL CONTEXT####################################

def convolutionalLayer(inpu, feature_map, batch, length, window, dim, prefix, params, names):
    down = window / 2
    up = window - down - 1
    zodown = T.zeros((batch, 1, down, dim), dtype=theano.config.floatX)
    zoup = T.zeros((batch, 1, up, dim), dtype=theano.config.floatX)
    
    inps = T.cast(T.concatenate([zoup, inpu, zodown], axis=2), dtype=theano.config.floatX)
    
    fan_in = window * dim
    fan_out = feature_map * window * dim / length #(length - window + 1)

    filter_shape = (feature_map, 1, window, dim)
    image_shape = (batch, 1, length + down + up, dim)

    #if non_linear=="none" or non_linear=="relu":
    #    conv_W = theano.shared(0.2 * numpy.random.uniform(low=-1.0,high=1.0,\
    #                            size=filter_shape).astype(theano.config.floatX))
        
    #else:
    #    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    #    conv_W = theano.shared(numpy.random.uniform(low=-W_bound,high=W_bound,\
    #                            size=filter_shape).astype(theano.config.floatX))

    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    conv_W = theano.shared(numpy.random.uniform(low=-W_bound,high=W_bound,\
                            size=filter_shape).astype(theano.config.floatX))

    conv_b = theano.shared(numpy.zeros(filter_shape[0], dtype=theano.config.floatX))

    # bundle
    params += [ conv_W, conv_b ]
    names += [ prefix + '_convL_W_' + str(window), prefix + '_convL_b_' + str(window) ]

    conv_out = conv.conv2d(input=inps, filters=conv_W, filter_shape=filter_shape, image_shape=image_shape)

    conv_out = T.tanh(conv_out + conv_b.dimshuffle('x', 0, 'x', 'x'))

    return conv_out.dimshuffle(0,2,1,3).flatten(3)
    
def convContextLs(inps, feature_map, convWins, batch, length, dim, prefix, params, names):
    cx = T.cast(inps.reshape((inps.shape[0], 1, inps.shape[1], inps.shape[2])), dtype=theano.config.floatX)

    fts = []
    for i, convWin in enumerate(convWins):
        fti = convolutionalLayer(cx, feature_map, batch, length, convWin, dim, prefix + '_winL' + str(i), params, names)
        fts += [fti]

    convRep = T.cast(T.concatenate(fts, axis=2), dtype=theano.config.floatX)

    return convRep
    
def LeNetConvPoolLayer(inps, feature_map, batch, length, window, dim, prefix, params, names):
    fan_in = window * dim
    fan_out = feature_map * window * dim / (length - window + 1)

    filter_shape = (feature_map, 1, window, dim)
    image_shape = (batch, 1, length, dim)
    pool_size = (length - window + 1, 1)

    #if non_linear=="none" or non_linear=="relu":
    #    conv_W = theano.shared(0.2 * numpy.random.uniform(low=-1.0,high=1.0,\
    #                            size=filter_shape).astype(theano.config.floatX))
        
    #else:
    #    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    #    conv_W = theano.shared(numpy.random.uniform(low=-W_bound,high=W_bound,\
    #                            size=filter_shape).astype(theano.config.floatX))

    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    conv_W = theano.shared(numpy.random.uniform(low=-W_bound,high=W_bound,\
                            size=filter_shape).astype(theano.config.floatX))

    conv_b = theano.shared(numpy.zeros(filter_shape[0], dtype=theano.config.floatX))

    # bundle
    params += [ conv_W, conv_b ]
    names += [ prefix + '_conv_W_' + str(window), prefix + '_conv_b_' + str(window) ]

    conv_out = conv.conv2d(input=inps, filters=conv_W, filter_shape=filter_shape, image_shape=image_shape)

        
    conv_out_act = T.tanh(conv_out + conv_b.dimshuffle('x', 0, 'x', 'x'))
    conv_output = downsample.max_pool_2d(input=conv_out_act, ds=pool_size, ignore_border=True)

    return conv_output.flatten(2)

def convContext(inps, feature_map, convWins, batch, length, dim, prefix, params, names):

    cx = T.cast(inps.reshape((inps.shape[0], 1, inps.shape[1], inps.shape[2])), dtype=theano.config.floatX)

    fts = []
    for i, convWin in enumerate(convWins):
        fti = LeNetConvPoolLayer(cx, feature_map, batch, length, convWin, dim, prefix + '_win' + str(i), params, names)
        fts += [fti]

    convRep = T.cast(T.concatenate(fts, axis=1), dtype=theano.config.floatX)

    return convRep
    
#############################Multilayer NNs################################

def HiddenLayer(inputs, nin, nout, params, names, prefix):
    W_bound = numpy.sqrt(6. / (nin + nout))
    multi_W = theano.shared(numpy.random.uniform(low=-W_bound,high=W_bound,\
                            size=(nin, nout)).astype(theano.config.floatX))

    multi_b = theano.shared(numpy.zeros(nout, dtype=theano.config.floatX))
    res = []
    for input in inputs:
        out = T.nnet.sigmoid(T.dot(input, multi_W) + multi_b)
        res += [out]
    
    params += [multi_W, multi_b]
    names += [prefix + '_multi_W', prefix + '_multi_b']
    
    return res

def MultiHiddenLayers(inputs, hids, params, names, prefix):
    
    hiddenVector = inputs
    id = 0
    for nin, nout in zip(hids, hids[1:]):
        id += 1
        hiddenVector = HiddenLayer(hiddenVector, nin, nout, params, names, prefix + '_layer' + str(id))
    return hiddenVector
    
def HiddenLayerGiven(inputs, multi_W, multi_b):

    res = []
    for input in inputs:
        out = T.nnet.sigmoid(T.dot(input, multi_W) + multi_b)
        res += [out]
    
    return res
    
def MultiHiddenLayersGiven(inputs, matrices):
    
    hiddenVector = inputs
    holder = [hiddenVector]
    for multi_W, multi_b in matrices:
        hiddenVector = HiddenLayerGiven(hiddenVector, multi_W, multi_b)
        holder += [hiddenVector]
    return holder
    
def generateMatrixForMNN(hids, prefix, params, names):
    res = []
    id = 0
    for nin, nout in zip(hids, hids[1:]):
        id += 1
        W_bound = numpy.sqrt(6. / (nin + nout))
        multi_W = theano.shared(numpy.random.uniform(low=-W_bound,high=W_bound,\
                            size=(nin, nout)).astype(theano.config.floatX))

        multi_b = theano.shared(numpy.zeros(nout, dtype=theano.config.floatX))
        if params != None: params += [multi_W, multi_b]
        if names != None: names += [prefix + '_layer' + str(id) + '_multi_W', prefix + '_layer' + str(id) + '_multi_b']
        res += [(multi_W, multi_b)]
    
    return res
        
#########################################################################################

class BaseModel(object):

    def __init__(self, args):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        #de :: dimension of the word embeddings
        cs :: word window context size
        '''
        self.container = {}
        
        self.args = args
        self.args['rng'] = numpy.random.RandomState(3435)
        self.args['dropoutTrigger'] = args['dropoutTrigger'] if args['dropoutTrigger'] > 0. else 0.
        self.args['dropoutArg'] = args['dropoutArg'] if args['dropoutArg'] > 0. else 0.
        
        # parameters of the model
        
        self.container['params'], self.container['names'] = [], []
        
        self.container['embDict'] = OrderedDict()
        self.container['vars'] = OrderedDict()
        self.container['dimIn'] = 0
        

        print '******************FEATURES******************'
        for ed in self.args['features']:
            if self.args['features'][ed] == 0:
                self.container['embDict'][ed] = theano.shared(self.args['embs'][ed].astype(theano.config.floatX))
                
                if self.args['updateEmbs']:
                    print '@@@@@@@ Will update embedding table: ', ed
                    self.container['params'] += [self.container['embDict'][ed]]
                    self.container['names'] += [ed]

            if self.args['features'][ed] == 0:
                self.container['vars'][ed] = T.imatrix()
                dimAdding = self.args['embs'][ed].shape[1]
                self.container['dimIn'] += dimAdding         
            elif self.args['features'][ed] == 1:
                self.container['vars'][ed] = T.tensor3()
                dimAdding = self.args['features_dim'][ed]
                self.container['dimIn'] += dimAdding

            if self.args['features'][ed] >= 0:
                print 'represetation - ', ed, ' : ', dimAdding 
                                
        print 'REPRESENTATION DIMENSION = ', self.container['dimIn']
        
        if self.args['distanceFet'] == 0:
            self.container['embDict']['dist1'] = theano.shared(self.args['embs']['dist1'].astype(theano.config.floatX))
            self.container['embDict']['dist2'] = theano.shared(self.args['embs']['dist2'].astype(theano.config.floatX))
            self.container['embDict']['dist3'] = theano.shared(self.args['embs']['dist3'].astype(theano.config.floatX))
            
            if self.args['updateEmbs']:
                print '@@@@@@@ Will update distance embedding tables'
                self.container['params'] += [self.container['embDict']['dist1'], self.container['embDict']['dist2'], self.container['embDict']['dist3']]
                self.container['names'] += ['dist1', 'dist2', 'dist3']
        
        if self.args['triggerGlob'] == 0:
            self.container['embDict']['trigger'] = theano.shared(self.args['embs']['trigger'].astype(theano.config.floatX))
            
            if self.args['updateEmbs']:
                print '@@@@@@@ Will update trigger embedding table'
                self.container['params'] += [ self.container['embDict']['trigger'] ]
                self.container['names'] += ['trigger']
        
        #self.container['sentLength'] = T.ivector('sentLength')
        
        self.container['triggerAnn'] = T.imatrix('triggerAnn')
        self.container['triggerMaskTrain'] = T.matrix('triggerMaskTrain')
        self.container['triggerMaskTest'] = T.imatrix('triggerMaskTest')
        self.container['triggerMaskTrainArg'] = T.matrix('triggerMaskTrainArg')
        self.container['triggerMaskTestArg'] = T.imatrix('triggerMaskTestArg')
        
        self.container['entities'] = T.imatrix('entities')
        
        self.container['argumentEntityIdAnn'] = T.itensor3('argumentEntityIdAnn')
        self.container['argumentPosAnn'] = T.itensor3('argumentPosAnn')
        self.container['argumentLabelAnn'] = T.itensor3('argumentLabelAnn')
        self.container['argumentMaskTrain'] = T.tensor3('argumentMaskTrain')
        
        self.container['possibleEnityIdByTrigger'] = T.itensor3('possibleEnityIdByTrigger')
        self.container['possibleEnityPosByTrigger'] = T.itensor3('possibleEnityPosByTrigger')
        self.container['argumentMaskTest'] = T.itensor3('argumentMaskTest')
        
        self.container['relDistBinary'] = T.tensor4('relDistBinary') #dimshuffle(1,0,2,3) first
        self.container['relDistIdxs'] = T.itensor3('relDistIdxs') #dimshuffle(1,0,2) first
        
        self.container['NodeFets'] = T.itensor3('NodeFets')
        self.container['EdgeFets'] = T.itensor4('EdgeFets')
        
        #self.container['numEntities'] = T.iscalar('numEntities')
        self.container['lr'] = T.scalar('lr')
        self.container['zeroVector'] = T.vector('zeroVector')
        
        self.glob = {}
        self.glob['batch'] = self.args['batch']
        self.glob['maxSentLength'] = self.args['maxSentLength']
        self.glob['numTrigger'] = self.args['numTrigger']
        self.glob['numArg'] = self.args['numArg']
        self.glob['maxNumEntities'] = self.args['maxNumEntities']
        self.glob['eachTrigger'] = theano.shared(numpy.zeros([self.glob['batch'], self.glob['maxSentLength'], self.glob['numTrigger']]).astype(theano.config.floatX))
        self.glob['eachArg'] = theano.shared(numpy.zeros([self.glob['batch'], self.glob['maxSentLength'], self.glob['numArg']]).astype(theano.config.floatX))
        self.glob['eachTriggerId'] = theano.shared(numpy.zeros([self.glob['batch'], self.glob['maxSentLength']]).astype('int32'))
        self.glob['eachArgId'] = theano.shared(numpy.zeros([self.glob['batch'], self.glob['maxSentLength']]).astype('int32'))
        
        self.glob['trigger'] = theano.shared(numpy.zeros([self.glob['batch'], self.glob['numTrigger']]).astype(theano.config.floatX))
        self.glob['arg'] = theano.shared(numpy.zeros([self.glob['batch'], self.glob['numArg']]).astype(theano.config.floatX))
        self.glob['argTrigger'] = theano.shared(numpy.zeros([self.glob['batch'], self.glob['maxNumEntities'], self.glob['numTrigger']]).astype(theano.config.floatX))
        self.glob['argArg'] = theano.shared(numpy.zeros([self.glob['batch'], self.glob['maxNumEntities'], self.glob['numArg']]).astype(theano.config.floatX))
        
        self.globZero = {}
        self.globZero['eachTrigger'] = numpy.zeros([self.glob['batch'], self.glob['maxSentLength'], self.glob['numTrigger']]).astype(theano.config.floatX)
        self.globZero['eachArg'] = numpy.zeros([self.glob['batch'], self.glob['maxSentLength'], self.glob['numArg']]).astype(theano.config.floatX)
        self.globZero['eachTriggerId'] = numpy.zeros([self.glob['batch'], self.glob['maxSentLength']]).astype('int32')
        self.globZero['eachArgId'] = numpy.zeros([self.glob['batch'], self.glob['maxSentLength']]).astype('int32')
        
        self.globZero['trigger'] = numpy.zeros([self.glob['batch'], self.glob['numTrigger']]).astype(theano.config.floatX)
        self.globZero['arg'] = numpy.zeros([self.glob['batch'], self.glob['numArg']]).astype(theano.config.floatX)
        self.globZero['argTrigger'] = numpy.zeros([self.glob['batch'], self.glob['maxNumEntities'], self.glob['numTrigger']]).astype(theano.config.floatX)
        self.globZero['argArg'] = numpy.zeros([self.glob['batch'], self.glob['maxNumEntities'], self.glob['numArg']]).astype(theano.config.floatX)
        
        self.globVar = {}
        self.globVar['eachTrigger'] = T.tensor3()
        self.globVar['eachArg'] = T.tensor3()
        self.globVar['eachTriggerId'] = T.imatrix()
        self.globVar['eachArgId'] = T.imatrix()
        
        self.globVar['trigger'] = T.matrix()
        self.globVar['arg'] = T.matrix()
        self.globVar['argTrigger'] = T.tensor3()
        self.globVar['argArg'] = T.tensor3()
        
        self.globFunc = {}
        
        self.container['setZero'] = OrderedDict()
        self.container['zeroVecs'] = OrderedDict()

    def buildFunctions(self, nll, y_pred, updateTrain=None, updateTest=None):
        
        gradients = T.grad( nll, self.container['params'] )

        classifyInput = [ self.container['vars'][ed] for ed in self.args['features'] if self.args['features'][ed] >= 0 ]
        self.classificationVariables = [ 'triggerMaskTest', 'triggerMaskTestArg', 'possibleEnityIdByTrigger', 'possibleEnityPosByTrigger', 'argumentMaskTest', 'relDistIdxs', 'NodeFets', 'EdgeFets' ] #'relDistBinary'
        classifyInput += [ self.container[van] for van in self.classificationVariables ]
        
        # theano functions
        if not updateTest:
            self.classify = theano.function(inputs=classifyInput, outputs=y_pred, on_unused_input='ignore')
        else:
            self.classify = theano.function(inputs=classifyInput, outputs=y_pred, on_unused_input='ignore', updates=updateTest)

        trainInput = [ self.container['vars'][ed] for ed in self.args['features'] if self.args['features'][ed] >= 0 ]
        self.trainVariables = [ 'triggerAnn', 'argumentEntityIdAnn', 'argumentPosAnn', 'argumentLabelAnn', 'relDistIdxs', 'triggerMaskTrain', 'triggerMaskTrainArg', 'argumentMaskTrain', 'NodeFets', 'EdgeFets' ] #'relDistBinary'
        trainInput += [ self.container[van] for van in self.trainVariables ]

        self.f_grad_shared, self.f_update_param = eval(self.args['optimizer'])(trainInput,nll,updateTrain,self.container['names'],self.container['params'],gradients,self.container['lr'],self.args['norm_lim'])
        
        for ed in self.container['embDict']:
            self.container['zeroVecs'][ed] = numpy.zeros(self.args['embs'][ed].shape[1],dtype='float32')
            self.container['setZero'][ed] = theano.function([self.container['zeroVector']], updates=[(self.container['embDict'][ed], T.set_subtensor(self.container['embDict'][ed][0,:], self.container['zeroVector']))])
            
        for ppr in self.globZero:
            self.globFunc[ppr] = theano.function([self.globVar[ppr]], updates=[( self.glob[ppr], T.set_subtensor(self.glob[ppr][:], self.globVar[ppr]) )])

    def resetGlobalVariables(self):
        for ppr in self.globZero:
            self.globFunc[ppr](self.globZero[ppr])
    
    def save(self, folder):   
        for param, name in zip(self.container['params'], self.container['names']):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())
            

class rnnJoint(BaseModel):
    def __init__(self, args):

        BaseModel.__init__(self, args)
        
        _rnnJoint(self)

def localWordEmbeddingsTrigger(model, pos, ewords, win, maxwin):
    idxs = ewords[:,(maxwin+pos-win):(maxwin+pos+win+1)]
    res = model.container['embDict']['word'][idxs.flatten()].reshape((idxs.shape[0], idxs.shape[1]*model.args['embs']['word'].shape[1]))
    return res
    
def localWordEmbeddingsArg(model, triggerPos, slice_aposs, ewords, win, maxwin):
    idxs = ewords[:,(maxwin+triggerPos-win):(maxwin+triggerPos+win+1)]
    triggerWed = model.container['embDict']['word'][idxs.flatten()].reshape((idxs.shape[0], idxs.shape[1]*model.args['embs']['word'].shape[1]))
    ones = T.ones((1, model.args['maxNumEntities'], 1), dtype='float32')
    triggerWed = ones * triggerWed[:, numpy.newaxis, :]
    
    addent = T.arange(2*win+1).reshape((1,1,2*win+1)) - win + maxwin
    aidxs = slice_aposs[:, :, numpy.newaxis] + addent
    aar = T.arange(aidxs.shape[0]).reshape((aidxs.shape[0], 1, 1))
    wids = ewords[aar, aidxs]
    argWed = model.container['embDict']['word'][wids.flatten()].reshape((wids.shape[0], wids.shape[1], wids.shape[2]*model.args['embs']['word'].shape[1]))
    return triggerWed, argWed

def appendDistInfo(model, ini, poss, embNames, relDistIdxs, relDistBinary=None):
    mats = [ini]
        
    for ename, pos in zip(embNames, poss):
        if model.args['distanceFet'] == 1:
            dmat = relDistBinary[pos].dimshuffle(1,0,2) ## be careful float64
        elif model.args['distanceFet'] == 0:   
            idxs = relDistIdxs[pos].T
            dmat = model.container['embDict'][ename][idxs.flatten()].reshape((idxs.shape[0], idxs.shape[1], model.args['distanceDim']))
        
        mats += [dmat]
    
    res = T.cast(T.concatenate(mats, axis=2), dtype=theano.config.floatX)
    return res
    
def appendDistInfo4(model, ini, slice_aposs, ename, relDistIdxs, relDistBinary=None):
    tens = [ini]
    
    id_slice_aposs = T.arange(slice_aposs.shape[0]).reshape((slice_aposs.shape[0], 1))
    
    if model.args['distanceFet'] == 1:
        ten = relDistBinary.dimshuffle(1,0,2,3)[id_slice_aposs, slice_aposs].dimshuffle(2,0,1,3)
    elif model.args['distanceFet'] == 0:
        idxs = relDistIdxs.dimshuffle(1,0,2)[id_slice_aposs, slice_aposs].dimshuffle(2,0,1)
        ten = model.container['embDict'][ename][idxs.flatten()].reshape((idxs.shape[0], idxs.shape[1], idxs.shape[2], model.args['distanceDim']))
    
    tens += [ten]
    
    res = T.cast(T.concatenate(tens, axis=3), dtype=theano.config.floatX)
    return res

def getTriggerLabelFeatures(model, tlabs):
    zvec = T.zeros((tlabs.shape[0], model.args['numTrigger']), dtype=theano.config.floatX)
    zvec = T.set_subtensor(zvec[T.arange(tlabs.shape[0]), tlabs], 1.0)
    ones = T.ones((1, model.args['maxNumEntities'], 1), dtype='float32')
    return ones * zvec[:, numpy.newaxis, :]

def getExternalArgTriggerRep(model, nodeFets, extMatTrigger):
    erep = extMatTrigger[nodeFets.flatten()].reshape((nodeFets.shape[0], nodeFets.shape[1], extMatTrigger.shape[1]))
    erep = erep.sum(axis=1)
    ones = T.ones((1, model.args['maxNumEntities'], 1), dtype='float32')
    return ones * erep[:, numpy.newaxis, :]

def getExternalArgArgRep(model, edgeFets, extMatArg):
    erep = extMatArg[edgeFets.flatten()].reshape((edgeFets.shape[0], edgeFets.shape[1], edgeFets.shape[2], extMatArg.shape[1]))
    erep = erep.sum(axis=2)
    return erep
    
def getExternalArgTriggerTypeRep(model, triggerTypes, extTTypeMat):
    erep = extTTypeMat[triggerTypes]
    ones = T.ones((1, model.args['maxNumEntities'], 1), dtype='float32')
    return ones * erep[:, numpy.newaxis, :]

def getExternalArgRep(model, nodeFets, edgeFets, triggerTypes, extFirstATW, extFirstAAW, extFirstATTypeW, extFirstAb, extAMats):
    extArgTrig = getExternalArgTriggerRep(model, nodeFets, extFirstATW)
    extArgArg = getExternalArgArgRep(model, edgeFets, extFirstAAW)
    extArgTType = getExternalArgTriggerTypeRep(model, triggerTypes, extFirstATTypeW)
    ext = extArgTrig + extArgArg + extArgTType + extFirstAb
    ext = T.nnet.sigmoid(ext)
    ext = MultiHiddenLayersGiven([ext], extAMats)[-1][0]
    return ext

def _rnnJoint(model):
  
    _x = eval(model.args['rep'])(model.container['embDict'], model.container['vars'], model.args['features'], model.args['features_dim'], model.container['dimIn'], model.args['nh'], model.args['batch'], 'gruJoint', model.container['params'], model.container['names'])
    
    if model.args['distanceFet'] >= 0: dimDist = model.args['distanceDim']
    else: dimDist = 0
    if model.args['triggerGlob'] >= 0: dimGlobTrigger = model.args['triggerDim']
    else: dimGlobTrigger = 0
    if model.args['argGlob'] >= 0: dimGlobArg = model.args['numArg']
    else: dimGlobArg = 0
    
    dim_to_joint = model.args['nh']
    if 'Bi' in model.args['rep']: dim_to_joint = 2 * model.args['nh']
    
    dimTrigger = dim_to_joint #+ model.glob['numTrigger'] + model.glob['numArg']
    #dimArg = dim_to_joint + model.args['numTrigger'] + (model.glob['numTrigger']-1) + (model.glob['numArg']-1)
    dimArg = dim_to_joint + (model.glob['numTrigger']-1) #+ (model.glob['numArg']-1)
    
    if dimDist == 0: dimArg += dim_to_joint
    
    if model.args['winTrigger'] >= 0: dimTrigger += (1+2*model.args['winTrigger']) * model.args['embs']['word'].shape[1]
    if model.args['winArg'] >= 0: dimArg += (2+4*model.args['winArg']) * model.args['embs']['word'].shape[1]
    
    hidTrigger = [dimTrigger] + model.args['multilayerTrigger']
    hidArg = [dimArg] + model.args['multilayerArg']
    
    matTrigger = generateMatrixForMNN(hidTrigger, 'trigger', model.container['params'], model.container['names'])
    matArg = generateMatrixForMNN(hidArg, 'arg', model.container['params'], model.container['names'])
    
    matTemp = generateMatrixForMNN([ hidTrigger[-1], model.args['numTrigger'] ], 'triggerFinal', model.container['params'], model.container['names'])
    trigger_Wf, trigger_bf = matTemp[0]
    
    hidPossibleExtArg = hidArg[-1]
    if len(model.args['multilayerArgExternal']) > 0: hidPossibleExtArg += model.args['multilayerArgExternal'][-1]
    matTemp = generateMatrixForMNN([ hidPossibleExtArg, model.args['numArg'] ], 'argFinal', model.container['params'], model.container['names'])
    arg_Wf, arg_bf = matTemp[0]
    
    if dimDist > 0:
        hidAttTrigger = [dim_to_joint + dimDist] + model.args['multilayerTriggerAtt']
        matAttTrigger = generateMatrixForMNN(hidAttTrigger, 'attentionTrigger', model.container['params'], model.container['names'])
        
        matTemp = generateMatrixForMNN([ hidAttTrigger[-1], 1 ], 'triggerAttFinal', model.container['params'], model.container['names'])
        trigger_AWf, trigger_Abf = matTemp[0]
        
        hidAttArg = [dim_to_joint + 2*dimDist] + model.args['multilayerArgAtt']
        matAttArg = generateMatrixForMNN(hidAttArg, 'attentionArg', model.container['params'], model.container['names'])
        
        matTemp = generateMatrixForMNN([ hidAttArg[-1], 1 ], 'argAttFinal', model.container['params'], model.container['names'])
        arg_AWf, arg_Abf = matTemp[0]
        
    maxWinTA = model.args['winTrigger'] if model.args['winTrigger'] > model.args['winArg'] else model.args['winArg']
    extendedWords = model.container['vars']['word']
    if maxWinTA > 0:
        wleft = T.zeros((extendedWords.shape[0], maxWinTA), dtype='int32')
        wright = T.zeros((extendedWords.shape[0], maxWinTA), dtype='int32')
        extendedWords = T.cast(T.concatenate([wleft, extendedWords, wright], axis=1), dtype='int32')
        
    if len(model.args['multilayerArgExternal']) > 0:
        extFirstATW, extFirstAb = generateMatrixForMNN([ model.args['nodeFetDim'], model.args['multilayerArgExternal'][0] ], 'extFirstAT', model.container['params'], model.container['names'])[0]
        
        model.container['zeroVecs'][model.container['names'][-2]] = numpy.zeros(model.args['multilayerArgExternal'][0], dtype='float32')
        model.container['setZero'][model.container['names'][-2]] = theano.function([model.container['zeroVector']], updates=[(model.container['params'][-2], T.set_subtensor(model.container['params'][-2][0,:], model.container['zeroVector']))])
        
        extFirstAAW, _ = generateMatrixForMNN([ model.args['edgeFetDim'], model.args['multilayerArgExternal'][0] ], 'extFirstAA', model.container['params'], model.container['names'])[0]
        
        model.container['zeroVecs'][model.container['names'][-2]] = numpy.zeros(model.args['multilayerArgExternal'][0], dtype='float32')
        model.container['setZero'][model.container['names'][-2]] = theano.function([model.container['zeroVector']], updates=[(model.container['params'][-2], T.set_subtensor(model.container['params'][-2][0,:], model.container['zeroVector']))])
        
        model.container['params'], model.container['names'] = model.container['params'][0:-1], model.container['names'][0:-1]
        extFirstATTypeW, _ = generateMatrixForMNN([ model.args['numTrigger'], model.args['multilayerArgExternal'][0] ], 'extFirstATType', model.container['params'], model.container['names'])[0]
        
        model.container['zeroVecs'][model.container['names'][-2]] = numpy.zeros(model.args['multilayerArgExternal'][0], dtype='float32')
        model.container['setZero'][model.container['names'][-2]] = theano.function([model.container['zeroVector']], updates=[(model.container['params'][-2], T.set_subtensor(model.container['params'][-2][0,:], model.container['zeroVector']))])
        
        model.container['params'], model.container['names'] = model.container['params'][0:-1], model.container['names'][0:-1]
        extAMats = generateMatrixForMNN(model.args['multilayerArgExternal'][1:], 'extAT', model.container['params'], model.container['names'])
    
    #sequence at _alabels
    def recurTrain(idx, _tlabels, _entIds, _aposs, _alabels, _nodeFets, _edgeFets, _rep, _relDistIdxs, _extendedWords, _gts, _gas, _gats, _gaas): #_relDistBinary
        if dimDist == 0:
            c_x = _rep[idx]
        else:
            inter = appendDistInfo(model, _rep, [idx], ['dist1'], _relDistIdxs) #_relDistBinary
            inter = MultiHiddenLayersGiven([inter], matAttTrigger)[-1][0]
            
            inter = T.dot(inter, trigger_AWf) + trigger_Abf
            inter = inter[:,:,0]
            inter = T.exp(inter - inter.max(axis=0, keepdims=True))
            inter = inter / inter.sum(axis=0, keepdims=True)
            
            inter = _rep * inter.dimshuffle(0, 1, 'x')
            c_x = inter.sum(axis=0)
        
        _tvecs = [c_x]#, _gts, _gas]
        if model.args['winTrigger'] >= 0:
            lc = localWordEmbeddingsTrigger(model, idx, _extendedWords, model.args['winTrigger'], maxWinTA)
            _tvecs += [lc]
        if len(_tvecs) > 1: tvec = T.cast(T.concatenate(_tvecs, axis=1), dtype=theano.config.floatX)
        else: tvec = _tvecs[0]
        
        tvec = MultiHiddenLayersGiven([tvec], matTrigger)[-1][0]
        
        id_aposs = T.arange(_aposs.shape[0]).reshape((_aposs.shape[0], 1))
        if dimDist == 0:
            t_rep = _rep[idx]
            ones = T.ones((1, model.args['maxNumEntities'], 1), dtype='float32')
            t_rep = ones * t_rep[:, numpy.newaxis, :]

            a_rep = _rep.dimshuffle(1,0,2)[id_aposs, _aposs]
            
            ca_x = T.cast(T.concatenate([t_rep, a_rep], axis=2), dtype=theano.config.floatX)
        else:
            inter = appendDistInfo(model, _rep, [idx], ['dist2'], _relDistIdxs) #_relDistBinary
            ones = T.ones((1, 1, model.args['maxNumEntities'], 1), dtype='float32')
            inter = ones * inter[:, :, numpy.newaxis, :]
            
            c_rep = ones * _rep[:, :, numpy.newaxis, :]
            
            inter = appendDistInfo4(model, inter, _aposs, 'dist3', _relDistIdxs) #_relDistBinary
            inter = MultiHiddenLayersGiven([inter], matAttArg)[-1][0]
            inter = T.dot(inter, arg_AWf) + arg_Abf
            inter = inter[:,:,:,0]
            inter = T.exp(inter - inter.max(axis=0, keepdims=True))
            inter = inter / inter.sum(axis=0, keepdims=True)
            
            inter = c_rep * inter.dimshuffle(0, 1, 2, 'x')
            ca_x = inter.sum(axis=0)
        
        ttv = getTriggerLabelFeatures(model, _tlabels)
        m_gats = _gats[id_aposs, _entIds][:,:,1:]
        m_gaas = _gaas[id_aposs, _entIds][:,:,1:]
        #_avecs = [ca_x, ttv, m_gats, m_gaas]
        _avecs = [ca_x, m_gats]#, m_gaas]
        if model.args['winArg'] >= 0:
            tlc, alc = localWordEmbeddingsArg(model, idx, _aposs, _extendedWords, model.args['winArg'], maxWinTA)
            _avecs += [tlc, alc]
        
        avec = T.cast(T.concatenate(_avecs, axis=2), dtype=theano.config.floatX)
        
        avec = MultiHiddenLayersGiven([avec], matArg)[-1][0]
        
        if len(model.args['multilayerArgExternal']) > 0:
            ext_arg = getExternalArgRep(model, _nodeFets, _edgeFets, _tlabels, extFirstATW, extFirstAAW, extFirstATTypeW, extFirstAb, extAMats)
        else: ext_arg = T.zeros((1,), dtype=theano.config.floatX)
        
        #masa = T.ones(avec.shape, dtype=theano.config.floatX)
        
        chosen = T.cast(_tlabels > 0, dtype=theano.config.floatX)
        #chosen = chosen.reshape((chosen.shape[0], 1, 1))
        
        #masa = chosen * masa
        
        #_gts = T.set_subtensor(_gts[T.arange(model.glob['batch']), _tlabels], 1.0)
        #_gaas = T.set_subtensor(_gaas[id_aposs, _entIds, _alabels], 1.0)
        
        #_gas = T.set_subtensor(_gas[id_aposs, _alabels], 1.0)
        
        gones = T.ones((1, model.glob['maxNumEntities']), dtype='int32')
        g_tlabels = gones * _tlabels[:, numpy.newaxis]
        g_tlabels = g_tlabels * T.cast(_entIds >= 0, dtype='int32')
        #_gats = T.set_subtensor(_gats[id_aposs, _entIds, g_tlabels], 1.0)
        
        return ([tvec, avec, ext_arg, chosen], {_gts : T.set_subtensor(_gts[T.arange(model.glob['batch']), _tlabels], 1.0), _gas : T.set_subtensor(_gas[id_aposs, _alabels], 1.0), _gats : T.set_subtensor(_gats[id_aposs, _entIds, g_tlabels], 1.0), _gaas : T.set_subtensor(_gaas[id_aposs, _entIds, _alabels], 1.0)})
    
    outscanTrain, updateTrain = theano.scan(fn=recurTrain, sequences=[T.arange(model.args['maxSentLength']), model.container['triggerAnn'].T, model.container['argumentEntityIdAnn'].dimshuffle(1,0,2), model.container['argumentPosAnn'].dimshuffle(1,0,2), model.container['argumentLabelAnn'].dimshuffle(1,0,2), model.container['NodeFets'].dimshuffle(1,0,2), model.container['EdgeFets'].dimshuffle(1,0,2,3)], non_sequences = [_x, model.container['relDistIdxs'].dimshuffle(1,0,2), extendedWords, model.glob['trigger'], model.glob['arg'], model.glob['argTrigger'], model.glob['argArg']], outputs_info=[None, None, None, None], n_steps=model.args['maxSentLength']) #model.container['relDistBinary'].dimshuffle(1,0,2,3)
    
    randTrigger = generateBinomial((model.args['maxSentLength'], model.args['batch'], hidTrigger[-1]), model.args['rng'], model.args['dropoutTrigger'])
    randArg = generateBinomial((model.args['maxSentLength'], model.args['batch'], model.args['maxNumEntities'], hidArg[-1]), model.args['rng'], model.args['dropoutArg'])
    
    score_trigger = outscanTrain[0] * randTrigger
    score_arg = outscanTrain[1] * randArg
    score_ext_arg = outscanTrain[2]
    out_arg_mask = outscanTrain[3].T
    
    if len(model.args['multilayerArgExternal']) > 0:
        score_arg = T.cast(T.concatenate([score_arg, score_ext_arg], axis=3), dtype=theano.config.floatX)
    
    score_trigger = T.dot(score_trigger, trigger_Wf) + trigger_bf
    score_trigger = T.exp(score_trigger - score_trigger.max(axis=2, keepdims=True))
    score_trigger = T.log(score_trigger / score_trigger.sum(axis=2, keepdims=True))
    score_trigger = score_trigger.dimshuffle(1,0,2)
    
    score_arg = T.dot(score_arg, arg_Wf) + arg_bf
    score_arg = T.exp(score_arg - score_arg.max(axis=3, keepdims=True))
    score_arg = T.log(score_arg / score_arg.sum(axis=3, keepdims=True))
    score_arg = score_arg.dimshuffle(1,0,2,3)
    
    #out_arg_mask = out_arg_mask.dimshuffle(1,0,2,3)
    
    score_trigger = score_trigger * model.container['triggerMaskTrain'].dimshuffle(0,1,'x')
    score_trigger = score_trigger.reshape((score_trigger.shape[0]*score_trigger.shape[1], score_trigger.shape[2]))
    label_trigger = model.container['triggerAnn'].flatten()
    
    nll = -T.sum(score_trigger[T.arange(score_trigger.shape[0]), label_trigger])
    
    score_arg = score_arg * model.container['triggerMaskTrainArg'].dimshuffle(0,1,'x','x') * model.container['argumentMaskTrain'].dimshuffle(0,1,2,'x') * out_arg_mask.dimshuffle(0,1,'x','x')
    score_arg = score_arg.reshape((score_arg.shape[0]*score_arg.shape[1]*score_arg.shape[2], score_arg.shape[3]))
    label_arg = model.container['argumentLabelAnn'].flatten()
    
    nll -= T.sum(score_arg[T.arange(score_arg.shape[0]), label_arg])
    nll /= model.args['batch']
    
    def recurTest(idx, _tmasks, _tmasksArg, _nodeFets, _edgeFets, _rep, p_entIds, p_aposs, p_amasks, _relDistIdxs, _extendedWords, _gts, _gas, _gats, _gaas): #_relDistBinary
        if dimDist == 0:
            c_x = _rep[idx]
        else:
            inter = appendDistInfo(model, _rep, [idx], ['dist1'], _relDistIdxs) #_relDistBinary
            inter = MultiHiddenLayersGiven([inter], matAttTrigger)[-1][0]
            
            inter = T.dot(inter, trigger_AWf) + trigger_Abf
            inter = inter[:,:,0]
            inter = T.exp(inter - inter.max(axis=0, keepdims=True))
            inter = inter / inter.sum(axis=0, keepdims=True)
            
            inter = _rep * inter.dimshuffle(0, 1, 'x')
            c_x = inter.sum(axis=0)
        
        _tvecs = [c_x]#, _gts, _gas]
        if model.args['winTrigger'] >= 0:
            lc = localWordEmbeddingsTrigger(model, idx, _extendedWords, model.args['winTrigger'], maxWinTA)
            _tvecs += [lc]
        if len(_tvecs) > 1: tvec = T.cast(T.concatenate(_tvecs, axis=1), dtype=theano.config.floatX)
        else: tvec = _tvecs[0]
        
        tvec = MultiHiddenLayersGiven([tvec], matTrigger)[-1][0]
        
        s_trigger = T.dot(tvec, (1.0 - model.args['dropoutTrigger']) * trigger_Wf) + trigger_bf
        #s_trigger = T.exp(s_trigger - s_trigger.max(axis=1, keepdims=True))
        #s_trigger = s_trigger / s_trigger.sum(axis=1, keepdims=True)
        
        _tlabels = T.argmax(s_trigger, axis=1) * _tmasks
        
        _aposs = p_aposs[T.arange(model.args['batch']), _tlabels]
        _entIds = p_entIds[T.arange(model.args['batch']), _tlabels]
        _amasks = p_amasks[T.arange(model.args['batch']), _tlabels]
        
        id_aposs = T.arange(_aposs.shape[0]).reshape((_aposs.shape[0], 1))
        if dimDist == 0:
            t_rep = _rep[idx]
            ones = T.ones((1, model.args['maxNumEntities'], 1), dtype='float32')
            t_rep = ones * t_rep[:, numpy.newaxis, :]

            a_rep = _rep.dimshuffle(1,0,2)[id_aposs, _aposs]
            
            ca_x = T.cast(T.concatenate([t_rep, a_rep], axis=2), dtype=theano.config.floatX)
        else:
            inter = appendDistInfo(model, _rep, [idx], ['dist2'], _relDistIdxs) #_relDistBinary
            ones = T.ones((1, 1, model.args['maxNumEntities'], 1), dtype='float32')
            inter = ones * inter[:, :, numpy.newaxis, :]
            
            c_rep = ones * _rep[:, :, numpy.newaxis, :]
            
            inter = appendDistInfo4(model, inter, _aposs, 'dist3', _relDistIdxs) #_relDistBinary
            inter = MultiHiddenLayersGiven([inter], matAttArg)[-1][0]
            inter = T.dot(inter, arg_AWf) + arg_Abf
            inter = inter[:,:,:,0]
            inter = T.exp(inter - inter.max(axis=0, keepdims=True))
            inter = inter / inter.sum(axis=0, keepdims=True)
            
            inter = c_rep * inter.dimshuffle(0, 1, 2, 'x')
            ca_x = inter.sum(axis=0)
        
        ttv = getTriggerLabelFeatures(model, _tlabels)
        m_gats = _gats[id_aposs, _entIds][:,:,1:]
        m_gaas = _gaas[id_aposs, _entIds][:,:,1:]
        #_avecs = [ca_x, ttv, m_gats, m_gaas]
        _avecs = [ca_x, m_gats]#, m_gaas]
        if model.args['winArg'] >= 0:
            tlc, alc = localWordEmbeddingsArg(model, idx, _aposs, _extendedWords, model.args['winArg'], maxWinTA)
            _avecs += [tlc, alc]
        
        avec = T.cast(T.concatenate(_avecs, axis=2), dtype=theano.config.floatX)
        
        avec = MultiHiddenLayersGiven([avec], matArg)[-1][0]
        
        if len(model.args['multilayerArgExternal']) > 0:
            ext_arg = getExternalArgRep(model, _nodeFets, _edgeFets, _tlabels, extFirstATW, extFirstAAW, extFirstATTypeW, extFirstAb, extAMats)
            avec = T.cast(T.concatenate([avec, ext_arg], axis=2), dtype=theano.config.floatX)
        
        s_arg = T.dot(avec, (1.0 - model.args['dropoutArg']) * arg_Wf) + arg_bf
        #s_arg = T.exp(s_arg - s_arg.max(axis=2, keepdims=True))
        #s_arg = s_arg / s_arg.sum(axis=2, keepdims=True)
        
        _alabels = T.argmax(s_arg, axis=2) * _amasks * _tmasksArg.reshape((_tmasksArg.shape[0], 1))
        
        #_gts = T.set_subtensor(_gts[T.arange(model.glob['batch']), _tlabels], 1.0)
        #_gaas = T.set_subtensor(_gaas[id_aposs, _entIds, _alabels], 1.0)
        
        #_gas = T.set_subtensor(_gas[id_aposs, _alabels], 1.0)
        
        gones = T.ones((1, model.glob['maxNumEntities']), dtype='int32')
        g_tlabels = gones * _tlabels[:, numpy.newaxis]
        g_tlabels = g_tlabels * T.cast(_entIds >= 0, dtype='int32')
        #_gats = T.set_subtensor(_gats[id_aposs, _entIds, g_tlabels], 1.0)
        
        return ([_tlabels, _entIds, _alabels], {_gts : T.set_subtensor(_gts[T.arange(model.glob['batch']), _tlabels], 1.0), _gas : T.set_subtensor(_gas[id_aposs, _alabels], 1.0), _gats : T.set_subtensor(_gats[id_aposs, _entIds, g_tlabels], 1.0), _gaas : T.set_subtensor(_gaas[id_aposs, _entIds, _alabels], 1.0)})
    
    outscanTest, updateTest = theano.scan(fn=recurTest, sequences=[T.arange(model.args['maxSentLength']), model.container['triggerMaskTest'].T, model.container['triggerMaskTestArg'].T, model.container['NodeFets'].dimshuffle(1,0,2), model.container['EdgeFets'].dimshuffle(1,0,2,3)], non_sequences = [_x, model.container['possibleEnityIdByTrigger'], model.container['possibleEnityPosByTrigger'], model.container['argumentMaskTest'], model.container['relDistIdxs'].dimshuffle(1,0,2), extendedWords, model.glob['trigger'], model.glob['arg'], model.glob['argTrigger'], model.glob['argArg']], outputs_info=[None, None, None], n_steps=model.args['maxSentLength']) #model.container['relDistBinary'].dimshuffle(1,0,2,3)
        
    overal_prediction = [outscanTest[0].T, outscanTest[1].dimshuffle(1,0,2), outscanTest[2].dimshuffle(1,0,2)]

    model.buildFunctions(nll, overal_prediction, updateTrain, updateTest)
    
###############
