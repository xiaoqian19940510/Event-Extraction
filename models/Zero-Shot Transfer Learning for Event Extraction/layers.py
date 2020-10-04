import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import theano.sandbox.neighbours as TSN


class LogisticRegressionMulti(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W = None, b = None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        if W == None:
          # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
          self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='softmax_W', borrow=True)
        else:
          self.W = W

        if b == None:
          # initialize the baises b as a vector of n_out 0s
          self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='softmax_b', borrow=True)
        else:
          self.b = b



        # compute vector of class-membership probabilities in symbolic form
        #self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b + 1e-7)
        self.p_y_given_x = T.nnet.softmax(input.dot(self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def results(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        #self.all_p_y_given_x = self.in2.dot(self.W) + self.b

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return [T.argmax(self.p_y_given_x, axis=1), T.max(self.p_y_given_x, axis=1), self.p_y_given_x]
            #return self.p_y_given_x
        else:
            raise NotImplementedError()


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh, name=""):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if name != "":
          prefix = name
        else:
          prefix = "mlp_"
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name=prefix+'W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name=prefix+'b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]


class LeNetConvLayer(object):
    """Pool Layer of a convolutional network """
    def kmaxPooling(self, conv_out, k):
      neighborsForPooling = TSN.images2neibs(ten4=conv_out, neib_shape=(1,conv_out.shape[3]), mode='ignore_borders')
      self.neighbors = neighborsForPooling

      neighborsArgSorted = T.argsort(neighborsForPooling, axis=1)
      self.neighborsArgSorted = neighborsArgSorted
      kNeighborsArg = neighborsArgSorted[:,-k:]
      self.neigborsSorted = kNeighborsArg
      kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1)
      ii = T.repeat(T.arange(neighborsForPooling.shape[0]), k)
      jj = kNeighborsArgSorted.flatten()
      self.ii = ii
      self.jj = jj
      pooledkmaxTmp = neighborsForPooling[ii, jj]

      self.pooled = pooledkmaxTmp

      # reshape pooled_out
      new_shape = T.cast(T.join(0, conv_out.shape[:-2],
                         T.as_tensor([conv_out.shape[2]]),
                         T.as_tensor([k])),
                         'int64')
      pooledkmax = T.reshape(pooledkmaxTmp, new_shape, ndim=4)
      return pooledkmax

    def convStep(self, curInput, curFilter):
      return conv.conv2d(input=curInput, filters=curFilter,
                filter_shape=self.filter_shape,
                image_shape=None)

    def __init__(self, rng, W, b, input, filter_shape, image_shape):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type W: theano.matrix
        :param W: the weight matrix used for convolution

        :type b: theano vector
        :param b: the bias used for convolution

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        self.W = W
        self.b = b
        self.filter_shape = filter_shape

        # convolve input feature maps with filters
        conv_out = self.convStep(self.input, self.W)

        conv_with_bias = T.tanh(conv_out+self.b.dimshuffle('x', 0, 'x', 'x'))

        self.output = conv_with_bias

        # k = 3

        # self.pooledkmax = self.kmaxPooling(conv_out, k)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        #self.output = T.tanh(self.pooledkmax + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


class LeNetConvLayerWithStrides(object):

    def convStep(self, curInput, curFilter):
      return conv.conv2d(input=curInput, filters=curFilter,
                filter_shape=self.filter_shape, subsample=(2, 2),
                image_shape=None)

    def __init__(self, rng, W, b, input, filter_shape, image_shape):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type W: theano.matrix
        :param W: the weight matrix used for convolution

        :type b: theano vector
        :param b: the bias used for convolution

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        self.W = W
        self.b = b
        self.filter_shape = filter_shape

        # convolve input feature maps with filters
        conv_out = self.convStep(self.input, self.W)

        conv_with_bias = T.tanh(conv_out+self.b.dimshuffle('x', 0, 'x', 'x'))

        self.output = conv_with_bias

        #k = poolsize[1]

        #self.pooledkmax = self.kmaxPooling(conv_out, k)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        #self.output = T.tanh(self.pooledkmax + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def kmaxPooling(self, conv_out, k):
      neighborsForPooling = TSN.images2neibs(ten4=conv_out, neib_shape=(1,conv_out.shape[3]), mode='ignore_borders')
      self.neighbors = neighborsForPooling

      neighborsArgSorted = T.argsort(neighborsForPooling, axis=1)
      self.neighborsArgSorted = neighborsArgSorted
      kNeighborsArg = neighborsArgSorted[:,-k:]
      self.neigborsSorted = kNeighborsArg
      kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1)
      ii = T.repeat(T.arange(neighborsForPooling.shape[0]), k)
      jj = kNeighborsArgSorted.flatten()
      self.ii = ii
      self.jj = jj
      pooledkmaxTmp = neighborsForPooling[ii, jj]

      self.pooled = pooledkmaxTmp

      # reshape pooled_out
      new_shape = T.cast(T.join(0, conv_out.shape[:-2],
                         T.as_tensor([conv_out.shape[2]]),
                         T.as_tensor([k])),
                         'int64')
      pooledkmax = T.reshape(pooledkmaxTmp, new_shape, ndim=4)
      return pooledkmax      

    def convStep(self, curInput, curFilter):
      return conv.conv2d(input=curInput, filters=curFilter,
                filter_shape=self.filter_shape,
                image_shape=None)

    def __init__(self, rng, W, b, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type W: theano.matrix
        :param W: the weight matrix used for convolution

        :type b: theano vector
        :param b: the bias used for convolution

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        self.W = W
        self.b = b
        self.filter_shape = filter_shape

        # convolve input feature maps with filters
        conv_out = self.convStep(self.input, self.W)

        k = poolsize[1]
        self.pooledkmax = self.kmaxPooling(conv_out, k)
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(self.pooledkmax + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


class LeNetConvPoolLayerWithStrides(object):
    """Pool Layer of a convolutional network """

    def kmaxPooling(self, conv_out, k):
      neighborsForPooling = TSN.images2neibs(ten4=conv_out, neib_shape=(1,conv_out.shape[3]), mode='ignore_borders')
      self.neighbors = neighborsForPooling

      neighborsArgSorted = T.argsort(neighborsForPooling, axis=1)
      self.neighborsArgSorted = neighborsArgSorted
      kNeighborsArg = neighborsArgSorted[:,-k:]
      self.neigborsSorted = kNeighborsArg
      kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1)
      ii = T.repeat(T.arange(neighborsForPooling.shape[0]), k)
      jj = kNeighborsArgSorted.flatten()
      self.ii = ii
      self.jj = jj
      pooledkmaxTmp = neighborsForPooling[ii, jj]

      self.pooled = pooledkmaxTmp

      # reshape pooled_out
      new_shape = T.cast(T.join(0, conv_out.shape[:-2],
                         T.as_tensor([conv_out.shape[2]]),
                         T.as_tensor([k])),
                         'int64')
      pooledkmax = T.reshape(pooledkmaxTmp, new_shape, ndim=4)
      return pooledkmax

    def convStep(self, curInput, curFilter):
      return conv.conv2d(input=curInput, filters=curFilter,
                filter_shape=self.filter_shape, subsample=(2, 2),
                image_shape=None)

    def __init__(self, rng, W, b, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type W: theano.matrix
        :param W: the weight matrix used for convolution

        :type b: theano vector
        :param b: the bias used for convolution

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        self.W = W
        self.b = b
        self.filter_shape = filter_shape

        # convolve input feature maps with filters
        self.conv_out = self.convStep(self.input, self.W)

        k = poolsize[1]
        self.pooledkmax = self.kmaxPooling(self.conv_out, k)
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(self.pooledkmax + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


class KMax_Pooling(object):
    def __init__(self, conv_out, k):
        self.conv_out = conv_out
        self.k = k;
        self.output = self.kmaxPooling(conv_out, k)

    def kmaxPooling(self, conv_out, k):
      neighborsForPooling = TSN.images2neibs(ten4=conv_out, neib_shape=(1,conv_out.shape[3]), mode='ignore_borders')
      self.neighbors = neighborsForPooling

      neighborsArgSorted = T.argsort(neighborsForPooling, axis=1)
      self.neighborsArgSorted = neighborsArgSorted
      kNeighborsArg = neighborsArgSorted[:,-k:]
      self.neigborsSorted = kNeighborsArg
      kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1)
      ii = T.repeat(T.arange(neighborsForPooling.shape[0]), k)
      jj = kNeighborsArgSorted.flatten()
      self.ii = ii
      self.jj = jj
      pooledkmaxTmp = neighborsForPooling[ii, jj]

      self.pooled = pooledkmaxTmp

      # reshape pooled_out
      new_shape = T.cast(T.join(0, conv_out.shape[:-2],
                         T.as_tensor([conv_out.shape[2]]),
                         T.as_tensor([k])),
                         'int64')
      pooledkmax = T.reshape(pooledkmaxTmp, new_shape, ndim=4)
      return pooledkmax


class Max_Pooling(object):
    """The input is output of Conv: a tensor.  The output here should also be tensor"""
    def __init__(self, rng, input):  # length_l, length_r: valid lengths after conv
        #input_l_matrix = input.reshape((input.shape[2], input.shape[3]))
        #input_l_matrix = debug_print(input_l_matrix[:, left_l:(input_l_matrix.shape[1] - right_l)],
                                         #'input_l_matrix')
        self.output_maxpooling = T.max(input, axis=3)


class ComposeLayerMatrix(object):
    def __init__(self, input, input_binary_relation, rel_w, rel_vec_size):

        self.input = input
        self.input_binary_relation = input_binary_relation
        self.rel_w = rel_w

        input_binary_relation_1 = T.transpose(input_binary_relation, [0,1,3,2]) # 100*1*5*26
        input_binary_relation_2 = input_binary_relation_1.reshape(
            (input_binary_relation_1.shape[0]*input_binary_relation_1.shape[1]*input_binary_relation_1.shape[2],
             input_binary_relation_1.shape[3])) #500*26
        r = input_binary_relation_2.dot(rel_w) # 500 * 160000

        input_1 = T.transpose(input, [0, 1, 3, 2]) # 100*1*5*400
        input_2 = input_1.reshape((input_1.shape[0]*input_1.shape[1]*input_1.shape[2], input_1.shape[3])) # 500*400
        input_3 = input_2.reshape((input_2.shape[0], input_2.shape[1], 1)) # 500*400*1
        input_4 = T.repeat(input_3, rel_vec_size, axis=2) # 500*400*400
        input_5 = T.transpose(input_4, [0,2,1]) # 500*400*400
        input_6 = input_5.reshape((input_5.shape[0], input_5.shape[1]*input_5.shape[2]))  # 500*160000

        r1 = input_6*r # 500*160000
        r2 = r1.reshape((r1.shape[0], rel_vec_size, rel_vec_size))
        r3 = T.mean(r2, axis=2)
        r4 = r3.reshape((input_1.shape[0], input_1.shape[1], input_1.shape[2], r3.shape[1]))
        input_update = T.transpose(r4, [0,1,3,2])
        self.output = input_update
        # parameters of the model
        self.params = [self.rel_w]


class ComposeLayerFactor(object):
    def __init__(self, input, input_binary_relation, rel_w, rel_vec_size):


        self.input = input
        self.input_binary_relation = input_binary_relation
        self.rel_w = rel_w  # relationsize * 400


        input_binary_relation_1 = T.transpose(input_binary_relation, [0,1,3,2]) # 100*1*5*26
        input_binary_relation_1 = input_binary_relation_1.reshape(
            (input_binary_relation_1.shape[0]*input_binary_relation_1.shape[1]*input_binary_relation_1.shape[2],
             input_binary_relation_1.shape[3])) # 500*26

        r = input_binary_relation_1.dot(rel_w) # 500 * 400

        input_1 = T.transpose(input, [0, 1, 3, 2]) # 100*1*5*400
        input_2 = input_1.reshape((input_1.shape[0]*input_1.shape[1]*input_1.shape[2], input_1.shape[3])) # 500*400

        r1 = input_2*r # 500*400
        r4 = r1.reshape((input_1.shape[0], input_1.shape[1], input_1.shape[2], r1.shape[1]))
        input_update = T.transpose(r4, [0,1,3,2])

        self.output = input_update

        self.params = [self.rel_w]


class ComposeLayerTensor(object):
    def __init__(self, input, tensor):
        self.input = input
        self.tensor = tensor  # 400*400*400

        r = self.tensor.reshape((self.tensor.shape[0]*self.tensor.shape[1], self.tensor.shape[2]))

        input_1 = T.transpose(input, [0, 1, 3, 2]) # 100*1*5*400
        input_2 = input_1.reshape((input_1.shape[0]*input_1.shape[1]*input_1.shape[2], input_1.shape[3])) # 500*400

        input_3 = input_2.reshape((input_2.shape[0], input_2.shape[1], 1))
        input_4 = input_2.reshape((input_2.shape[0], 1, input_2.shape[1]))
        input_dot = T.batched_dot(input_3, input_4)
        input_dot = input_dot.reshape((input_dot.shape[0], input_dot.shape[1]*input_dot.shape[2]))

        r1 = T.dot(input_dot, r) # 500*400
        r4 = r1.reshape((input_1.shape[0], input_1.shape[1], input_1.shape[2], r1.shape[1]))
        input_update = T.transpose(r4, [0,1,3,2])

        self.output = input_update
        self.params = [self.tensor]


class MaxRankingMarginCosine1(object):
    def __init__(self, rng, input, input_label, true_label, n_in, margin, batch_size, type_size, train_type_signal, pos_neg_flag):

        self.input = input
        self.input_label = input_label
        self.true_label = true_label

        W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_in)),
                    high=numpy.sqrt(6. / (n_in + n_in)),
                    size=(n_in, n_in)), dtype=theano.config.floatX)

        W = theano.shared(value=W_values, borrow=True)

        self.W = W

        self.margin = margin

        self.params = [self.W]

        sim = self.cosine(input, input_label)

        true_sim = sim*true_label
        neg_sim = sim  ## batchsize*labelsize

        self.true_sim = true_sim
        self.neg_sim = neg_sim

        true_sim = T.sum(true_sim, axis=1)
        true_sim = true_sim.reshape((batch_size, 1))
        true_sim = T.repeat(true_sim, type_size, axis=1) ## batchsize*labelsize

        train_type_signal = train_type_signal.reshape((train_type_signal.shape[0], 1))
        train_type_signal = T.repeat(train_type_signal, batch_size, axis=1)
        train_type_signal = T.transpose(train_type_signal, [1,0])  ## batchsize*labelsize

        sim_norm = sim*train_type_signal

        max_sim = T.max(sim, axis=1) ## batchsize*1
        max_sim = max_sim.reshape((max_sim.shape[0], 1))
        max_sim = T.repeat(max_sim, type_size, axis=1) ## batchsize*labelsize
        max_sim = max_sim*train_type_signal

        pos_neg_flag_norm = T.repeat(pos_neg_flag, type_size, axis=1)  ## pos == 1,  neg == 0

        pos_neg_flag_norm_1 = 1-pos_neg_flag_norm

        f1 = T.maximum(0, margin-max_sim+sim_norm)  ## num = #of train types
        loss1 = T.mean(T.sum(f1*pos_neg_flag_norm_1))

        f = T.maximum(0, margin-true_sim+neg_sim)

        self.pos_neg_flag_norm = pos_neg_flag_norm
        self.pos_neg_flag_norm_1 = pos_neg_flag_norm_1
        self.max_sim = max_sim
        self.sim_norm = sim_norm

        loss = T.mean(T.sum(T.max((T.maximum(0, margin-true_sim+neg_sim))*pos_neg_flag_norm, axis=1))) + \
               T.mean(T.sum(T.max(T.maximum(0, margin-max_sim+sim_norm)*pos_neg_flag_norm_1, axis=1)))

        self.p_y_given_x = sim
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.p = T.max(self.p_y_given_x, axis=1)

        self.loss = loss
        self.loss1 = loss1

    def results(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        #self.all_p_y_given_x = self.in2.dot(self.W) + self.b

        # check if y has same dimension of y_pred

        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction

            return [T.argmax(self.p_y_given_x, axis=1)+1, T.max(self.p_y_given_x, axis=1), self.p_y_given_x]
            #return self.p_y_given_x
        else:
            raise NotImplementedError()

            raise NotImplementedError()


    def cosine(self, input_1, input_2):
        input_2_t = input_2.T  ## n_in * labelsize
        multi = input_1.dot(input_2_t)  ## batchsize*labelsize

        l1 = T.sqrt(T.sum(T.sqr(input_1), axis=1)) ## batchsize*1
        l2 = T.sqrt(T.sum(T.sqr(input_2), axis=1)) ## labelsize*1

        l1 = l1.reshape((input_1.shape[0], 1))
        l1 = T.repeat(l1, input_2.shape[0], axis=1)

        l2 = l2.reshape((input_2.shape[0], 1))
        l2 = T.repeat(l2, input_1.shape[0], axis=1)
        l2 = T.transpose(l2, [1,0])

        sim_matrix = multi/(l1*l2)

        return sim_matrix


class MaxRankingMarginCosine1Arg1(object):
    def __init__(self, rng, input, input_label, true_label, n_in, n_in2, margin, batch_size, type_size,
                 train_type_signal, pos_neg_flag, limited_role):

        self.input = input
        self.input_label = input_label
        self.true_label = true_label

        W_values = numpy.asarray(rng.uniform(
                    # low=-numpy.sqrt(6. / (n_in * n_in2)),
                    # high=numpy.sqrt(6. / (n_in * n_in2)),
                    low=-numpy.sqrt(numpy.sqrt(6./(n_in * n_in2))),
                    high=numpy.sqrt(numpy.sqrt(6./(n_in * n_in2))),
                    size=(n_in, n_in2)), dtype=theano.config.floatX)

        W = theano.shared(value=W_values, borrow=True)

        self.W = W

        self.margin = margin

        self.params = [self.W]

        input_1 = input.dot(self.W)

        self.input_1 = input_1

        sim_0 = self.cosine(input, input_label)
        sim = sim_0*limited_role

        true_sim = sim*true_label
        neg_sim = sim  ## batchsize*labelsize

        self.true_sim = true_sim
        self.neg_sim = neg_sim

        true_sim = T.sum(true_sim, axis=1)
        true_sim = true_sim.reshape((batch_size, 1))
        true_sim = T.repeat(true_sim, type_size, axis=1) ## batchsize*labelsize

        train_type_signal = train_type_signal.reshape((train_type_signal.shape[0], 1))
        train_type_signal = T.repeat(train_type_signal, batch_size, axis=1)
        train_type_signal = T.transpose(train_type_signal, [1,0])  ## batchsize*labelsize

        sim_norm = sim*train_type_signal

        max_sim = T.max(sim, axis=1) ## batchsize*1
        max_sim = max_sim.reshape((max_sim.shape[0], 1))
        max_sim = T.repeat(max_sim, type_size, axis=1) ## batchsize*labelsize
        max_sim = max_sim*train_type_signal

        pos_neg_flag_norm = T.repeat(pos_neg_flag, type_size, axis=1)  ## pos == 1,  neg == 0

        pos_neg_flag_norm_1 = 1-pos_neg_flag_norm

        f1 = T.maximum(0, margin-max_sim+sim_norm)  ## num = #of train types
        loss1 = T.mean(T.sum(f1*pos_neg_flag_norm_1))

        f = T.maximum(0, margin-true_sim+neg_sim)
        ## loss = T.mean(T.sum(f*posNegFlagNorm)) + 0.5*loss1

        self.pos_neg_flag_norm = pos_neg_flag_norm
        self.pos_neg_flag_norm_1 = pos_neg_flag_norm_1
        self.max_sim = max_sim
        self.sim_norm = sim_norm

        ## loss1 = T.mean(T.sum(T.max(T.maximum(0, margin-maxSimRepeatNorm+simNorm)*posNegFlagNorm1, axis=1)))

        loss = T.mean(T.sum(T.max((T.maximum(0, margin-true_sim+neg_sim))*pos_neg_flag_norm, axis=1))) + \
               T.mean(T.sum(T.max(T.maximum(0, margin-max_sim+sim_norm)*pos_neg_flag_norm_1, axis=1)))

        ## loss = T.maximum(0, margin-trueSim3+negSim)

        self.p_y_given_x = sim
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.p = T.max(self.p_y_given_x, axis=1)

        self.loss = loss
        self.loss1 = loss1

    def results(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        #self.all_p_y_given_x = self.in2.dot(self.W) + self.b

        # check if y has same dimension of y_pred

        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction

            return [T.argmax(self.p_y_given_x, axis=1)+1, T.max(self.p_y_given_x, axis=1), self.p_y_given_x]
            #return self.p_y_given_x
        else:
            raise NotImplementedError()

            raise NotImplementedError()


    def cosine(self, input_1, input_2):
        input_2_t = input_2.T  ## n_in * labelsize
        multi = input_1.dot(input_2_t)  ## batchsize*labelsize

        l1 = T.sqrt(T.sum(T.sqr(input_1), axis=1)) ## batchsize*1
        l2 = T.sqrt(T.sum(T.sqr(input_2), axis=1)) ## labelsize*1

        l1 = l1.reshape((input_1.shape[0], 1))
        l1 = T.repeat(l1, input_2.shape[0], axis=1)

        l2 = l2.reshape((input_2.shape[0], 1))
        l2 = T.repeat(l2, input_1.shape[0], axis=1)
        l2 = T.transpose(l2, [1,0])

        sim_matrix = multi/(l1*l2)

        return sim_matrix



