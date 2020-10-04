import numpy
import time
import argparse
import theano
import cPickle
import theano.tensor as T
from sklearn.metrics import f1_score
from layers import LeNetConvPoolLayer, ComposeLayerMatrix, ComposeLayerTensor, MaxRankingMarginCosine1
from utils import load_word_vec, load_training_data, read_relation_index, type_matrix, \
    random_init_rel_vec_factor, load_types_1, get_types_for_train, input_matrix_1, input_matrix_1_test


__author__ = 'wilbur'


def main(args):
    # initial parameters
    embedding_size = args.embedding_size
    mention_context_size = args.mention_context_size
    type_context_size = args.type_context_size
    embedding_file = args.embedding_path
    hidden_units = args.hidden_units
    learning_rate = args.learning_rate
    margin = args.margin
    batch_size = args.batch_size
    n_epochs = args.num_epochs

    relation_size = 82
    nkerns = [500]
    filter_size = [1, 1]
    pool = [1, 1]
    l1 = 0.000001
    l2 = 0.000002

    newbob = False
    network_file = args.model_path
    test_file = args.test
    test_result_file = args.test_result
    label_file = args.ontology_path
    label_file_norm = args.norm_ontology_path
    relation_file = args.relation_path
    train_type_flag = args.seen_types

    tup_representation_size = embedding_size * 2

    # load word vectors
    word_vectors, vector_size = load_word_vec(embedding_file)

    # read train and dev file
    print ("start loading train and dev file ... ")
    doc_id_list_test, type_list_test, trigger_list_test, left_word_list_test, relation_list_test, \
        right_word_list_test = load_training_data(test_file)

    print ("start loading arg and relation files ... ")
    all_type_list, all_type_structures = load_types_1(label_file_norm)
    rel_index, index_rel = read_relation_index(relation_file)
    type_size = len(all_type_list)

    # using a matrix to represent each relation
    relation_matrix = random_init_rel_vec_factor(relation_file, tup_representation_size*tup_representation_size)

    train_types = get_types_for_train(train_type_flag, label_file)

    # prepare data structure
    print ("start preparing data structures ... ")
    curSeed = 23455
    rng = numpy.random.RandomState(curSeed)
    seed = rng.get_state()[1][0]
    print ("seed: ", seed)

    result_index_test_matrix, result_vector_test_matrix, input_context_test_matrix, input_trigger_test_matrix, \
        relation_binary_test_matrix, pos_neg_test_matrix = input_matrix_1_test(
            type_list_test, trigger_list_test, left_word_list_test, relation_list_test, right_word_list_test,
            embedding_size, mention_context_size, relation_size, label_file, word_vectors, rel_index, train_type_flag)

    input_type_matrix, input_type_structure_matrix = type_matrix(
        all_type_list, all_type_structures, embedding_file, type_context_size)

    time1 = time.time()
    dt = theano.config.floatX
    test_set_content = theano.shared(numpy.matrix(input_context_test_matrix, dtype=dt))
    test_set_trigger = theano.shared(numpy.matrix(input_trigger_test_matrix, dtype=dt))
    test_set_relation_binary = theano.shared(numpy.matrix(relation_binary_test_matrix, dtype=dt))
    test_set_posneg = theano.shared(numpy.matrix(pos_neg_test_matrix, dtype=dt))
    test_set_y = theano.shared(numpy.array(result_index_test_matrix, dtype=numpy.dtype(numpy.int32)))
    test_set_y_vector = theano.shared(numpy.matrix(result_vector_test_matrix, dtype=dt))

    train_set_type = theano.shared(numpy.matrix(input_type_matrix, dtype=dt))
    train_set_type_structure = theano.shared(numpy.matrix(input_type_structure_matrix, dtype=dt))

    train_types = theano.shared(numpy.matrix(train_types, dtype=dt))

    # compute number of minibatches for training, validation and testing
    n_test_batches = input_trigger_test_matrix.shape[0]
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x_content = T.matrix('x_content')  # the data is presented as rasterized images
    x_trigger = T.matrix('x_trigger')  # the data is presented as rasterized images
    x_relation_binary = T.matrix('x_relation_binary')
    x_pos_neg_flag = T.matrix('x_pos_neg_flag')
    x_type = T.matrix('x_type')
    x_type_structure = T.matrix('x_type_structure')
    y = T.ivector('y')  # the labels are presented as 1D vector of
    y_vector = T.matrix('y_vector')  # the labels are presented as 1D vector of
    x_train_types = T.matrix('x_train_types')

    # [int] labels
    i_shape = [tup_representation_size, mention_context_size]  # this is the size of context matrizes

    time2 = time.time()
    print ("time for preparing data structures: ", time2 - time1)

    # build actual model

    print ('start building the model ... ')
    time1 = time.time()

    rel_w = theano.shared(value=relation_matrix, borrow=True)  ## 26*400

    # Construct the mention structure input Layer
    layer0_input = x_content.reshape((batch_size, 1, i_shape[0], i_shape[1]))
    layer0_input_binary_relation = x_relation_binary.reshape((batch_size, 1, relation_size, i_shape[1]))  ## 100*1*26*5

    # compose amr relation matrix to each tuple
    compose_layer = ComposeLayerMatrix(input=layer0_input, input_binary_relation=layer0_input_binary_relation,
                                      rel_w=rel_w, rel_vec_size=tup_representation_size)
    layer1_input = compose_layer.output

    # initialize the convolution weight matrix
    filter_shape = (nkerns[0], 1, tup_representation_size, filter_size[1])
    pool_size = (pool[0], pool[1])

    fan_in = numpy.prod(filter_shape[1:])
    fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(pool_size))

    w_bound = numpy.sqrt(6. / (fan_in + fan_out))

    conv_w = theano.shared(numpy.asarray(
        rng.uniform(low=-w_bound, high=w_bound, size=filter_shape),
        dtype=theano.config.floatX),
        borrow=True)

    b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
    conv_b = theano.shared(value=b_values, borrow=True)

    # conv with pool layer
    layer1_conv = LeNetConvPoolLayer(rng, W=conv_w, b=conv_b, input=layer1_input,
                                     image_shape=(batch_size, 1, i_shape[0], i_shape[1]),
                                     filter_shape=filter_shape, poolsize=pool_size)

    layer1_output = layer1_conv.output
    layer1_flattened = layer1_output.flatten(2)

    trigger_features_shaped = x_trigger.reshape((batch_size, embedding_size))

    layer2_input = T.concatenate([layer1_flattened, trigger_features_shaped], axis=1)

    # Construct the type structure input Layer
    layer_type_input = x_type_structure.reshape((type_size, 1, tup_representation_size, type_context_size))
    filter_shape_type = (nkerns[0], 1, tup_representation_size, filter_size[1])
    pool_size_type = (pool[0], pool[1])

    # initialize the implicit relation tensor
    type_tensor_shape = (tup_representation_size, tup_representation_size, tup_representation_size)
    type_tensor_w = theano.shared(numpy.asarray(rng.uniform(low=-w_bound, high=w_bound, size=type_tensor_shape),
                                                dtype=theano.config.floatX), borrow=True)

    # compose relation tensor to each tuple
    compose_type_layer = ComposeLayerTensor(input=layer_type_input, tensor=type_tensor_w)
    layer_type_input1 = compose_type_layer.output

    # conv with pool layer
    layer1_conv_type = LeNetConvPoolLayer(rng, W=conv_w, b=conv_b, input=layer_type_input1,
                                          image_shape=(type_size, 1, tup_representation_size, type_context_size),
                                          filter_shape=filter_shape_type, poolsize=pool_size_type)

    layer1_type_output = layer1_conv_type.output
    layer1_type_flattened = layer1_type_output.flatten(2)

    types_shaped = x_type.reshape((type_size, embedding_size))

    layer2_type_input = T.concatenate([layer1_type_flattened, types_shaped], axis=1)
    layer2_type_input_size = nkerns[0] ** pool[1] + embedding_size

    # ranking based max margin loss layer
    train_types_signal = x_train_types.reshape((type_size, 1))
    pos_neg_flag = x_pos_neg_flag.reshape((batch_size, 1))

    layer3 = MaxRankingMarginCosine1(rng=rng, input=layer2_input, input_label=layer2_type_input, true_label=y_vector,
                                     n_in=layer2_type_input_size, margin=margin, batch_size=batch_size,
                                     type_size=type_size, train_type_signal=train_types_signal,
                                     pos_neg_flag=pos_neg_flag)
    cost = layer3.loss

    # create a list of all model parameters to be fit by gradient descent
    param_list = [compose_layer.params, layer1_conv.params, compose_type_layer.params]

    params = []
    for p in param_list:
        params += p

    # the cost we minimize during training is the NLL of the model
    lambd1 = T.scalar('lambda1', dt)
    lambd2 = T.scalar('lambda2', dt)

    # L1 and L2 regularization possible
    reg2 = 0
    reg1 = 0
    for p in param_list:
        reg2 += T.sum(p[0] ** 2)
        reg1 += T.sum(abs(p[0]))

    cost += lambd2 * reg2
    cost += lambd1 * reg1

    lr = T.scalar('lr', dt)

    start = index * batch_size
    end = (index + 1) * batch_size

    testVariables = {}
    testVariables[x_content] = test_set_content[start: end]
    testVariables[x_trigger] = test_set_trigger[start: end]
    testVariables[x_relation_binary] = test_set_relation_binary[start: end]
    testVariables[x_type] = train_set_type
    testVariables[x_type_structure] = train_set_type_structure
    testVariables[y] = test_set_y[start: end]
    testVariables[y_vector] = test_set_y_vector[start: end]
    testVariables[x_train_types] = train_types
    testVariables[x_pos_neg_flag] = test_set_posneg[start: end]

    print ("length of train variables ", len(testVariables))

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by SGD Since this model has many parameters,
    # it would be tedious to manually create an update rule for each model parameter. We thus create the updates
    # list by automatically looping over all (params[i],grads[i]) pairs.
    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - lr * grad_i))

    test_model_confidence = theano.function([index], layer3.results(y), on_unused_input='ignore', givens=testVariables)

    time2 = time.time()
    print ("time for building the model: ", time2 - time1)

    print ("loading saved network")
    netfile = open(network_file)

    relW = cPickle.load(netfile)
    compose_layer.params[0].set_value(relW, borrow=True)

    convolW = cPickle.load(netfile)
    convolB = cPickle.load(netfile)
    layer1_conv.params[0].set_value(convolW, borrow=True)
    layer1_conv.params[1].set_value(convolB, borrow=True)
    layer1_conv_type.params[0].set_value(convolW, borrow=True)
    layer1_conv_type.params[1].set_value(convolB, borrow=True)

    typeW = cPickle.load(netfile)
    compose_type_layer.params[0].set_value(typeW, borrow=True)
    netfile.close()

    print ("finish loading network")

    test_batch_size = 100
    all_batches = len(result_index_test_matrix)/test_batch_size

    confidence_prob = []
    confidence_value = []
    confidence_list = []
    confidence = [test_model_confidence(i) for i in xrange(all_batches)]
    for r in range(0,len(confidence)):
      for r1 in range(0, test_batch_size):
        hypo_result = confidence[r][0].item(r1)
        confidence_prob.append(confidence[r][2][r1])
        confidence_value.append(confidence[r][1][r1])
        confidence_list.append(hypo_result)

    y_pred = confidence_list

    f = open(test_result_file, "w")
    for i in range(0,len(y_pred)):
        f.write(str(y_pred[i]) + "\t" + str(confidence_value[i]) + "\t")
        for j in range(0, type_size):
            f.write(str(confidence_prob[i][j]) + " ")
        f.write("\n")

    f.close()

if __name__ == '__main__':
    root = ""
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default=root+'ACL-Zero-Shot-Data/classification/aceEventStructure.test.ace.format.transfer.neg.10.txt',
                        help='path for test file')
    parser.add_argument('--test_result', type=str, default=root + 'ACL-Zero-Shot-Data/classification/trigger.test.result',
                        help='path for test result file')
    parser.add_argument('--model_path', type=str, default=root+'ACL-Zero-Shot-Data/model/framenet.model',
                        help='path for saving trained models')
    parser.add_argument('--embedding_path', type=str,
                        default=root+'ACL-Zero-Shot-Data/embedding/wsd.model.ace.filter.txt',
                        help='path for pretrained word embedding')
    parser.add_argument('--ontology_path', type=str, default=root+'ACL-Zero-Shot-Data/frame-ontology/event.ontology.new.txt',
                        help='path for predefined ontology')
    parser.add_argument('--norm_ontology_path', type=str, default=root+'ACL-Zero-Shot-Data/frame-ontology/event.ontology.normalize.new.txt',
                        help='path for predefined ontology')
    parser.add_argument('--seen_types', type=str, default=root+'ACL-Zero-Shot-Data/flags/train.10', help='tag file for seen event types')
    parser.add_argument('--relation_path', type=str, default=root+'ACL-Zero-Shot-Data/amr/amrRelations.txt', help='amr relations')

    parser.add_argument('--mention_context_size', type=int, default=5, help='number of mention tuples')
    parser.add_argument('--type_context_size', type=int, default=5, help='number of type tuples')
    parser.add_argument('--hidden_units', type=int, default=200, help='size of hidden units')
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--embedding_size', type=int, default=200)
    parser.add_argument('--num_epochs', type=int, default=5)

    args = parser.parse_args()
    print(args)
    main(args)
