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
    train_file = args.train
    dev_file = args.dev
    test_file = args.test
    label_file = args.ontology_path
    label_file_norm = args.norm_ontology_path
    relation_file = args.relation_path
    train_type_flag = args.seen_types

    tup_representation_size = embedding_size * 2

    # load word vectors
    word_vectors, vector_size = load_word_vec(embedding_file)

    # read train and dev file
    print ("start loading train and dev file ... ")
    doc_id_list_train, type_list_train, trigger_list_train, left_word_list_train, relation_list_train, \
        right_word_list_train = load_training_data(train_file)
    doc_id_list_dev, type_list_dev, trigger_list_dev, left_word_list_dev, relation_list_dev, \
        right_word_list_dev = load_training_data(dev_file)
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

    result_index_train_matrix, result_vector_train_matrix, input_context_train_matrix, input_trigger_train_matrix, \
        relation_binary_train_matrix, pos_neg_train_matrix = input_matrix_1(
            type_list_train, trigger_list_train, left_word_list_train, relation_list_train, right_word_list_train,
            embedding_size, mention_context_size, relation_size, label_file, word_vectors, rel_index, train_type_flag)

    result_index_dev_matrix, result_vector_dev_matrix, input_context_dev_matrix, input_trigger_dev_matrix, \
        relation_binary_dev_matrix, pos_neg_dev_matrix = input_matrix_1_test(
            type_list_dev, trigger_list_dev, left_word_list_dev, relation_list_dev, right_word_list_dev, embedding_size,
            mention_context_size, relation_size, label_file, word_vectors, rel_index, train_type_flag)

    result_index_test_matrix, result_vector_test_matrix, input_context_test_matrix, input_trigger_test_matrix, \
        relation_binary_test_matrix, pos_neg_test_matrix = input_matrix_1_test(
            type_list_test, trigger_list_test, left_word_list_test, relation_list_test, right_word_list_test,
            embedding_size, mention_context_size, relation_size, label_file, word_vectors, rel_index, train_type_flag)

    input_type_matrix, input_type_structure_matrix = type_matrix(
        all_type_list, all_type_structures, embedding_file, type_context_size)

    time1 = time.time()
    dt = theano.config.floatX
    train_set_content = theano.shared(numpy.matrix(input_context_train_matrix, dtype=dt))
    valid_set_content = theano.shared(numpy.matrix(input_context_dev_matrix, dtype=dt))
    test_set_content = theano.shared(numpy.matrix(input_context_test_matrix, dtype=dt))

    train_set_trigger = theano.shared(numpy.matrix(input_trigger_train_matrix, dtype=dt))
    valid_set_trigger = theano.shared(numpy.matrix(input_trigger_dev_matrix, dtype=dt))
    test_set_trigger = theano.shared(numpy.matrix(input_trigger_test_matrix, dtype=dt))

    train_set_relation_binary = theano.shared(numpy.matrix(relation_binary_train_matrix, dtype=dt))
    valid_set_relation_binary = theano.shared(numpy.matrix(relation_binary_dev_matrix, dtype=dt))
    test_set_relation_binary = theano.shared(numpy.matrix(relation_binary_test_matrix, dtype=dt))

    train_set_posneg = theano.shared(numpy.matrix(pos_neg_train_matrix, dtype=dt))
    valid_set_posneg = theano.shared(numpy.matrix(pos_neg_dev_matrix, dtype=dt))
    test_set_posneg = theano.shared(numpy.matrix(pos_neg_test_matrix, dtype=dt))

    train_set_y = theano.shared(numpy.array(result_index_train_matrix, dtype=numpy.dtype(numpy.int32)))
    valid_set_y = theano.shared(numpy.array(result_index_dev_matrix, dtype=numpy.dtype(numpy.int32)))
    test_set_y = theano.shared(numpy.array(result_index_test_matrix, dtype=numpy.dtype(numpy.int32)))

    train_set_y_vector = theano.shared(numpy.matrix(result_vector_train_matrix, dtype=dt))
    valid_set_y_vector = theano.shared(numpy.matrix(result_vector_dev_matrix, dtype=dt))
    test_set_y_vector = theano.shared(numpy.matrix(result_vector_test_matrix, dtype=dt))

    train_set_type = theano.shared(numpy.matrix(input_type_matrix, dtype=dt))
    train_set_type_structure = theano.shared(numpy.matrix(input_type_structure_matrix, dtype=dt))

    train_types = theano.shared(numpy.matrix(train_types, dtype=dt))

    # compute number of minibatches for training, validation and testing
    n_train_batches = input_trigger_train_matrix.shape[0]
    n_valid_batches = input_trigger_dev_matrix.shape[0]
    n_test_batches = input_trigger_test_matrix.shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
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

    validVariables = {}
    validVariables[x_content] = valid_set_content[start: end]
    validVariables[x_trigger] = valid_set_trigger[start: end]
    validVariables[x_relation_binary] = valid_set_relation_binary[start: end]
    validVariables[x_type] = train_set_type
    validVariables[x_type_structure] = train_set_type_structure
    validVariables[y] = valid_set_y[start: end]
    validVariables[y_vector] = valid_set_y_vector[start: end]
    validVariables[x_train_types] = train_types
    validVariables[x_pos_neg_flag] = valid_set_posneg[start: end]

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

    trainVariables = {}
    trainVariables[x_content] = train_set_content[start: end]
    trainVariables[x_trigger] = train_set_trigger[start: end]
    trainVariables[x_relation_binary] = train_set_relation_binary[start: end]
    trainVariables[x_type] = train_set_type
    trainVariables[x_type_structure] = train_set_type_structure
    trainVariables[y] = train_set_y[start: end]
    trainVariables[y_vector] = train_set_y_vector[start: end]
    trainVariables[x_train_types] = train_types
    trainVariables[x_pos_neg_flag] = train_set_posneg[start: end]

    print ("length of train variables ", len(trainVariables))

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by SGD Since this model has many parameters,
    # it would be tedious to manually create an update rule for each model parameter. We thus create the updates
    # list by automatically looping over all (params[i],grads[i]) pairs.
    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - lr * grad_i))

    test_model_confidence = theano.function([index], layer3.results(y), on_unused_input='ignore', givens=testVariables)
    eval_model_confidence = theano.function([index], layer3.results(y), on_unused_input='ignore', givens=validVariables)
    train_model = theano.function([index, lr, lambd1, lambd2], [cost, layer3.loss], updates=updates,
                                  on_unused_input='ignore', givens=trainVariables)

    time2 = time.time()
    print ("time for building the model: ", time2 - time1)

    # Train the  MODEL
    print ('start training ... ')
    time1 = time.time()

    num_examples_per_epoch = len(trigger_list_train)
    validation_frequency = num_examples_per_epoch / batch_size  # validate after each epoch
    best_params = []
    best_micro_fscore = -1
    last_fscore = -1
    best_macro_fscore = -1
    best_iter = 0
    best_micro_fscore_eval = -1
    best_macro_fscore_eval = -1
    best_iter_eval = 0

    start_time = time.clock()

    epoch = 0
    done_looping = False

    max_no_improvement = 5
    no_improvement = 0

    while (epoch < n_epochs) and (not done_looping):
        print ('epoch = ', epoch)
        epoch += 1
        this_n_train_batches = num_examples_per_epoch / batch_size
        for minibatch_index in xrange(this_n_train_batches):
            iter = (epoch - 1) * this_n_train_batches + minibatch_index
            if iter % 100 == 0:
                print ('training @ iter = ', iter)
            cost_ij, loss = train_model(minibatch_index, learning_rate, l1, l2)

            print ("cost: " + str(cost_ij))
            print ("loss:   " + str(loss))

            if (iter + 1) % validation_frequency == 0:
                # test
                confidence_eval = [test_model_confidence(i) for i in xrange(n_test_batches)]
                confidence_list_eval = []
                for r in range(0, len(confidence_eval)):
                    for r1 in range(0, batch_size):
                        hypo_result_eval = confidence_eval[r][0].item(r1)
                        confidence_list_eval.append(hypo_result_eval)

                y_pred_eval = confidence_list_eval
                y_true_eval = result_index_test_matrix[:n_test_batches * batch_size]
                y_true_eval_2 = []
                for i in range(len(y_true_eval)):
                    y_true_eval_2.append(int(y_true_eval[i]))

                labels1 = []
                for l in range(1, 380):
                    labels1.append(l)
                this_micro_fscore_eval = f1_score(y_true_eval_2, y_pred_eval, labels=labels1, average='micro')
                this_macro_fscore_eval = f1_score(y_true_eval_2, y_pred_eval, labels=labels1, average='macro')
                print('EVAL: ***   epoch %i, best_validation %f, best_validation_m1 %f, learning_rate %f,  '
                      'minibatch %i/%i, validation fscore %f %%' % (epoch, best_micro_fscore_eval * 100.,
                                                                    best_macro_fscore_eval * 100, learning_rate,
                                                                    minibatch_index + 1, this_n_train_batches,
                                                                    this_micro_fscore_eval * 100.))

                if this_micro_fscore_eval > best_micro_fscore_eval:
                    best_micro_fscore_eval = this_micro_fscore_eval
                    best_macro_fscore_eval = this_macro_fscore_eval
                    best_iter_eval = iter

                # validate
                confidence = [eval_model_confidence(i) for i in xrange(n_valid_batches)]

                confidence_list = []
                for r in range(0, len(confidence)):
                    for r1 in range(0, batch_size):
                        hypo_result = confidence[r][0].item(r1)
                        confidence_list.append(hypo_result)

                y_pred = confidence_list
                y_true = result_index_dev_matrix[:n_valid_batches * batch_size]
                y_true_2 = []
                for i in range(len(y_true)):
                    y_true_2.append(int(y_true[i]))

                labels = []
                for l in range(1, 380):
                    labels.append(l)
                this_micro_fscore = f1_score(y_true_2, y_pred, labels=labels, average='micro')
                this_macro_fscore = f1_score(y_true_2, y_pred, labels=labels, average='macro')

                print('epoch %i, best_validation %f, best_validation_m1 %f, learning_rate %f, minibatch %i/%i, '
                      'validation fscore %f %%' % (epoch, best_micro_fscore * 100., best_macro_fscore * 100,
                                                   learning_rate, minibatch_index + 1, this_n_train_batches,
                                                   this_micro_fscore * 100.))

                # if we got the best validation score until now
                if this_micro_fscore > best_micro_fscore:
                    best_micro_fscore = this_micro_fscore
                    best_macro_fscore = this_macro_fscore
                    best_iter = iter

                    best_params = []
                    for p in param_list:
                        p_param = []
                        for part in p:
                            p_param.append(part.get_value(borrow=False))
                        best_params.append(p_param)
                    no_improvement = 0
                else:
                    if this_micro_fscore > last_fscore:
                        no_improvement -= 1
                        no_improvement = max(no_improvement, 0)
                    else:
                        no_improvement += 1
                        updatestep = minibatch_index + this_n_train_batches * (epoch - 1)
                        if newbob:  # learning rate schedule depending on dev result
                            learning_rate /= 1.2
                            print ("reducing learning rate to ", learning_rate)
                last_fscore = this_micro_fscore
            if newbob:  # learning rate schedule depending on dev result
                if no_improvement > max_no_improvement or learning_rate < 0.0000001:
                    done_looping = True
                    break

        if not newbob:
            if epoch + 1 > 10:
                learning_rate /= 1.2
                print ("reducing learning rate to ", learning_rate)
            if epoch + 1 > 50:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained for c=%i, nk=%i, f=%i, h=%i  at iteration %i,' %
          (best_micro_fscore * 100., mention_context_size, nkerns[0], filter_size[1], hidden_units, best_iter + 1))

    time2 = time.time()
    print ("time for training: ", time2 - time1)

    print('Saving net.')
    save_file = open(network_file, 'wb')
    for p in best_params:
        for p_part in p:
            cPickle.dump(p_part, save_file, -1)
    save_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='data/aceEventStructure.train.ace.format.transfer.neg.10.txt',
                        help='path for train file')
    parser.add_argument('--dev', type=str, default='data/aceEventStructure.dev.ace.format.transfer.neg.10.txt',
                        help='path for dev file')
    parser.add_argument('--test', type=str, default='data/aceEventStructure.test.ace.format.transfer.neg.10.txt',
                        help='path for test file')
    parser.add_argument('--model_path', type=str, default='model/final.model',
                        help='path for saving trained models')
    parser.add_argument('--embedding_path', type=str,
                        default='data/wsd.model.ace.filter.txt',
                        help='path for pretrained word embedding')
    parser.add_argument('--ontology_path', type=str, default='data/aceArgs.txt',
                        help='path for predefined ontology')
    parser.add_argument('--norm_ontology_path', type=str, default='data/aceArgs.normalize.txt',
                        help='path for predefined ontology')
    parser.add_argument('--seen_types', type=str, default='originalData/train.10', help='tag file for seen event types')
    parser.add_argument('--relation_path', type=str, default='data/amrRelations.txt', help='amr relations')

    parser.add_argument('--mention_context_size', type=int, default=5, help='number of mention tuples')
    parser.add_argument('--type_context_size', type=int, default=5, help='number of type tuples')
    parser.add_argument('--hidden_units', type=int, default=200, help='size of hidden units')
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--embedding_size', type=int, default=200)
    parser.add_argument('--num_epochs', type=int, default=50)

    args = parser.parse_args()
    print(args)
    main(args)