import argparse
import numpy
import time
import cPickle
import theano
import theano.tensor as T
from sklearn.metrics import f1_score
from theano import config
from layers import LeNetConvPoolLayer, ComposeLayerMatrix, ComposeLayerTensor, MaxRankingMarginCosine1Arg1
from utils import load_word_vec, load_types, read_relation_index, random_init_rel_vec_factor, load_arg_data, \
    input_arg_matrix, load_roles_1, get_trigger_arg_matrix, role_matrix_1, get_roles_for_train_1, input_arg_matrix_test


__author__ = 'wilbur'


def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def sgd(lr, tparams, grads, x, mask, y, cost):
    """ Stochastic Gradient Descent"""

    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup, name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    f_update = theano.function([lr], [], updates=pup, name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, mask, y, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]

    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore', name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, y, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / T.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')
    return f_grad_shared, f_update


def main(args):
    # initial parameters
    embedding_size = args.embedding_size
    arg_context_size = args.arg_context_size
    role_context_size = args.role_context_size
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
    l1 = 0.001
    l2 = 0.002

    newbob = False
    arg_network_file = args.model_path
    arg_train_file = args.train
    arg_dev_file = args.dev
    arg_test_file = args.test
    arg_label_file = args.ontology_path
    arg_label_file_norm = args.norm_ontology_path
    relation_file = args.relation_path
    train_role_flag = args.seen_args
    arg_path_file_merge = args.arg_path_file
    arg_path_file_universal = args.arg_path_file_universal
    trigger_role_matrix_file = args.trigger_role_matrix
    tup_representation_size = embedding_size * 2

    # load word vectors
    word_vectors, vector_size = load_word_vec(embedding_file)


    # read train and dev file
    print ("start loading train and dev file ... ")

    arg_trigger_list_train, arg_trigger_type_list_train, arg_list_train, arg_path_left_list_train, \
        arg_path_rel_list_train, arg_path_right_list_train, arg_role_list_train = load_arg_data(arg_train_file)

    arg_trigger_list_dev, arg_trigger_type_list_dev, arg_list_dev, arg_path_left_list_dev, \
        arg_path_rel_list_dev, arg_path_right_list_dev, arg_role_list_dev = load_arg_data(arg_dev_file)

    arg_trigger_list_test, arg_trigger_type_list_test, arg_list_test, arg_path_left_list_test, \
        arg_path_rel_list_test, arg_path_right_list_test, arg_role_list_test = load_arg_data(arg_test_file)


    num_examples_per_epoch = len(arg_trigger_list_train)

    print ("start loading arg and relation files ... ")
    all_type_list, all_type_structures = load_types(arg_label_file_norm)
    type_size = len(all_type_list)

    all_arg_role_list, all_type_role_structures, index_2_role, trigger_role_2_index, index_2_norm_role, \
    trigger_norm_role_2_index = load_roles_1(arg_path_file_merge)
    role_size = len(all_arg_role_list)

    trigger_role_matrix = get_trigger_arg_matrix(trigger_role_matrix_file, type_size, role_size)
    train_roles = get_roles_for_train_1(train_role_flag, arg_path_file_merge)

    rel_2_index, index_2_rel = read_relation_index(relation_file)
    relation_matrix = random_init_rel_vec_factor(relation_file, tup_representation_size*tup_representation_size)

    print ("start preparing data structures ... ")
    curSeed = 23455
    rng = numpy.random.RandomState(curSeed)
    seed = rng.get_state()[1][0]
    print ("seed: ", seed)

    # arg data matrix
    role_index_train_matrix, role_vector_train_matrix, input_arg_context_train_matrix, input_arg_train_matrix, \
        arg_relation_binary_train_matrix, pos_neg_role_train_matrix, limited_roles_train_matrix = \
        input_arg_matrix(arg_trigger_list_train, arg_trigger_type_list_train, arg_list_train, arg_path_left_list_train,
                         arg_path_rel_list_train, arg_path_right_list_train, arg_role_list_train, word_vectors,
                         all_arg_role_list, trigger_role_2_index, vector_size, arg_context_size, relation_size,
                         rel_2_index, train_roles, trigger_role_matrix, arg_label_file)

    role_index_dev_matrix, role_vector_dev_matrix, input_arg_context_dev_matrix, input_arg_dev_matrix, \
        arg_relation_binary_dev_matrix, pos_neg_role_dev_matrix, limited_roles_dev_matrix = \
        input_arg_matrix_test(arg_trigger_list_dev, arg_trigger_type_list_dev, arg_list_dev, arg_path_left_list_dev,
                         arg_path_rel_list_dev,arg_path_right_list_dev, arg_role_list_dev, word_vectors,
                         all_arg_role_list, trigger_role_2_index, vector_size, arg_context_size, relation_size,
                         rel_2_index, train_roles, trigger_role_matrix, arg_label_file)

    role_index_test_matrix, role_vector_test_matrix, input_arg_context_test_matrix, input_arg_test_matrix, \
        arg_relation_binary_test_matrix, pos_neg_role_test_matrix, limited_roles_test_matrix = \
        input_arg_matrix_test(arg_trigger_list_test, arg_trigger_type_list_test, arg_list_test, arg_path_left_list_test,
                              arg_path_rel_list_test, arg_path_right_list_test, arg_role_list_test, word_vectors,
                              all_arg_role_list, trigger_role_2_index, vector_size, arg_context_size, relation_size,
                              rel_2_index, train_roles, trigger_role_matrix, arg_label_file)

    input_role_matrix, input_role_structure_matrix = role_matrix_1(
        all_arg_role_list, all_type_role_structures, embedding_file, role_context_size)

    time1 = time.time()
    dt = theano.config.floatX

    ## arg data
    train_set_content_arg = theano.shared(numpy.matrix(input_arg_context_train_matrix, dtype=dt))
    valid_set_content_arg = theano.shared(numpy.matrix(input_arg_context_dev_matrix, dtype=dt))
    test_set_content_arg = theano.shared(numpy.matrix(input_arg_context_test_matrix, dtype=dt))

    train_set_arg = theano.shared(numpy.matrix(input_arg_train_matrix, dtype=dt))
    valid_set_arg = theano.shared(numpy.matrix(input_arg_dev_matrix, dtype=dt))
    test_set_arg = theano.shared(numpy.matrix(input_arg_test_matrix, dtype=dt))

    train_set_relation_binary_arg = theano.shared(numpy.matrix(arg_relation_binary_train_matrix, dtype=dt))
    valid_set_relation_binary_arg = theano.shared(numpy.matrix(arg_relation_binary_dev_matrix, dtype=dt))
    test_set_relation_binary_arg = theano.shared(numpy.matrix(arg_relation_binary_test_matrix, dtype=dt))

    train_set_posneg_arg = theano.shared(numpy.matrix(pos_neg_role_train_matrix, dtype=dt))
    valid_set_posneg_arg = theano.shared(numpy.matrix(pos_neg_role_dev_matrix, dtype=dt))
    test_set_posneg_arg = theano.shared(numpy.matrix(pos_neg_role_test_matrix, dtype=dt))

    train_set_arg_y = theano.shared(numpy.array(role_index_train_matrix, dtype=numpy.dtype(numpy.int32)))
    valid_set_arg_y = theano.shared(numpy.array(role_index_dev_matrix, dtype=numpy.dtype(numpy.int32)))
    test_set_arg_y = theano.shared(numpy.array(role_index_test_matrix, dtype=numpy.dtype(numpy.int32)))

    train_set_arg_y_vector = theano.shared(numpy.matrix(role_vector_train_matrix, dtype=dt))
    valid_set_arg_y_vector = theano.shared(numpy.matrix(role_vector_dev_matrix, dtype=dt))
    test_set_arg_y_vector = theano.shared(numpy.matrix(role_vector_test_matrix, dtype=dt))

    train_set_arg_limited_role = theano.shared(numpy.matrix(limited_roles_train_matrix, dtype=dt))
    valid_set_arg_limited_role = theano.shared(numpy.matrix(limited_roles_dev_matrix, dtype=dt))
    test_set_arg_limited_role = theano.shared(numpy.matrix(limited_roles_test_matrix, dtype=dt))

    train_set_role = theano.shared(numpy.matrix(input_role_matrix, dtype=dt))
    train_set_role_structure = theano.shared(numpy.matrix(input_role_structure_matrix, dtype=dt))

    train_roles = theano.shared(numpy.matrix(train_roles, dtype=dt))

    # compute number of minibatches for training, validation and testing
    n_train_batches = input_arg_train_matrix.shape[0]
    n_valid_batches = input_arg_dev_matrix.shape[0]
    n_test_batches = input_arg_test_matrix.shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x_content_arg = T.matrix('x_content_arg')
    x_arg = T.matrix('x_arg')
    x_relation_binary_arg = T.matrix('x_relation_binary_arg')
    x_pos_neg_flag_arg = T.matrix('x_pos_neg_flag_arg')
    x_role = T.matrix('x_role')
    x_role_structure = T.matrix('x_role_structure')
    x_train_roles = T.matrix('x_train_roles')
    arg_y = T.ivector('arg_y')
    arg_y_vector = T.matrix('arg_y_vector')
    arg_limited_role = T.matrix('arg_limited_role')

    # [int] labels
    ishape = [tup_representation_size, arg_context_size]  # this is the size of context matrizes

    time2 = time.time()
    print ("time for preparing data structures: ", time2 - time1)

    # build the actual model
    print ('start building the model ... ')
    time1 = time.time()

    # argument representation layer
    layer0_arg_input = x_content_arg.reshape((batch_size, 1, ishape[0], ishape[1]))
    layer0_input_binary_relation = x_relation_binary_arg.reshape((batch_size, 1, relation_size, ishape[1]))  ## 100*1*26*5

    # compose amr relation matrix to each tuple
    rel_w = theano.shared(value=relation_matrix, borrow=True)  ## 26*400
    compose_layer = ComposeLayerMatrix(input=layer0_arg_input, input_binary_relation=layer0_input_binary_relation,
                                       rel_w=rel_w, rel_vec_size=tup_representation_size)

    layer1_input = compose_layer.output

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

    layer1_arg_conv = LeNetConvPoolLayer(rng, W=conv_w, b=conv_b, input=layer1_input,
                                         image_shape=(batch_size, 1, ishape[0], arg_context_size),
                                         filter_shape=filter_shape, poolsize=pool_size)

    layer1_arg_output = layer1_arg_conv.output
    layer1_arg_flattened = layer1_arg_output.flatten(2)
    arg_features_shaped = x_arg.reshape((batch_size, embedding_size))
    layer2_arg_input = T.concatenate([layer1_arg_flattened, arg_features_shaped], axis=1)
    layer2_arg_input_size = nkerns[0] * pool[1] + embedding_size

    # arg role representation layer
    layer_role_input = x_role_structure.reshape((role_size, 1, tup_representation_size, role_context_size))
    filter_shape_role = (nkerns[0], 1, tup_representation_size, filter_size[1])
    pool_size_role = (pool[0], pool[1])

    # initialize the implicit relation tensor
    type_tensor_shape = (tup_representation_size, tup_representation_size, tup_representation_size)
    type_tensor_w = theano.shared(numpy.asarray(rng.uniform(low=-w_bound, high=w_bound, size=type_tensor_shape),
                                                dtype=theano.config.floatX), borrow=True)

    # compose relation tensor to each tuple
    compose_type_layer = ComposeLayerTensor(input=layer_role_input, tensor=type_tensor_w)
    layer_type_input1 = compose_type_layer.output

    layer1_conv_role = LeNetConvPoolLayer(rng, W=conv_w, b=conv_b, input=layer_type_input1,
                                          image_shape=(role_size, 1, tup_representation_size, role_context_size),
                                          filter_shape=filter_shape_role, poolsize=pool_size_role)

    layer1_role_output = layer1_conv_role.output
    layer1_role_flattened = layer1_role_output.flatten(2)

    role_shaped = x_role.reshape((role_size, embedding_size))

    layer2_role_input = T.concatenate([layer1_role_flattened, role_shaped], axis=1)
    layer2_role_input_size = nkerns[0] ** pool[1] + embedding_size

    # ranking based max margin loss layer
    train_roles_signal = x_train_roles.reshape((role_size, 1))
    pos_neg_flag_arg = x_pos_neg_flag_arg.reshape((batch_size, 1))
    limited_role = arg_limited_role.reshape((batch_size, role_size))

    layer3 = MaxRankingMarginCosine1Arg1(rng=rng, input=layer2_arg_input, input_label=layer2_role_input,
                                         true_label=arg_y_vector, n_in=layer2_arg_input_size,
                                         n_in2=layer2_role_input_size, margin=margin, batch_size=batch_size,
                                         type_size=role_size, train_type_signal=train_roles_signal,
                                         pos_neg_flag=pos_neg_flag_arg, limited_role=limited_role)

    # cost and parameters update
    cost = layer3.loss
    # create a list of all model parameters to be fit by gradient descent
    param_list = [layer1_arg_conv.params, compose_layer.params, compose_type_layer.params]

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

    print ("reg1 ", reg1)
    print ("reg2 ", reg2)

    cost += lambd2 * reg2
    cost += lambd1 * reg1

    lr = T.scalar('lr', dt)

    start = index * batch_size
    end = (index + 1) * batch_size

    validVariables = {}
    validVariables[x_content_arg] = valid_set_content_arg[start: end]
    validVariables[x_arg] = valid_set_arg[start: end]
    validVariables[x_role] = train_set_role
    validVariables[x_role_structure] = train_set_role_structure
    validVariables[x_relation_binary_arg] = valid_set_relation_binary_arg[start: end]
    validVariables[arg_y] = valid_set_arg_y[start: end]
    validVariables[arg_y_vector] = valid_set_arg_y_vector[start: end]
    validVariables[x_train_roles] = train_roles
    validVariables[x_pos_neg_flag_arg] = valid_set_posneg_arg[start: end]
    validVariables[arg_limited_role] = valid_set_arg_limited_role[start: end]

    testVariables = {}
    testVariables[x_content_arg] = test_set_content_arg[start: end]
    testVariables[x_arg] = test_set_arg[start: end]
    testVariables[x_role] = train_set_role
    testVariables[x_role_structure] = train_set_role_structure
    testVariables[x_relation_binary_arg] = test_set_relation_binary_arg[start: end]
    testVariables[arg_y] = test_set_arg_y[start: end]
    testVariables[arg_y_vector] = test_set_arg_y_vector[start: end]
    testVariables[x_train_roles] = train_roles
    testVariables[x_pos_neg_flag_arg] = test_set_posneg_arg[start: end]
    testVariables[arg_limited_role] = test_set_arg_limited_role[start: end]

    trainVariables = {}
    trainVariables[x_content_arg] = train_set_content_arg[start: end]
    trainVariables[x_arg] = train_set_arg[start: end]
    trainVariables[x_role] = train_set_role
    trainVariables[x_role_structure] = train_set_role_structure
    trainVariables[x_relation_binary_arg] = train_set_relation_binary_arg[start: end]
    trainVariables[arg_y] = train_set_arg_y[start: end]
    trainVariables[arg_y_vector] = train_set_arg_y_vector[start: end]
    trainVariables[x_train_roles] = train_roles
    trainVariables[x_pos_neg_flag_arg] = train_set_posneg_arg[start: end]
    trainVariables[arg_limited_role] = train_set_arg_limited_role[start: end]

    print ("length of train variables ", trainVariables)

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by SGD Since this model has many parameters,
    # it would be tedious to manually create an update rule for each model parameter. We thus create the updates
    # list by automatically looping over all (params[i],grads[i]) pairs.
    updates = []
    rho = 0.9
    epsilon = 1e-6
    # for param_i, grad_i in zip(params, grads):
    #     updates.append((param_i, param_i - lr * grad_i))
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))

    test_model_confidence = theano.function([index], layer3.results(arg_y), on_unused_input='ignore',
                                            givens=testVariables)
    eval_model_confidence = theano.function([index], layer3.results(arg_y), on_unused_input='ignore',
                                            givens=validVariables)
    train_model = theano.function([index, lr, lambd1, lambd2], [cost, layer3.loss], updates=updates,
                                  on_unused_input='ignore', givens=trainVariables)

    time2 = time.time()
    print ("time for building the model: ", time2 - time1)

    # train the model
    print ('start training ... ')
    time1 = time.time()

    validation_frequency = num_examples_per_epoch / batch_size  # validate after each epoch
    best_params = []
    best_fscore = -1
    last_fscore = -1
    best_fscore_m1 = -1
    best_iter = 0
    best_fscoreEval = -1
    best_fscore_m1Eval = -1
    best_iterEval = 0

    start_time = time.clock()

    epoch = 0
    done_looping = False

    maxNoImprovement = 5
    noImprovement = 0

    while (epoch < n_epochs) and (not done_looping):
        print ('epoch = ', epoch)
        epoch = epoch + 1

        this_n_train_batches = num_examples_per_epoch / batch_size

        for minibatch_index in xrange(this_n_train_batches):

            iter = (epoch - 1) * this_n_train_batches + minibatch_index

            if iter % 100 == 0:
                print ('training @ iter = ', iter)

            cost_ij, loss = train_model(minibatch_index, learning_rate, l1, l2)

            print ("cost: ", cost_ij)
            print ("loss1:   ", loss)

            if (iter + 1) % validation_frequency == 0:

                # test data
                confidence_eval = [test_model_confidence(i) for i in xrange(n_test_batches)]
                confidence_list_eval = []
                for r in range(0, len(confidence_eval)):
                    for r1 in range(0, batch_size):
                        hypo_result_eval = confidence_eval[r][0].item(r1)
                        confidence_list_eval.append(hypo_result_eval)

                y_pred_eval = confidence_list_eval
                y_true_eval = role_index_test_matrix[:n_test_batches * batch_size]
                y_true_eval_2 = []
                for i in range(len(y_true_eval)):
                    y_true_eval_2.append(int(y_true_eval[i]))

                labels1 = [13, 14, 15, 16, 17]
                this_fscore_eval = f1_score(y_true_eval_2, y_pred_eval, labels=labels1, average='micro')
                this_fscore_macro_eval = f1_score(y_true_eval_2, y_pred_eval, labels=labels1, average='macro')
                print( 'EVAL: ***   epoch %i, best_validation %f, best_validation_m1 %f, learning_rate %f, '
                       'minibatch %i/%i, validation fscore %f %%' % ( epoch, best_fscoreEval * 100.,
                                                                      best_fscore_m1Eval * 100, learning_rate,
                                                                      minibatch_index + 1, this_n_train_batches,
                                                                      this_fscore_eval * 100.))

                if this_fscore_eval > best_fscoreEval:
                    best_fscoreEval = this_fscore_eval
                    best_fscore_m1Eval = this_fscore_macro_eval
                    best_iterEval = iter

                # dev data
                confidence = [eval_model_confidence(i) for i in xrange(n_valid_batches)]

                confidence_list = []
                for r in range(0, len(confidence)):
                    for r1 in range(0, batch_size):
                        hypo_result = confidence[r][0].item(r1)
                        confidence_list.append(hypo_result)

                y_pred = confidence_list
                y_true = role_index_dev_matrix[:n_valid_batches * batch_size]
                y_true_2 = []
                for i in range(len(y_true)):
                    y_true_2.append(int(y_true[i]))

                labels = []
                for i in range(1, role_size):
                    labels.append(i)
                this_fscore = f1_score(y_true_2, y_pred, labels=labels, average='micro')
                this_fscore_macro = f1_score(y_true_2, y_pred, labels=labels, average='macro')
                print( 'epoch %i, best_validation %f, best_validation_m1 %f, learning_rate %f, minibatch %i/%i, '
                       'validation fscore %f %%' % ( epoch, best_fscore * 100., best_fscore_m1 * 100, learning_rate,
                                                     minibatch_index + 1, this_n_train_batches, this_fscore * 100.))

                # if we got the best validation score until now
                if this_fscore > best_fscore:
                    best_fscore = this_fscore
                    best_fscore_m1 = this_fscore_macro
                    best_iter = iter

                    best_params = []
                    for p in param_list:
                        p_param = []
                        for part in p:
                            p_param.append(part.get_value(borrow=False))
                        best_params.append(p_param)
                    noImprovement = 0
                else:
                    if this_fscore > last_fscore:
                        noImprovement -= 1
                        noImprovement = max(noImprovement, 0)
                    else:
                        noImprovement += 1
                        updatestep = minibatch_index + this_n_train_batches * (epoch - 1)
                        if newbob:  # learning rate schedule depending on dev result
                            learning_rate /= 1.2
                            print ("reducing learning rate to ", learning_rate)
                last_fscore = this_fscore
            if newbob:  # learning rate schedule depending on dev result
                if noImprovement > maxNoImprovement or learning_rate < 0.0000001:
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
          (best_fscore * 100., arg_context_size, nkerns[0], filter_size[1], hidden_units, best_iter + 1))

    time2 = time.time()
    print ("time for training: ", time2 - time1)

    print('Saving net.')
    save_file = open(arg_network_file, 'wb')
    for p in best_params:
        for p_part in p:
            cPickle.dump(p_part, save_file, -1)
    save_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='ACL-Zero-Shot-Data/classification/aceEventStructure.train.ace.format.transfer.neg.arg.10.tx',
                        help='path for train file')
    parser.add_argument('--dev', type=str, default='ACL-Zero-Shot-Data/classification/aceEventStructure.test.ace.format.transfer.neg.arg.10.tx',
                        help='path for dev file')
    parser.add_argument('--test', type=str, default='ACL-Zero-Shot-Data/classification/aceEventStructure.test.ace.format.transfer.neg.arg.10.tx',
                        help='path for test file')
    parser.add_argument('--model_path', type=str, default='ACL-Zero-Shot-Data/model/framenet.arg.model',
                        help='path for saving trained models')
    parser.add_argument('--embedding_path', type=str,
                        default='ACL-Zero-Shot-Data/embedding/wsd.model.ace.filter.txt',
                        help='path for pretrained word embedding')
    parser.add_argument('--ontology_path', type=str, default='ACL-Zero-Shot-Data/frame-ontology/event.ontology.new.txt',
                        help='path for predefined ontology')
    parser.add_argument('--norm_ontology_path', type=str, default='ACL-Zero-Shot-Data/frame-ontology/event.ontology.normalize.new.txt',
                        help='path for predefined ontology')
    parser.add_argument('--arg_path_file', type=str, default='ACL-Zero-Shot-Data/frame-ontology/event.ontology.args.merge.all.txt',
                        help='path for arg paths')
    parser.add_argument('--arg_path_file_universal', type=str,
                        default='ACL-Zero-Shot-Data/frame-ontology/event.ontology.args.universal.txt',
                        help='path for arg paths')
    parser.add_argument('--trigger_role_matrix', type=str, default='ACL-Zero-Shot-Data/frame-ontology/event.ontology.trigger.arg.matrix.new.txt',
                        help='path for trigger role matrix file')
    parser.add_argument('--seen_args', type=str, default='ACL-Zero-Shot-Data/flags/train.arg.10',
                        help='tag file for seen event args')
    parser.add_argument('--relation_path', type=str, default='ACL-Zero-Shot-Data/amr/amrRelations.txt', help='amr relations')

    parser.add_argument('--arg_context_size', type=int, default=5, help='number of mention tuples')
    parser.add_argument('--role_context_size', type=int, default=5, help='number of type tuples')
    parser.add_argument('--hidden_units', type=int, default=200, help='size of hidden units')
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--embedding_size', type=int, default=200)
    parser.add_argument('--num_epochs', type=int, default=2)

    args = parser.parse_args()
    print(args)
    main(args)