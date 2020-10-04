import random
import string
import numpy

__author__ = 'wilbur'


# read type index, e.g., Attack -- 14
def read_type_index(file_name):
    type_2_index = {}
    index_2_type = {}

    f = open(file_name, 'r')
    for line in f:
        parts = line.strip().split('\t')
        type = parts[2]
        index = parts[0]
        type_2_index[type] = index
        index_2_type[index] = type
    return type_2_index, index_2_type


# read relation index, e.g., ARG0 -- 1
def read_relation_index(file_name):
    rel_2_index = {}
    index_2_rel = {}

    f = open(file_name, 'r')
    for line in f:
        parts = line.strip().split('\t')
        rel = parts[1]
        index = parts[0]
        rel_2_index[rel] = index
        index_2_rel[index] = rel
    return rel_2_index, index_2_rel


# load train data to lists
def load_training_data(file_name):
    f = open(file_name, 'r')
    doc_id_list = []
    type_list = []
    trigger_list = []
    left_word_list = []
    relation_list = []
    right_word_list = []
    for line in f:
        line = line.strip()
        parts = line.split(' :: ')
        doc_id = parts[0]
        type = parts[1]
        trigger = parts[2]

        doc_id_list.append(doc_id)
        type_list.append(type)
        trigger_list.append(trigger)

        left_words = []
        right_words = []
        relations = []

        if len(parts) > 3:
            structure = parts[3]

            structure_parts = structure.strip().split('	')

            for unit in structure_parts:
                unit_parts = unit.strip().split('##')
                word1 = unit_parts[0]
                relation = unit_parts[1]
                word2 = unit_parts[2]

                left_words.append(word1)
                relations.append(relation)
                right_words.append(word2)

        left_word_list.append(left_words)
        relation_list.append(relations)
        right_word_list.append(right_words)
    return doc_id_list, type_list, trigger_list, left_word_list, relation_list, right_word_list


# loading the whole word2vec file
def load_word_vec(file_name):
    word_vectors = {}
    vector_size = 0
    vocab = []
    f = open(file_name, 'r')
    for line in f:
        parts = line.split()
        if len(parts) > 2:
            word = string.lower(parts[0])
            parts.pop(0)
            word_vectors[word] = parts
            vector_size = len(parts)
            vocab.append(word)
    f.close()
    return [word_vectors, vector_size]


# load all type labels
def load_types(type_file):
    all_types = {}
    all_type_structures = {}
    f = open(type_file, 'r')
    for line in f:
        parts = line.strip().split("\t")
        index = parts[0]
        trigger = parts[2]
        all_types[index] = trigger
        structures = []
        for j in range(3, len(parts)):
            s = trigger + "\t" + parts[j]
            structures.append(s)
        all_type_structures[index] = structures
    return all_types, all_type_structures


# load all type labels
def load_types_1(type_file):
    all_types = {}
    all_type_structures = {}
    f = open(type_file, 'r')
    for line in f:
        parts = line.strip().split("\t")
        index = parts[0]
        trigger = parts[2]
        all_types[index] = trigger
        structures = []
        for j in range(3, len(parts)):
            s = trigger + "\t" + parts[j]
            structures.append(s)
            all_type_structures[index] = structures
    return all_types, all_type_structures


# get the matrix for all input parts
def get_input_context_matrix(left_word, right_word, word_vectors, representation_size, context_size):
    matrix = numpy.zeros(shape=(representation_size * 2, context_size))
    i = 0

    while i < context_size:
        word1 = left_word[i].lower()
        word2 = right_word[i].lower()

        # current word
        if not word1 in word_vectors:
            if str(word1).find('-') == -1:
                word1 = "<unk>"
            else:
                idx = word1.find('-')
                word1 = word1[:idx]
                if not word1 in word_vectors:
                    word1 = "<unk>"
        curVector1 = word_vectors[word1]
        for j in range(0, representation_size):
            if j > len(curVector1):
                print "ERROR: mismatch in word vector lengths: " + str(len(curVector1)) + " vs " + representation_size
                exit()
            elem = float(curVector1[j])
            matrix[j, i] = elem

        if not word2 in word_vectors:
            if str(word2).find('-') == -1:
                word2 = "<unk>"
            else:
                idx = word2.find('-')
                word2 = word2[:idx]
                if not word2 in word_vectors:
                    word2 = "<unk>"
        cur_vector_2 = word_vectors[word2]
        for j in range(0, representation_size):
            if j > len(curVector1):
                print "ERROR: mismatch in word vector lengths: " + str(len(cur_vector_2)) + " vs " + representation_size
                exit()
            elem = float(cur_vector_2[j])
            matrix[j + representation_size, i] = elem
        i += 1
    return matrix


# get the matrix for all triggers
def get_input_trigger_matrix(trigger, word_vectors, representation_size):
    matrix = numpy.zeros(shape=(representation_size, 1))

    trigger = trigger.lower()

    if not trigger in word_vectors:
        if str(trigger).find('-') == -1:
            trigger = "<unk>"
        else:
            idx = trigger.find('-')
            trigger = trigger[:idx]
            if not trigger in word_vectors:
                trigger = "<unk>"
    cur_vector_1 = word_vectors[trigger]
    for j in range(0, representation_size):
        if j > len(cur_vector_1):
            print "ERROR: mismatch in word vector lengths: " + str(len(cur_vector_1)) + " vs " + representation_size
            exit()
        elem = float(cur_vector_1[j])
        matrix[j, 0] = elem
    return matrix


# load all initial random relation vectors
def load_relation_vectors(relation_vec_file, representation_size):
    matrix = numpy.zeros(shape=(representation_size, 1))
    f = open(relation_vec_file, 'r')
    index = 0
    for line in f:
        parts = line.strip().split()
        for j in range(0, len(parts)):
            elem = float(parts[j])
            matrix[j, index] = elem
        index += 1
    return matrix


# generate random relation vectors
def random_init_rel_vec(relations_file, size):
    f = open(relations_file, 'r')
    index = 0
    for line in f:
        index += 1

    rel_vec_matrix = numpy.empty(shape=(index, size * size))

    for i in range(0, index):
        for j in range(0, size * size):
            d1 = random.random() - 0.5
            rel_vec_matrix[i, j] = d1
    return rel_vec_matrix


# generate random relation vectors
def random_init_rel_vec_factor(relations_file, size):
    f = open(relations_file, 'r')
    index = 0
    for line in f:
        index += 1

    rel_vec_matrix = numpy.empty(shape=(index, size))

    for i in range(0, index):
        for j in range(0, size):
            d1 = random.random() - 0.5
            rel_vec_matrix[i, j] = d1
    return rel_vec_matrix


# get the relation binary vectors for each relation
def get_binary_relation_matrix(relation, relation_size, context_size, rel_2_index):
    matrix = numpy.zeros(shape=(relation_size, context_size))
    i = 0
    while i < context_size:
        rel = relation[i]
        idx = int(rel_2_index[rel])
        for j in range(0, relation_size):
            if idx == j:
                matrix[j, i] = 1
            else:
                matrix[j, i] = 0
        i += 1
    return matrix


# get all input matrix for nn
def input_matrix(type_list, trigger_list, left_word_list, relation_list, right_word_list, representation_size,
                 context_size, relation_size, type_file, word_2_vec, rel_2_index):
    num_samples = len(left_word_list)

    print "processing" + str(num_samples) + " examples per epoch"

    type_2_index, index_2_type = read_type_index(type_file)
    type_size = len(type_2_index)

    representation_size_1 = representation_size * 2

    input_context_matrix = numpy.empty(shape=(num_samples, representation_size_1 * context_size))
    input_trigger_matrix = numpy.empty(shape=(num_samples, representation_size * 1))
    result_index_matrix = []
    result_vector_matrix = numpy.empty(shape=(num_samples, type_size * 1))
    relation_binary_matrix = numpy.empty(shape=(num_samples, relation_size * context_size))

    for sample in range(0, num_samples):
        left_word = left_word_list[sample]
        relation = relation_list[sample]
        right_word = right_word_list[sample]

        trigger = trigger_list[sample]
        type = type_list[sample]

        matrix = numpy.zeros(shape=(type_size, 1))
        for m in range(0, type_size):
            matrix[m, 0] = 0
        type = type.lower()
        if type_2_index.has_key(type):
            type_index = type_2_index[type]
            result_index_matrix.append(type_index)

            type_index_1 = int(type_index) - 1
            matrix[type_index_1, 0] = 1
        else:
            result_index_matrix.append(0)

        while len(left_word) < context_size:
            left_word.append("PADDING")
            relation.append(":other")
            right_word.append("PADDING")

        matrix_word_pair = \
            get_input_context_matrix(left_word, right_word, word_2_vec, representation_size, context_size)
        matrix_word_pair = numpy.reshape(matrix_word_pair, representation_size_1 * context_size)
        input_context_matrix[sample, :] = matrix_word_pair

        matrix_trigger = get_input_trigger_matrix(trigger, word_2_vec, representation_size)
        matrix_trigger = numpy.reshape(matrix_trigger, representation_size * 1)
        input_trigger_matrix[sample, :] = matrix_trigger

        matrix_relation = get_binary_relation_matrix(relation, relation_size, context_size, rel_2_index)
        matrix_relation = numpy.reshape(matrix_relation, relation_size * context_size)
        relation_binary_matrix[sample, :] = matrix_relation

        matrix = numpy.reshape(matrix, type_size)
        result_vector_matrix[sample, :] = matrix

    return result_index_matrix, result_vector_matrix, input_context_matrix, input_trigger_matrix, relation_binary_matrix


def get_train_type_flag(train_type_flag):
    f = open(train_type_flag, 'r')
    map = {}
    for line in f:
        parts = line.strip().split("\t")
        type = parts[0]
        map[type] = type
    return map


# get all input matrix for nn, add neg examples
def input_matrix_1(type_list, trigger_list, left_word_list, relation_list, right_word_list, representation_size,
                   context_size, relation_size, type_file, word_2_vec, rel_2_index, train_type_flag):
    num_samples = len(left_word_list)

    train_type_flags = get_train_type_flag(train_type_flag)

    print "processing" + str(num_samples) + " examples per epoch"

    type_2_index, index_2_type = read_type_index(type_file)
    type_size = len(type_2_index)

    representation_size_1 = representation_size * 2

    input_context_matrix = numpy.empty(shape=(num_samples, representation_size_1 * context_size))
    input_trigger_matrix = numpy.empty(shape=(num_samples, representation_size * 1))
    result_index_matrix = []
    result_vector_matrix = numpy.empty(shape=(num_samples, type_size * 1))
    relation_binary_matrix = numpy.empty(shape=(num_samples, relation_size * context_size))
    pos_neg_matrix = numpy.empty(shape=(num_samples, 1))

    for sample in range(0, num_samples):
        left_word = left_word_list[sample]
        relation = relation_list[sample]
        right_word = right_word_list[sample]

        trigger = trigger_list[sample]
        type = type_list[sample]

        matrix = numpy.zeros(shape=(type_size, 1))
        for m in range(0, type_size):
            matrix[m, 0] = 0

        if train_type_flags.has_key(type):

            pos_neg_matrix[sample, 0] = 1

            type = type.lower()
            type_index = type_2_index[type]
            result_index_matrix.append(type_index)

            type_index_1 = int(type_index) - 1
            matrix[type_index_1, 0] = 1
        else:
            result_index_matrix.append(0)
            pos_neg_matrix[sample, 0] = 0

        ## each relation will be annotated as a binary vector amond all types
        while len(left_word) < context_size:
            left_word.append("PADDING")
            relation.append(":other")
            right_word.append("PADDING")

        matrix_word_pair = get_input_context_matrix(left_word, right_word, word_2_vec, representation_size,
                                                    context_size)
        matrix_word_pair = numpy.reshape(matrix_word_pair, representation_size_1 * context_size)
        input_context_matrix[sample, :] = matrix_word_pair

        matrix_trigger = get_input_trigger_matrix(trigger, word_2_vec, representation_size)
        matrix_trigger = numpy.reshape(matrix_trigger, representation_size * 1)
        input_trigger_matrix[sample, :] = matrix_trigger

        matrix_relation = get_binary_relation_matrix(relation, relation_size, context_size, rel_2_index)
        matrix_relation = numpy.reshape(matrix_relation, relation_size * context_size)
        relation_binary_matrix[sample, :] = matrix_relation

        matrix = numpy.reshape(matrix, type_size)
        result_vector_matrix[sample, :] = matrix

    return result_index_matrix, result_vector_matrix, input_context_matrix, input_trigger_matrix, \
           relation_binary_matrix, pos_neg_matrix


# get all input matrix for nn, add neg examples
def input_matrix_1_test(type_list, trigger_list, left_word_list, relation_list, right_word_list, representation_size,
                        context_size, relation_size, type_file, word_2_vec, rel_2_index, train_type_flag):
    num_samples = len(left_word_list)

    train_type_flags = get_train_type_flag(train_type_flag)

    print "processing" + str(num_samples) + " examples per epoch"

    type_2_index, index_2_type = read_type_index(type_file)
    type_size = len(type_2_index)

    representation_size_1 = representation_size * 2

    intput_context_matrix = numpy.empty(shape=(num_samples, representation_size_1 * context_size))
    input_trigger_matrix = numpy.empty(shape=(num_samples, representation_size * 1))
    result_index_matrix = []
    result_vector_matrix = numpy.empty(shape=(num_samples, type_size * 1))
    relation_binary_matrix = numpy.empty(shape=(num_samples, relation_size * context_size))
    pos_neg_matrix = numpy.empty(shape=(num_samples, 1))

    for sample in range(0, num_samples):
        left_word = left_word_list[sample]
        relation = relation_list[sample]
        right_word = right_word_list[sample]

        trigger = trigger_list[sample]
        type = type_list[sample]

        matrix = numpy.zeros(shape=(type_size, 1))
        for m in range(0, type_size):
            matrix[m, 0] = 0

        pos_neg_matrix[sample, 0] = 1

        type = type.lower()
        type_index = type_2_index[type]
        result_index_matrix.append(type_index)

        type_index_1 = int(type_index) - 1
        matrix[type_index_1, 0] = 1

        # each relation will be annotated as a binary vector amond all types
        while len(left_word) < context_size:
            left_word.append("PADDING")
            relation.append(":other")
            right_word.append("PADDING")

        matrix_word_pair = get_input_context_matrix(left_word, right_word, word_2_vec, representation_size,
                                                    context_size)
        matrix_word_pair = numpy.reshape(matrix_word_pair, representation_size_1 * context_size)
        intput_context_matrix[sample, :] = matrix_word_pair

        matrix_trigger = get_input_trigger_matrix(trigger, word_2_vec, representation_size)
        matrix_trigger = numpy.reshape(matrix_trigger, representation_size * 1)
        input_trigger_matrix[sample, :] = matrix_trigger

        matrix_relation = get_binary_relation_matrix(relation, relation_size, context_size, rel_2_index)
        matrix_relation = numpy.reshape(matrix_relation, relation_size * context_size)
        relation_binary_matrix[sample, :] = matrix_relation

        matrix = numpy.reshape(matrix, type_size)
        result_vector_matrix[sample, :] = matrix

    return result_index_matrix, result_vector_matrix, intput_context_matrix, input_trigger_matrix, \
           relation_binary_matrix, pos_neg_matrix


def get_word_vec(phrase, word_vectors, vector_size):
    vec = []
    parts = str(phrase).lower().strip().split(" ")
    for i in range(0, vector_size):
        vec.append(0)

    count = 0
    for i in range(0, len(parts)):
        word = parts[i]
        if word != "<empty>":
            if word in word_vectors:
                count += 1.0
                cur_vector = word_vectors[word]
                for j in range(0, vector_size):
                    elem = float(cur_vector[j])
                    vec[j] += elem

    if count > 0:
        for j in range(0, vector_size):
            vec[j] = vec[j] / count
    else:
        for j in range(0, vector_size):
            elem = random.random()
            vec[j] = float(elem)
            word_vectors[word] = vec
    return vec


def type_matrix(all_type_list, all_type_structure_list, word_2_vec_file, context):
    word_vectors, vector_size = load_word_vec(word_2_vec_file)
    num_of_types = len(all_type_list)
    rep_size = vector_size * 2
    input_type_structure_matrix = numpy.empty(shape=(num_of_types, rep_size * context))
    input_type_matrix = numpy.empty(shape=(num_of_types, vector_size))
    for i in range(0, num_of_types):
        type = all_type_list[str(i + 1)]
        type_vec = get_word_vec(type, word_vectors, vector_size)
        for j in range(0, vector_size):
            input_type_matrix[i, j] = type_vec[j]

        type_structures = all_type_structure_list[str(i + 1)]

        while len(type_structures) < context:
            type_structures.append("PADDING" + "\t" + "PADDING")

        for j in range(0, context):
            structure = type_structures[j]
            parts = structure.strip().split("\t")
            vec1 = get_word_vec(parts[0], word_vectors, vector_size)
            vec2 = get_word_vec(parts[1], word_vectors, vector_size)

            for m in range(0, vector_size):
                input_type_structure_matrix[i, j * rep_size + m] = vec1[m]
                input_type_structure_matrix[i, j * rep_size + vector_size + m] = vec2[m]

    return input_type_matrix, input_type_structure_matrix


def read_arg_paths(file_name):
    f = open(file_name, 'r')
    index_2_path = {}
    index_2_arg = {}
    type_2_arg_2_index = {}

    for line in f:
        parts = line.strip().split('\t')
        index = parts[0]
        type = parts[1]
        nor_type = parts[2]
        arg = parts[3]
        normalize_arg = parts[4]
        path = type + "\t" + normalize_arg

        index_2_path[index] = path
        index_2_arg[index] = normalize_arg
        if type_2_arg_2_index.has_key(nor_type):
            arg_2_index = type_2_arg_2_index[nor_type]
            arg_2_index[arg] = index
            type_2_arg_2_index[nor_type] = arg_2_index
        else:
            arg_2_index = {arg: index}
            type_2_arg_2_index[nor_type] = arg_2_index
    return index_2_path, index_2_arg, type_2_arg_2_index


def load_arg_data(file_name):
    f = open(file_name, 'r')

    trigger_list = []
    trigger_type_list = []
    arg_list = []
    arg_path_left_list = []
    arg_path_rel_list = []
    arg_path_right_list = []
    arg_role_list = []

    for line in f:
        parts = line.strip().split(" :: ")
        # print line
        if len(parts) < 7:
            print "<7  " + str(line)
        trigger = parts[3]
        trigger_type = parts[2]
        arg = parts[6]
        arg_path = parts[7]
        arg_role = parts[5]

        trigger_list.append(trigger)
        trigger_type_list.append(trigger_type)
        arg_list.append(arg)
        arg_role_list.append(arg_role)

        path_left_list = []
        path_rel_list = []
        path_right_list = []

        paths = arg_path.strip().split('\t')

        for structure in paths:
            structure_array = structure.strip().split('##')
            left_word = structure_array[0]
            rel = structure_array[1]
            right_word = structure_array[2]

            path_left_list.append(left_word)
            path_rel_list.append(rel)
            path_right_list.append(right_word)

        arg_path_left_list.append(path_left_list)
        arg_path_rel_list.append(path_rel_list)
        arg_path_right_list.append(path_right_list)

    return trigger_list, trigger_type_list, arg_list, arg_path_left_list, arg_path_rel_list, arg_path_right_list, \
           arg_role_list


def get_types_for_train(train_type_flag, type_label_file):
    type_2_index, index_2_type = read_type_index(type_label_file)
    types = []
    for i in range(len(type_2_index)):
        types.append(0)

    f = open(train_type_flag, 'r')
    for line in f:
        parts = line.strip().split("\t")
        type = parts[0].lower()
        index = type_2_index[type]
        index1 = int(index) - 1
        types[index1] = 1

    return types


def input_arg_matrix(trigger_list, trigger_type_list, arg_list, arg_path_left_list, arg_path_rel_list,
                     arg_path_right_list, arg_role_list, word_2_vec, role_list, trigger_role_2_index, vector_size,
                     context_size, relation_size, rel_2_index, roles_flag, trigger_role_matrix, label_file):
    num_samples = len(arg_list)

    type_2_index, index_2_type = read_type_index(label_file)

    print "processing" + str(num_samples) + " examples per epoch"

    # index2Path, index2Arg, type2Arg2Index = readArgPaths(argPathFile)
    arg_size = len(role_list)

    vector_size_1 = vector_size * 2

    pos_neg_matrix = numpy.empty(shape=(num_samples, 1))

    intput_arg_context_matrix = numpy.empty(shape=(num_samples, vector_size_1 * context_size))

    input_arg_matrix = numpy.empty(shape=(num_samples, vector_size * 1))
    role_index_matrix = []
    role_vector_matrix = numpy.empty(shape=(num_samples, arg_size * 1))
    relation_binary_matrix = numpy.empty(shape=(num_samples, relation_size * context_size))

    limited_roles_train_matrix = numpy.empty(shape=(num_samples, arg_size))

    for sample in range(0, num_samples):
        left_word = arg_path_left_list[sample]
        relation = arg_path_rel_list[sample]
        right_word = arg_path_right_list[sample]

        arg = arg_list[sample]
        role = arg_role_list[sample]
        trigger_type = trigger_type_list[sample]

        trigger_type_lower = trigger_type.lower()

        trigger_index = int(type_2_index[trigger_type_lower]) - 1
        limited_roles = trigger_role_matrix[trigger_index, :]
        limited_roles_train_matrix[sample, :] = limited_roles

        role_lower = role.lower()
        role_index = 0
        key1 = role_lower
        key2 = trigger_type_lower + "##" + role_lower
        if trigger_role_2_index.has_key(key1):
            role_index = trigger_role_2_index[key1]
        if trigger_role_2_index.has_key(key2):
            role_index = trigger_role_2_index[key2]
        role_index_tmp = int(role_index)

        role_index_matrix.append(role_index_tmp)

        matrix = numpy.zeros(shape=(arg_size, 1))
        for m in range(0, arg_size):
            matrix[m, 0] = 0
        role_index_1 = int(role_index) - 1
        matrix[role_index_1, 0] = 1

        pos_neg_matrix[sample, 0] = roles_flag[role_index_1]

        # each relation will be annotated as a binary vector amond all types
        while len(left_word) < context_size:
            left_word.append("PADDING")
            relation.append(":other")
            right_word.append("PADDING")

        matrix_word_pair = get_input_context_matrix(left_word, right_word, word_2_vec, vector_size, context_size)
        matrix_word_pair = numpy.reshape(matrix_word_pair, vector_size_1 * context_size)
        intput_arg_context_matrix[sample, :] = matrix_word_pair

        matrix_arg = get_input_trigger_matrix(arg, word_2_vec, vector_size)
        matrix_arg = numpy.reshape(matrix_arg, vector_size * 1)
        input_arg_matrix[sample, :] = matrix_arg

        matrix_relation = get_binary_relation_matrix(relation, relation_size, context_size, rel_2_index)
        matrix_relation = numpy.reshape(matrix_relation, relation_size * context_size)
        relation_binary_matrix[sample, :] = matrix_relation

        matrix = numpy.reshape(matrix, arg_size)
        role_vector_matrix[sample, :] = matrix

    return role_index_matrix, role_vector_matrix, intput_arg_context_matrix, input_arg_matrix, relation_binary_matrix, \
           pos_neg_matrix, limited_roles_train_matrix


def input_arg_matrix_test(trigger_list, trigger_type_list, arg_list, arg_path_left_list, arg_path_rel_list,
                          arg_path_right_list, arg_role_list, word_2_vec, role_list, trigger_role_2_index, vector_size,
                          context_size, relation_size, rel_2_index, roles_flag, trigger_role_matrix, label_file):
    num_samples = len(arg_list)
    type_2_index, index_2_type = read_type_index(label_file)
    print "processing" + str(num_samples) + " examples per epoch"

    arg_size = len(role_list)
    vector_size_1 = vector_size * 2
    pos_neg_matrix = numpy.empty(shape=(num_samples, 1))
    intput_arg_context_matrix = numpy.empty(shape=(num_samples, vector_size_1 * context_size))

    input_arg_matrix = numpy.empty(shape=(num_samples, vector_size * 1))
    role_index_matrix = []
    role_vector_matrix = numpy.empty(shape=(num_samples, arg_size * 1))
    relation_binary_matrix = numpy.empty(shape=(num_samples, relation_size * context_size))
    limited_roles_train_matrix = numpy.empty(shape=(num_samples, arg_size))

    for sample in range(0, num_samples):
        left_word = arg_path_left_list[sample]
        relation = arg_path_rel_list[sample]
        right_word = arg_path_right_list[sample]

        arg = arg_list[sample]
        role = arg_role_list[sample]
        trigger_type = trigger_type_list[sample]

        trigger_type_lower = trigger_type.lower()

        trigger_index = int(type_2_index[trigger_type_lower]) - 1
        limited_roles = trigger_role_matrix[trigger_index, :]
        limited_roles_train_matrix[sample, :] = limited_roles

        role_lower = role.lower()
        role_index = 0
        key1 = role_lower
        key2 = trigger_type_lower + "##" + role_lower
        if trigger_role_2_index.has_key(key1):
            role_index = trigger_role_2_index[key1]
        if trigger_role_2_index.has_key(key2):
            role_index = trigger_role_2_index[key2]
        role_index_tmp = int(role_index)
        role_index_matrix.append(role_index_tmp)

        matrix = numpy.zeros(shape=(arg_size, 1))
        for m in range(0, arg_size):
            matrix[m, 0] = 0
        role_index_1 = int(role_index) - 1
        matrix[role_index_1, 0] = 1

        pos_neg_matrix[sample, 0] = roles_flag[role_index_1]

        # each relation will be annotated as a binary vector amond all types
        while len(left_word) < context_size:
            left_word.append("PADDING")
            relation.append(":other")
            right_word.append("PADDING")

        matrix_word_pair = get_input_context_matrix(left_word, right_word, word_2_vec, vector_size, context_size)
        matrix_word_pair = numpy.reshape(matrix_word_pair, vector_size_1 * context_size)
        intput_arg_context_matrix[sample, :] = matrix_word_pair

        matrix_arg = get_input_trigger_matrix(arg, word_2_vec, vector_size)
        matrix_arg = numpy.reshape(matrix_arg, vector_size * 1)
        input_arg_matrix[sample, :] = matrix_arg

        matrix_relation = get_binary_relation_matrix(relation, relation_size, context_size, rel_2_index)
        matrix_relation = numpy.reshape(matrix_relation, relation_size * context_size)
        relation_binary_matrix[sample, :] = matrix_relation

        matrix = numpy.reshape(matrix, arg_size)
        role_vector_matrix[sample, :] = matrix

    return role_index_matrix, role_vector_matrix, intput_arg_context_matrix, input_arg_matrix, relation_binary_matrix, \
           pos_neg_matrix, limited_roles_train_matrix


def load_roles(arg_path_file_specific, arg_path_file_generic):
    f1 = open(arg_path_file_specific, 'r')
    f2 = open(arg_path_file_generic, 'r')
    all_arg_role_list = {}
    index_2_role = {}
    trigger_role_2_index = {}
    index_2_norm_role = {}
    trigger_norm_role_2_index = {}

    for line in f1:
        parts = line.strip().split('\t')
        index = parts[0]
        ori_type = parts[1]
        norm_type = parts[2]
        ori_role = parts[3]
        norm_role = parts[4]

        all_arg_role_list[index] = norm_role
        index_2_role[index] = ori_role
        trigger_role = str(ori_type) + "##" + str(ori_role)
        trigger_role_2_index[trigger_role] = index

        index_2_norm_role[index] = norm_role
        trigger_norm_role = str(ori_type) + "##" + str(norm_role)
        trigger_norm_role_2_index[trigger_norm_role] = index

    f1.close()

    for line in f2:
        parts = line.strip().split('\t')
        index = parts[0]
        ori_role = parts[1]
        norm_role = parts[1]

        all_arg_role_list[index] = norm_role
        index_2_role[index] = ori_role
        trigger_role_2_index[ori_role] = index

        index_2_norm_role[index] = norm_role
        trigger_norm_role_2_index[norm_role] = index
    f2.close()

    return all_arg_role_list, index_2_role, trigger_role_2_index, index_2_norm_role, trigger_norm_role_2_index


def load_roles_1(arg_path_file_merge):
    f1 = open(arg_path_file_merge, 'r')
    all_arg_role_list = {}
    all_trigger_role_structure = {}
    index_2_role = {}
    trigger_role_2_index = {}
    index_2_norm_role = {}
    trigger_norm_role_2_index = {}

    for line in f1:
        parts = line.strip().split('\t')
        index = parts[0]
        ori_type = parts[1]
        norm_type = parts[2]
        ori_role = parts[3]
        norm_role = parts[4]

        all_arg_role_list[index] = norm_role
        index_2_role[index] = ori_role
        trigger_role = str(ori_type) + "##" + str(ori_role)
        trigger_role_2_index[trigger_role] = index

        index_2_norm_role[index] = norm_role
        trigger_norm_role = str(ori_type) + "##" + str(norm_role)
        trigger_norm_role_2_index[trigger_norm_role] = index

        structure = []
        structure.append(norm_type + "\t" + norm_role)
        all_trigger_role_structure[index] = structure

    f1.close()

    return all_arg_role_list, all_trigger_role_structure, index_2_role, trigger_role_2_index, index_2_norm_role, \
           trigger_norm_role_2_index


def get_trigger_arg_matrix(trigger_role_matrix_file, type_size, role_size):
    trigger_role_matrix = numpy.zeros(shape=(type_size, role_size))
    f = open(trigger_role_matrix_file, 'r')
    for line in f:
        parts = line.strip().split('\t')
        trigger_index = int(parts[0]) - 1;

        units = parts[1].strip().split(' ')
        for i in range(0, len(units)):
            arg_index = int(units[i]) - 1
            trigger_role_matrix[trigger_index, arg_index] = 1
    return trigger_role_matrix


def role_matrix(all_arg_role_list, word_vector_file_type):
    word_vectors, vector_size = load_word_vec(word_vector_file_type)
    num_of_roles = len(all_arg_role_list)
    rep_size = vector_size * 2
    input_role_matrix = numpy.empty(shape=(num_of_roles, vector_size))
    for i in range(0, num_of_roles):
        role = all_arg_role_list[str(i + 1)]
        role_vec = get_word_vec(role, word_vectors, vector_size)
        for j in range(0, vector_size):
            input_role_matrix[i, j] = role_vec[j]

    return input_role_matrix


def role_matrix_1(all_arg_role_list, all_type_role_structure, word_vector_file_type, role_structure_context_size):
    word_vectors, vector_size = load_word_vec(word_vector_file_type)
    num_of_roles = len(all_arg_role_list)
    rep_size = vector_size * 2
    input_role_structure_matrix = numpy.empty(shape=(num_of_roles, rep_size * role_structure_context_size))
    input_role_matrix = numpy.empty(shape=(num_of_roles, vector_size))
    for i in range(0, num_of_roles):
        role = all_arg_role_list[str(i + 1)]
        roleVec = get_word_vec(role, word_vectors, vector_size)
        for j in range(0, vector_size):
            input_role_matrix[i, j] = roleVec[j]

        role_structures = all_type_role_structure[str(i + 1)]
        while len(role_structures) < role_structure_context_size:
            role_structures.append("PADDING" + "\t" + "PADDING")

        for j in range(0, role_structure_context_size):
            structure = role_structures[j]
            parts = structure.strip().split("\t")
            vec1 = get_word_vec(parts[0], word_vectors, vector_size)
            vec2 = get_word_vec(parts[1], word_vectors, vector_size)

            for m in range(0, vector_size):
                input_role_structure_matrix[i, j * rep_size + m] = vec1[m]
                input_role_structure_matrix[i, j * rep_size + vector_size + m] = vec2[m]
    return input_role_matrix, input_role_structure_matrix


def get_roles_for_train(train_role_flag, arg_path_file_specific, arg_path_file_generic):
    all_arg_role_list, index_2_role, trigger_role_2_index, index_2_norm_role, trigger_norm_role_2_index = \
        load_roles(arg_path_file_specific, arg_path_file_generic)
    types = []
    for i in range(0, len(index_2_role)):
        types.append(0)

    f = open(train_role_flag, 'r')
    for line in f:
        parts = line.strip().split("\t")
        type = parts[1].lower()
        for i in range(2, len(parts)):
            role = parts[i]
            trigger_role = type + "##" + role
            if trigger_role_2_index.has_key(role):
                index = int(trigger_role_2_index[role]) - 1
                types[index] = 1
            if trigger_role_2_index.has_key(trigger_role):
                index = int(trigger_role_2_index[trigger_role]) - 1
                types[index] = 1
    f.close()
    return types


def get_roles_for_train_1(train_role_flag, arg_path_file_merge):
    all_arg_role_list, all_trigger_role_structure, index_2_role, trigger_role_2_index, index_2_norm_role, \
    trigger_norm_role_2_index = load_roles_1(arg_path_file_merge)
    types = []
    for i in range(0, len(index_2_role)):
        types.append(0)

    f = open(train_role_flag, 'r')
    for line in f:
        parts = line.strip().split("\t")
        type = parts[1].lower()
        for i in range(2, len(parts)):
            role = parts[i]
            trigger_role = type + "##" + role
            if trigger_role_2_index.has_key(role):
                index = int(trigger_role_2_index[role]) - 1
                types[index] = 1
            if trigger_role_2_index.has_key(trigger_role):
                index = int(trigger_role_2_index[trigger_role]) - 1
                types[index] = 1
    f.close()
    return types
