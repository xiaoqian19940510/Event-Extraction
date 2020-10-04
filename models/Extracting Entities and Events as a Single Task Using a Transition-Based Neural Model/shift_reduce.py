import numpy as np
import dynet as dy
import nn
import ops
from dy_utils import ParamManager as pm
from actions import Actions
from vocab import Vocab
from event_constraints import EventConstraint
import io_utils

class MultiTask(object):

    def __init__(self, config, encoder_output_dim , action_dict, ent_dict, tri_dict, arg_dict):
        self.config = config
        self.model = pm.global_collection()
        bi_rnn_dim = encoder_output_dim  # config['rnn_dim'] * 2 #+ config['edge_embed_dim']
        lmda_dim = config['lmda_rnn_dim']
        part_ent_dim = config['part_ent_rnn_dim']

        self.lmda_dim = lmda_dim
        self.bi_rnn_dim = bi_rnn_dim

        hidden_input_dim = lmda_dim * 3 + bi_rnn_dim * 2   + config['out_rnn_dim']

        self.hidden_arg = nn.Linear(hidden_input_dim, config['output_hidden_dim'],
                                    activation='tanh')
        self.output_arg = nn.Linear(config['output_hidden_dim'], len(arg_dict))

        hidden_input_dim_co = lmda_dim * 3 + bi_rnn_dim * 2 + config['out_rnn_dim']
        self.hidden_ent_corel = nn.Linear(hidden_input_dim_co, config['output_hidden_dim'],
                                    activation='tanh')
        self.output_ent_corel = nn.Linear(config['output_hidden_dim'], 2)

        self.position_embed = nn.Embedding(500, 20)

        attn_input = self.bi_rnn_dim * 1 + 20 * 2
        self.attn_hidden = nn.Linear(attn_input, 80, activation='tanh')
        self.attn_out = nn.Linear(80, 1)

    def forward_ent_corel(self, beta_embed, lmda_embed, sigma_embed, delta_embed, out_embed, hidden_mat, start1, ent1, start2, end2, seq_len, last_h, gold_arg):
        attn_rep = self.position_aware_attn(hidden_mat, last_h, start1, ent1, start2, end2, seq_len)
        state_embed = ops.cat(
            [beta_embed, lmda_embed, sigma_embed, delta_embed, out_embed, attn_rep], dim=0)
        state_embed = dy.dropout(state_embed, 0.2)
        hidden = self.hidden_ent_corel(state_embed)
        out = self.output_ent_corel(hidden)

        loss = dy.pickneglogsoftmax(out, gold_arg)
        return loss

    def forward_arg(self, beta_embed, lmda_embed, sigma_embed, delta_embed, out_embed, hidden_mat, tri_idx, ent_start, ent_end, seq_len, last_h, gold_arg):
        attn_rep = self.position_aware_attn(hidden_mat, last_h, tri_idx, tri_idx, ent_start, ent_end, seq_len)
        state_embed = ops.cat(
            [beta_embed, lmda_embed, sigma_embed, delta_embed, out_embed, attn_rep], dim=0)
        state_embed = dy.dropout(state_embed, 0.25)
        hidden = self.hidden_arg(state_embed)
        out = self.output_arg(hidden)

        # probs = dy.softmax(out)
        # gold_prob = dy.pick(probs, gold_arg)
        # log_gold_prob = dy.log(gold_prob)
        # loss_weight = dy.pow(1.03 - gold_prob, dy.scalarInput(2))
        # loss = - loss_weight * log_gold_prob

        loss = dy.pickneglogsoftmax(out, gold_arg)
        return loss

    def decode_arg(self, beta_embed, lmda_embed, sigma_embed, delta_embed, out_embed, hidden_mat, tri_idx, ent_start, ent_end, seq_len, last_h):
        attn_rep = self.position_aware_attn(hidden_mat, last_h, tri_idx, tri_idx, ent_start, ent_end, seq_len)
        state_embed = ops.cat(
            [beta_embed, lmda_embed, sigma_embed, delta_embed, out_embed, attn_rep], dim=0)
        hidden = self.hidden_arg(state_embed)
        out = self.output_arg(hidden)
        np_score = out.npvalue().flatten()
        return np.argmax(np_score)



    def position_aware_attn(self, hidden_mat, last_h, start1, ent1, start2, end2, seq_len):
        tri_pos_list = []
        ent_pos_list = []

        for i in range(seq_len):
            tri_pos_list.append(io_utils.relative_position(start1, ent1, i))
            ent_pos_list.append(io_utils.relative_position(start2, end2, i))

        tri_pos_emb = self.position_embed(tri_pos_list)
        tri_pos_mat = ops.cat(tri_pos_emb, 1)
        ent_pos_emb = self.position_embed(ent_pos_list)
        ent_pos_mat = ops.cat(ent_pos_emb, 1)

        #expand_last_h = nn.cat([last_h] * seq_len, 1)
        # (birnn * 2 + pos_emb*2, seq_len)
        att_input = ops.cat([hidden_mat, tri_pos_mat, ent_pos_mat], 0)
        hidden = self.attn_hidden(att_input)
        attn_out = self.attn_out(hidden)
        # (1, seq_len)
        attn_prob = nn.softmax(attn_out, dim=1)
        # (rnn_dim * 2, 1)
        rep = hidden_mat * dy.transpose(attn_prob)

        return rep


class ShiftReduce(object):

    def __init__(self, config, encoder_output_dim , action_dict, ent_dict, tri_dict, arg_dict):
        self.config = config
        self.model = pm.global_collection()

        self.multi_task = MultiTask(config, encoder_output_dim , action_dict, ent_dict, tri_dict, arg_dict)
        self.arg_null_id = arg_dict[Vocab.NULL]

        bi_rnn_dim = encoder_output_dim  # config['rnn_dim'] * 2 #+ config['edge_embed_dim']
        lmda_dim = config['lmda_rnn_dim']
        part_ent_dim = config['part_ent_rnn_dim']

        self.lmda_dim = lmda_dim
        self.bi_rnn_dim = bi_rnn_dim
        self.lambda_var = nn.LambdaVar(lmda_dim)

        dp_state = config['dp_state']
        dp_state_h = config['dp_state_h']


        self.sigma_rnn = nn.StackLSTM(lmda_dim, lmda_dim, dp_state, dp_state_h)  # stack
        self.delta_rnn = nn.StackLSTM(lmda_dim, lmda_dim, dp_state, dp_state_h)   # will be pushed back

        self.part_ent_rnn = nn.StackLSTM(bi_rnn_dim, part_ent_dim, dp_state, dp_state_h)
        #self.beta = []  # buffer, unprocessed words
        self.actions_rnn = nn.StackLSTM(config['action_embed_dim'], config['action_rnn_dim'], dp_state, dp_state_h)
        self.out_rnn = nn.StackLSTM(bi_rnn_dim, config['out_rnn_dim'], dp_state, dp_state_h)

        self.act_table = nn.Embedding(len(action_dict), config['action_embed_dim'])
        self.ent_table = nn.Embedding(len(ent_dict), config['entity_embed_dim'])
        self.tri_table = nn.Embedding(len(tri_dict), config['trigger_embed_dim'])

        self.act= Actions(action_dict, ent_dict, tri_dict, arg_dict)

        hidden_input_dim = bi_rnn_dim + lmda_dim * 3 + part_ent_dim \
                           + config['action_rnn_dim'] + config['out_rnn_dim']

        self.hidden_linear = nn.Linear(hidden_input_dim, config['output_hidden_dim'], activation='tanh')
        self.output_linear = nn.Linear(config['output_hidden_dim'], len(action_dict))
        entity_embed_dim = config['entity_embed_dim']
        trigger_embed_dim = config['trigger_embed_dim']

        ent_to_lmda_dim = config['part_ent_rnn_dim'] + entity_embed_dim #+ config['sent_vec_dim'] * 4
        self.ent_to_lmda = nn.Linear(ent_to_lmda_dim, lmda_dim, activation='tanh')
        tri_to_lmda_dim = bi_rnn_dim + trigger_embed_dim #+ config['sent_vec_dim']
        self.tri_to_lmda = nn.Linear(tri_to_lmda_dim, lmda_dim, activation='tanh')

        self.hidden_arg = nn.Linear(lmda_dim * 2 + self.bi_rnn_dim, config['output_hidden_dim'],
                                    activation='tanh')
        self.output_arg = nn.Linear(config['output_hidden_dim'], len(arg_dict))
        self.empty_buffer_emb = self.model.add_parameters((bi_rnn_dim,), name='bufferGuardEmb')

        self.event_cons = EventConstraint(ent_dict, tri_dict, arg_dict)
        #self.cached_valid_args = self.cache_valid_args(ent_dict, tri_dict)

        self.empty_times = 0


    def cache_valid_args(self, ent_dict, tri_dict):
        cached_valid_args = {}
        for ent_type in ent_dict.vals():
            for tri_type in tri_dict.vals():
                valid_args = self.event_cons.get_constraint_arg_types(ent_type, tri_type)
                if valid_args is None:
                    valid_actions = []
                else:
                    valid_actions = self.act.get_act_ids_by_args(valid_args)
                cached_valid_args[(ent_type, tri_type)] = valid_actions
        print('cached_valid_args,',len(cached_valid_args))
        return cached_valid_args

    def get_valid_args(self, ent_type_id, tri_type_id):
        return self.cached_valid_args[(ent_type_id, tri_type_id)]


    def __call__(self, toks, hidden_state_list, last_h, oracle_actions=None,
                 oracle_action_strs=None, is_train=True, ents=None, tris=None, args=None):
        ent_dic = dict()
        tri_dic = dict()
        gold_arg_dict = {(arg[0], arg[2]): arg[-1] for arg in args}  # (ent_start, tri_idx):role_type

        same_event_ents = self.same(args)

        args = []

        hidden_mat = ops.cat(hidden_state_list, 1)
        seq_len = len(toks)


        buffer = nn.Buffer(self.bi_rnn_dim, hidden_state_list)

        losses = []
        loss_rels = []
        loss_roles = []
        pred_action_strs = []

        self.sigma_rnn.init_sequence(not is_train)
        self.delta_rnn.init_sequence(not is_train)
        self.part_ent_rnn.init_sequence(not is_train)
        self.actions_rnn.init_sequence(not is_train)
        self.out_rnn.init_sequence(not is_train)

        steps = 0
        while not (buffer.is_empty() and self.lambda_var.is_empty() and self.part_ent_rnn.is_empty()):
            pre_action = None if self.actions_rnn.is_empty() else self.actions_rnn.last_idx()
            # based on parser state, get valid actions
            valid_actions = []

            if pre_action is not None and self.act.is_ent_gen(pre_action):
                valid_actions += [self.act.entity_back_id] #[self.act.entity_back_id, self.act.entity_shift_id]

            # There are parts of the entity in e, we should finish this entity before process other actions
            elif not self.part_ent_rnn.is_empty():
                valid_actions += [self.act.entity_shift_id]
                valid_actions += self.act.get_ent_gen_list()


            elif not self.lambda_var.is_empty():

                if self.sigma_rnn.is_empty():
                    valid_actions += [self.act.shift_id, self.act.copy_shift_id]
                else:
                    valid_actions += [self.act.no_pass_id]
                    lmda_idx = self.lambda_var.idx
                    sigma_idx = self.sigma_rnn.last_idx()

                    if lmda_idx in ent_dic and sigma_idx in tri_dic:
                        valid_actions += [self.act.right_pass_id]

                    elif lmda_idx in tri_dic and sigma_idx in ent_dic:
                        valid_actions += [self.act.left_pass_id]


            else:
                valid_actions += [self.act.entity_shift_id, self.act.o_del_id]
                valid_actions += self.act.get_tri_gen_list()


            action = None

            if buffer.is_empty():
                self.empty_times += 1
            beta_embed = self.empty_buffer_emb if buffer.is_empty() else buffer.hidden_embedding()
            lmda_embed = self.lambda_var.embedding()
            sigma_embed = self.sigma_rnn.embedding()
            delta_embed = self.delta_rnn.embedding()
            part_ent_embed = self.part_ent_rnn.embedding()
            action_embed = self.actions_rnn.embedding()
            out_embed = self.out_rnn.embedding()


            state_embed = ops.cat([beta_embed, lmda_embed, sigma_embed, delta_embed, part_ent_embed, action_embed, out_embed], dim=0)
            if is_train:
                state_embed = dy.dropout(state_embed, self.config['dp_out'])
            hidden_rep = self.hidden_linear(state_embed)

            logits = self.output_linear(hidden_rep)
            if is_train:
                log_probs = dy.log_softmax(logits, valid_actions)
            else:
                log_probs = dy.log_softmax(logits, valid_actions)

            if is_train:
                action = oracle_actions[steps]
                action_str = oracle_action_strs[steps]
                if action not in valid_actions:
                    raise RuntimeError('Action %s dose not in valid_actions'%action_str)
                # append the action-specific loss
                #if self.act.is_o_del(action) or self.act.is_tri_gen(action):
                losses.append(dy.pick(log_probs, action))
                #val, idx = log_probs.tensor_value().topk(0, 5)

            else:
                np_log_probs = log_probs.npvalue()
                act_prob = np.max(np_log_probs)
                action = np.argmax(np_log_probs)
                action_str = self.act.to_act_str(action)
                pred_action_strs.append(action_str)
                #print(action_str)

            #if True:continue

            # execute the action to update the parser state
            if self.act.is_o_del(action):
                hx, idx = buffer.pop()
                self.out_rnn.push(hx, idx)


            elif self.act.is_tri_gen(action):
                hx, idx = buffer.pop()
                type_id = self.act.to_tri_id(action)
                tri_dic[idx] = (idx, type_id)

                tri_embed = self.tri_table[type_id]
                tri_rep = self.tri_to_lmda(ops.cat([hx, tri_embed], dim=0))
                #tri_rep = self.tri_to_lmda(hx)
                self.lambda_var.push(tri_rep, idx, nn.LambdaVar.TRIGGER)

            elif self.act.is_ent_shift(action):
                if buffer.is_empty():
                    break
                hx, idx = buffer.pop()
                self.part_ent_rnn.push(hx, idx)


            elif self.act.is_ent_gen(action):
                start, end = self.part_ent_rnn.idx_range()
                type_id = self.act.to_ent_id(action)
                ent = (start, end, type_id)
                ent_dic[start] = ent
                hx, _ = self.part_ent_rnn.last_state()
                ent_embed = self.ent_table[type_id]

                ent_rep = self.ent_to_lmda(ops.cat([hx, ent_embed], dim=0))
                #ent_rep = self.ent_to_lmda(hx)

                self.lambda_var.push(ent_rep, start, nn.LambdaVar.ENTITY)


            elif self.act.is_ent_back(action):
                new_idx = buffer.idx
                new_idx -= len(self.part_ent_rnn) - 1
                buffer.move_pointer(new_idx)

                self.part_ent_rnn.clear()


            elif self.act.is_shift(action):
                while not self.delta_rnn.is_empty():
                    self.sigma_rnn.push(*self.delta_rnn.pop())

                self.sigma_rnn.push(*self.lambda_var.pop())


            elif self.act.is_copy_shift(action):
                while not self.delta_rnn.is_empty():
                    self.sigma_rnn.push(*self.delta_rnn.pop())

                self.sigma_rnn.push(*self.lambda_var.pop())
                buffer.move_back()


            elif self.act.is_no_pass(action):
                lmda_idx = self.lambda_var.idx
                sigma_last_embed, sigma_last_idx = self.sigma_rnn.pop()

                if lmda_idx in ent_dic and sigma_last_idx in ent_dic:
                    ent_start1, ent_end1, _ = ent_dic[lmda_idx]
                    ent_start2, ent_end2, _ = ent_dic[sigma_last_idx]
                    corel = 1 if (lmda_idx, sigma_last_idx) in same_event_ents else 0
                    loss_corel = self.multi_task.forward_ent_corel(beta_embed, lmda_embed, sigma_embed, delta_embed, out_embed,
                                                                   hidden_mat, ent_start1, ent_end1, ent_start2, ent_end2, seq_len, last_h, corel)
                    loss_rels.append(loss_corel)

                self.delta_rnn.push(sigma_last_embed, sigma_last_idx)


            elif self.act.is_left_pass(action):
                lmda_idx = self.lambda_var.idx
                sigma_last_embed, sigma_last_idx = self.sigma_rnn.pop()
                tri_idx = lmda_idx
                ent_start, ent_end, _ = ent_dic[sigma_last_idx]

                if is_train:
                    role_label = gold_arg_dict.get((ent_start, tri_idx), self.arg_null_id)
                    loss_role = self.multi_task.forward_arg(beta_embed, lmda_embed, sigma_embed, delta_embed, out_embed,
                                                            hidden_mat, tri_idx, ent_start, ent_end, seq_len, last_h, role_label)
                    loss_roles.append(loss_role)

                else:
                    role_label = self.multi_task.decode_arg(beta_embed, lmda_embed, sigma_embed, delta_embed, out_embed,
                                                            hidden_mat, tri_idx, ent_start, ent_end, seq_len, last_h)

                event = (ent_start, ent_end, tri_idx, role_label)
                args.append(event)

                self.delta_rnn.push(sigma_last_embed, sigma_last_idx)


            elif self.act.is_right_pass(action):
                lmda_idx = self.lambda_var.idx
                sigma_last_embed, sigma_last_idx = self.sigma_rnn.pop()
                tri_idx = sigma_last_idx
                ent_start, ent_end, _ = ent_dic[lmda_idx]

                if is_train:
                    role_label = gold_arg_dict.get((ent_start, tri_idx), self.arg_null_id)
                    loss_role = self.multi_task.forward_arg(beta_embed, lmda_embed, sigma_embed, delta_embed, out_embed, hidden_mat, tri_idx, ent_start, ent_end, seq_len, last_h, role_label)
                    loss_roles.append(loss_role)

                else:
                    role_label = self.multi_task.decode_arg(beta_embed, lmda_embed, sigma_embed, delta_embed, out_embed, hidden_mat, tri_idx, ent_start, ent_end, seq_len, last_h)

                event = (ent_start, ent_end, tri_idx, role_label)
                args.append(event)

                self.delta_rnn.push(sigma_embed, sigma_last_idx)

            else:
                raise RuntimeError('Unknown action type:'+str(action))


            self.actions_rnn.push(self.act_table[action], action)

            steps += 1


        #if not is_train:print(len(self.actions_rnn.indices), self.actions_rnn.indices)

        pred_args = []
        if is_train:
            pred_args = set(args)

        else:
            for arg in args:
                ent_start, ent_end, tri_idx, role_type = arg
                ent_type_id = ent_dic[ent_start][-1]
                tri_type_id = tri_dic[tri_idx][-1]
                valid_args = self.event_cons.get_constraint_arg_types(ent_type_id, tri_type_id)
                if valid_args and role_type in valid_args:
                    pred_args.append(arg)


        self.clear()

        return losses, loss_roles, loss_rels, set(ent_dic.values()), set(tri_dic.values()), pred_args, pred_action_strs

    def clear(self):
        self.sigma_rnn.clear()
        self.delta_rnn.clear()
        self.part_ent_rnn.clear()
        self.actions_rnn.clear()
        self.lambda_var.clear()
        self.out_rnn.clear()

    def same(self, args):
        same_event_ents = set()
        for arg1 in args:
            ent_start1, ent_end1, tri_idx1, _ = arg1
            for arg2 in args:
                ent_start2, ent_end2, tri_idx2, _ = arg2
                if tri_idx1 == tri_idx2:
                    same_event_ents.add((ent_start1, ent_start2))
                    same_event_ents.add((ent_start2, ent_start1))

        return same_event_ents
