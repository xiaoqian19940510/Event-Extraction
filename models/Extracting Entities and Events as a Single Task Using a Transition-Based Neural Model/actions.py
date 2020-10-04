# -*- coding: utf-8 -*-

import numpy as np
import json




class Actions(object):

    trigger_gen = 'TRIGGER-GEN-'

    entity_shift = 'ENTITY-SHIFT'
    entity_gen = 'ENTITY-GEN-'
    entity_back = 'ENTITY-BACK'

    o_delete = 'O-DELETE'


    event_gen = 'EVENT-GEN-'

    shift = 'SHIFT'
    no_pass = 'NO-PASS'
    left_pass = 'LEFT-PASS'
    right_pass = 'RIGHT-PASS'

    back_shift = 'DUAL-SHIFT'
    # ------------------------------
    copy_shift = 'COPY-SHIFT'


    # ------------------------------
    # event_shift = 'EVENT-SHIFT-'
    # event_reduce = 'EVENT-REDUCE-'
    # no_reduce = 'NO-REDUCE'

    _PASS_PLACEHOLDER = 'PASS'
    _SHIFT_PLACEHOLDER = 'SHIFT'

    def __init__(self, action_dict, ent_dict, tri_dict, arg_dict, with_copy_shift=True):
        self.entity_shift_id = action_dict[Actions.entity_shift]
        self.entity_back_id = action_dict[Actions.entity_back]
        self.o_del_id = action_dict[Actions.o_delete]
        self.shift_id = action_dict[Actions.shift]
        self.left_pass_id = action_dict[Actions.left_pass]
        self.right_pass_id = action_dict[Actions.right_pass]
        if with_copy_shift:
            self.copy_shift_id = action_dict[Actions.copy_shift]
        else:
            self.back_shift_id = action_dict[Actions.back_shift]
        self.no_pass_id = action_dict[Actions.no_pass]
        self.ent_gen_group = set()
        self.tri_gen_group = set()
        #self.event_reduce_group = set()
        #self.event_shift_group = set()
        self.event_gen_group = set()

        self.act_to_ent_id = {}
        self.act_to_tri_id = {}
        self.act_to_arg_id = {}
        self.arg_to_act_id = {}

        self.act_id_to_str = {v:k for k, v in action_dict.items()}

        for name, id in action_dict.items():
            if name.startswith(Actions.entity_gen):
                self.ent_gen_group.add(id)
                self.act_to_ent_id[id] = ent_dict[name[len(Actions.entity_gen):]]

            elif name.startswith(Actions.trigger_gen):
                self.tri_gen_group.add(id)
                self.act_to_tri_id[id] = tri_dict[name[len(Actions.trigger_gen):]]

            elif name.startswith(Actions.event_gen):
                self.event_gen_group.add(id)
                self.act_to_arg_id[id] = arg_dict[name[len(Actions.event_gen):]]

        for k,v in self.act_to_arg_id.items():
            self.arg_to_act_id[v] = k

    def get_act_ids_by_args(self, arg_type_ids):
        acts = []
        for arg_id in arg_type_ids:
            acts.append(self.arg_to_act_id[arg_id])

        return acts

    def get_ent_gen_list(self):
        return list(self.ent_gen_group)

    def get_tri_gen_list(self):
        return list(self.tri_gen_group)


    def get_event_gen_list(self):
        return list(self.event_gen_group)


    def to_act_str(self, act_id):
        return self.act_id_to_str[act_id]

    def to_ent_id(self, act_id):
        return self.act_to_ent_id[act_id]

    def to_tri_id(self, act_id):
        return self.act_to_tri_id[act_id]

    def to_arg_id(self, act_id):
        return self.act_to_arg_id[act_id]

    # action check

    def is_ent_shift(self, act_id):
        return self.entity_shift_id == act_id

    def is_ent_back(self, act_id):
        return self.entity_back_id == act_id

    def is_o_del(self, act_id):
        return self.o_del_id == act_id

    def is_shift(self, act_id):
        return self.shift_id == act_id

    def is_back_shift(self, act_id):
        return self.back_shift_id == act_id

    def is_copy_shift(self, act_id):
        return self.copy_shift_id == act_id

    def is_no_pass(self, act_id):
        return self.no_pass_id == act_id

    def is_left_pass(self, act_id):
        return self.left_pass_id == act_id

    def is_right_pass(self, act_id):
        return self.right_pass_id == act_id

    def is_ent_gen(self, act_id):
        return act_id in self.ent_gen_group

    def is_tri_gen(self, act_id):
        return act_id in self.tri_gen_group


    def is_event_gen(self, act_id):
        return act_id in self.event_gen_group



    @staticmethod
    def make_oracle(tokens, triggers, ents, args, with_copy_shift=True):
        '''
        In this dataset, there are no nested entities sharing common start idx,
        therefore, we push back words from e to buffer exclude the first
        word in e.

        # TODO with_copy_shift
        trigger_list : [(idx, event_type)...] e.g. [(27, '500')...]
        ent_list :  [[start, end, ent_type],...]  e.g. [[3, 3, '402']...]
        arg_list :  [[arg_start, arg_end, trigger_idx, role_type]]  e.g. [[21, 21, 27, 'Vehicle'],...]
        '''


        ent_dic = {ent[0]:ent for ent in ents}

        trigger_dic = {tri[0]:tri[1] for tri in triggers}
        # (tri_idx, arg_start_idx)
        arg_dic = {(arg[2], arg[0]):arg for arg in args}

        # for tri in trigger_dic.keys():
        #     if tri in ent_dic:
        #         print(tri,'======', ent_dic[tri])


        actions = []

        # GEN entities and triggers
        tri_actions = {} # start_idx : actions list
        ent_actions = {} # start_idx : actions list

        for tri in triggers:
            idx, event_type = tri
            tri_actions[idx] = [Actions.trigger_gen + event_type]

        for ent in ents:
            start, end, ent_type, ref = ent
            act = []
            for _ in range(start, end + 1):
                act.append(Actions.entity_shift)

            act.append(Actions.entity_gen + ent_type)
            act.append(Actions.entity_back)
            ent_actions[start] = act




        for tri_i in trigger_dic:
            cur_actions = tri_actions[tri_i]
            for j in range(tri_i - 1, -1, -1):

                if j in trigger_dic:
                    cur_actions.append(Actions.no_pass)

                if j in ent_dic:
                    key = (tri_i, j)
                    if key in arg_dic:
                        arg_start, arg_end, trigger_idx, role_type = arg_dic[key]
                        cur_actions.append(Actions.left_pass)
                        #cur_actions.append(Actions.event_gen + role_type)
                    else:
                        cur_actions.append(Actions.no_pass)


            if tri_i in ent_dic:
                if with_copy_shift:
                    cur_actions.append(Actions.copy_shift)
                else:
                    cur_actions.append(Actions.back_shift)
            else:
                cur_actions.append(Actions.shift)


        for ent_i in ent_dic:
            cur_actions = ent_actions[ent_i]
            # Take into account that a word can be a trigger as well as an entity start
            if with_copy_shift and ent_i in trigger_dic:
                if (ent_i, ent_i) in arg_dic:
                    arg_start, arg_end, trigger_idx, role_type = arg_dic[(ent_i, ent_i)]
                    cur_actions.append(Actions.right_pass)
                    #cur_actions.append(Actions.event_gen + role_type)
                else:
                    cur_actions.append(Actions.no_pass)

            for j in range(ent_i - 1, -1, -1):

                if j in trigger_dic:
                    key = (j, ent_i)
                    if key in arg_dic:
                        arg_start, arg_end, trigger_idx, role_type = arg_dic[key]
                        cur_actions.append(Actions.right_pass)
                        #cur_actions.append(Actions.event_gen + role_type)

                    else:
                        cur_actions.append(Actions.no_pass)

                if j in ent_dic:
                    cur_actions.append(Actions.no_pass)


            cur_actions.append(Actions.shift)



        #print(tri_actions)
        #print(ent_actions)
        for i in range(len(tokens)):
            is_ent_or_tri = False
            if i in tri_actions:
                actions += tri_actions[i]
                is_ent_or_tri = True

            if i in ent_actions:
                actions += ent_actions[i]
                is_ent_or_tri = True

            if not is_ent_or_tri:
                actions.append(Actions.o_delete)


        return actions #, tri_actions, ent_actions







