

import os
from vocab import Vocab
from io_utils import read_yaml, read_lines, read_json_lines
from str_utils import capitalize_first_char, normalize_tok, normalize_sent, collapse_role_type

class EventConstraint(object):
    '''
       This class is used to make sure that (event types, entity types) -> (argument roles) obey event constraints. 
    '''

    def __init__(self, ent_dict, tri_dict, arg_dict):
        constraint_file = './data_files/argrole_dict.txt'

        self.constraint_list = [] # [(ent_type, tri_type, arg_type)]
        for line in read_lines(constraint_file):
            line = str(line).lower()
            arr = line.split()
            arg_type = arr[0]
            for pair in arr[1:]:
                pair_arr = pair.split(',')
                tri_type = pair_arr[0]
                ent_type = pair_arr[1]
                ent_type = self._replace_ent(ent_type)
                self.constraint_list.append((ent_type, tri_type, arg_type))

        print('Event constraint size:',len(self.constraint_list))
        # { (ent_type, tri_type) : (arg_type1, ...)}
        self.ent_tri_to_arg_hash = {}
        for cons in self.constraint_list:
            ent_id = ent_dict[cons[0]]
            tri_id = tri_dict[cons[1]]
            arg_id = arg_dict[cons[2]]
            # ent_id = cons[0]
            # tri_id = cons[1]
            # arg_id = cons[2]
            if (ent_id, tri_id) not in self.ent_tri_to_arg_hash:
                self.ent_tri_to_arg_hash[(ent_id, tri_id)] = set()

            self.ent_tri_to_arg_hash[(ent_id, tri_id)].add(arg_id)

        #print(self.ent_tri_to_arg_hash)

        # single = 0
        # for key, val in self.ent_tri_to_arg_hash.items():
        #     if len(val) == 1:
        #         single += 1
        # print(single)

    def _replace_ent(self, ent_type):
        if ent_type == 'time':
            return 'tim'

        if ent_type == 'value':
            return 'val'

        return ent_type

    def check_constraint(self, ent_type, tri_type, arg_type):
        if (ent_type, tri_type, arg_type) in self.constraint_list:
            return True
        else:
            return False


    def get_constraint_arg_types(self, ent_type_id, tri_type_id):
        return self.ent_tri_to_arg_hash.get((ent_type_id, tri_type_id), None)




