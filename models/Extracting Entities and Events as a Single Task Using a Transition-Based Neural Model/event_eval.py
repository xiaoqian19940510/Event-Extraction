
from vocab import Vocab

def to_set(input):
    out_set = set()
    out_type_set = set()
    for x in input:
        out_set.add(tuple(x[:-1]))
        out_type_set.add(tuple(x))

    return out_set, out_type_set

class EventEval(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.correct_ent = 0.
        self.correct_ent_with_type = 0.
        self.num_pre_ent = 0.
        self.num_gold_ent = 0.

        self.correct_trigger = 0.
        self.correct_trigger_with_type = 0.
        self.num_pre_trigger = 0.
        self.num_gold_trigger = 0.

        self.correct_arg = 0.
        self.correct_arg_with_role = 0.
        self.num_pre_arg_no_type = 0.
        self.num_gold_arg_no_type = 0.
        self.num_pre_arg = 0.
        self.num_gold_arg = 0.

        # ------------------------------
        self.num_tri_error = 0
        self.num_ent_bound_error = 0
        self.num_arg_error = 0
        self.num_arg_error_with_role = 0
        self.num_ent_not_in_arg_error = 0
        self.num_tri_type_not_in_arg_error = 0

        self.tri_type_error_count = {}
        self.arg_type_error_count = {}
        self.arg_error_chunk = {}


    def get_coref_ent(self, g_ent_typed):
        ent_ref_dict = {}
        for ent1 in g_ent_typed:
            start1, end1, ent_type1, ent_ref1 = ent1
            coref_ents = []
            ent_ref_dict[(start1, end1)] = coref_ents
            for ent2 in g_ent_typed:
                start2, end2, ent_type2, ent_ref2 = ent2
                if ent_ref1 == ent_ref2:
                    coref_ents.append((start2, end2))
        return ent_ref_dict

   
    def split_prob(self, pred_args):
        sp_args, probs = [], []
        for arg in pred_args:
            sp_args.append(arg[:-1])
            probs.append(arg[-1])
        return sp_args, probs

    def update(self, pred_ents, gold_ents, pred_triggers, gold_triggers, pred_args, gold_args, eval_arg=True, words=None):
        ent_ref_dict = self.get_coref_ent(gold_ents)

        p_ent, p_ent_typed = to_set(pred_ents)
        p_ent_to_type_dic = {(s,e):t for s,e,t in p_ent_typed}
        g_ent, g_ent_typed = to_set([ent[:-1] for ent in gold_ents])
        p_tri, p_tri_typed = to_set(pred_triggers)
        g_tri, g_tri_typed = to_set(gold_triggers)
        p_args, p_args_typed = to_set(pred_args)
        g_args, g_args_typed = to_set(gold_args)
        #p_args_typed = {(arg[2],arg[3]) for arg in p_args_typed}
        #g_args_typed = {(arg[2], arg[3]) for arg in g_args_typed}


        self.num_pre_ent += len(p_ent)
        self.num_gold_ent += len(g_ent)

        self.correct_ent += len(p_ent & g_ent)
        self.correct_ent_with_type += len(p_ent_typed & g_ent_typed)

        self.num_pre_trigger += len(p_tri)
        self.num_gold_trigger += len(g_tri)

        c_tri = p_tri & g_tri
        c_tri_typed = p_tri_typed & g_tri_typed
        self.correct_trigger += len(c_tri)
        self.correct_trigger_with_type += len(c_tri_typed)

        if not eval_arg:
            return

        c_tri_typed_indices = {tri[0] for tri in c_tri_typed}
        p_tri_dic = {tri[0]: tri[1] for tri in p_tri_typed}
        g_tri_dic = {tri[0]: tri[1] for tri in g_tri_typed}
        p_arg_mention = {(arg[0], arg[1], p_tri_dic[arg[2]]) for arg in p_args_typed}
        g_arg_mention = {(arg[0], arg[1], g_tri_dic[arg[2]]) for arg in g_args_typed}
        p_arg_mention_typed = {(arg[0], arg[1], p_tri_dic[arg[2]], arg[3]) for arg in p_args_typed}
        g_arg_mention_typed = {(arg[0], arg[1], g_tri_dic[arg[2]], arg[3]) for arg in g_args_typed}

        self.num_pre_arg_no_type += len(p_arg_mention)
        self.num_gold_arg_no_type += len(g_arg_mention)

        self.num_pre_arg += len(p_arg_mention_typed)
        self.num_gold_arg += len(g_arg_mention_typed)

        for p_arg in p_arg_mention:
            p_start, p_end, p_tri_type = p_arg
            # if p_tri_idx not in c_tri_typed_indices:
            #     continue
            if (p_start, p_end) not in ent_ref_dict:
                continue
            for coref_ent in ent_ref_dict[(p_start, p_end)]:
                if (coref_ent[0], coref_ent[1], p_tri_type) in g_arg_mention:
                    self.correct_arg += 1
                    break
            else:
                self.num_arg_error += 1


        # for p_arg in p_args_typed:
        #     #start, end, tri_idx, tri_type = p_arg
        #     tri_idx = p_arg[-2]
        #     if p_arg in g_args_typed and tri_idx in c_tri_typed_indices:
        #         self.correct_arg_with_role += 1


        for num, p_arg in enumerate(p_arg_mention_typed):
            p_start, p_end, p_tri_type, p_role_type = p_arg
            p_ent_type = p_ent_to_type_dic[(p_start, p_end)]
            #tri_idx = p_arg[-2]
            # if p_tri_idx not in c_tri_typed_indices:
            #     self.num_tri_error += 1
            #     continue
            if (p_start, p_end) not in ent_ref_dict:
                self.num_ent_bound_error += 1
                continue

            for coref_ent in ent_ref_dict[(p_start, p_end)]:
                if (coref_ent[0], coref_ent[1], p_tri_type, p_role_type) in g_arg_mention_typed:
                    self.correct_arg_with_role += 1
                    break

            else:
                self.num_arg_error_with_role += 1



            for coref_ent in ent_ref_dict[(p_start, p_end)]:
                has_ent = False
                for g_arg in g_arg_mention_typed:
                    if coref_ent[0]==g_arg[0] and  coref_ent[1]==g_arg[1]:
                        has_ent = True
                        break
                if has_ent:
                    break
            else:
                self.num_ent_not_in_arg_error += 1
                chunk = tuple([words[i].lower() for i in range(p_start, p_end+1)])
                # print(' '.join(words))
                # print(chunk, p_tri_type, p_role_type)
                # print(p_arg_mention_typed)
                # print(g_arg_mention_typed)
                # print('==',[(words[tri[0]], tri[1]) for tri in pred_triggers])
                # print('==',[(words[tri[0]], tri[1]) for tri in gold_triggers])

                if chunk not in self.arg_error_chunk:
                    self.arg_error_chunk[chunk] = 1
                else:
                    self.arg_error_chunk[chunk] += 1

                key = p_ent_type
                if key not in self.arg_type_error_count:
                    self.arg_type_error_count[key] = 1
                else:
                    self.arg_type_error_count[key] += 1


            for g_arg in g_arg_mention_typed:
                if p_tri_type == g_arg[2]:
                    break
            else:
                self.num_tri_type_not_in_arg_error += 1
                key = (p_tri_type, p_ent_type)
                if key not in self.tri_type_error_count:
                    self.tri_type_error_count[key] = 1
                else:
                    self.tri_type_error_count[key] += 1



    def report(self):
        p_ent = self.correct_ent / (self.num_pre_ent + 1e-18)
        r_ent = self.correct_ent / (self.num_gold_ent + 1e-18)
        f_ent = 2 * p_ent * r_ent / (p_ent + r_ent + 1e-18)

        p_ent_typed = self.correct_ent_with_type / (self.num_pre_ent + 1e-18)
        r_ent_typed = self.correct_ent_with_type / (self.num_gold_ent + 1e-18)
        f_ent_typed = 2 * p_ent_typed * r_ent_typed / (p_ent_typed + r_ent_typed + 1e-18)

        p_tri = self.correct_trigger / (self.num_pre_trigger + 1e-18)
        r_tri = self.correct_trigger / (self.num_gold_trigger + 1e-18)
        f_tri = 2 * p_tri * r_tri / (p_tri + r_tri + 1e-18)

        p_tri_typed = self.correct_trigger_with_type / (self.num_pre_trigger + 1e-18)
        r_tri_typed = self.correct_trigger_with_type / (self.num_gold_trigger + 1e-18)
        f_tri_typed = 2 * p_tri_typed * r_tri_typed / (p_tri_typed + r_tri_typed + 1e-18)


        p_arg = self.correct_arg / (self.num_pre_arg_no_type + 1e-18)
        r_arg = self.correct_arg / (self.num_gold_arg_no_type + 1e-18)
        f_arg = 2 * p_arg * r_arg / (p_arg + r_arg + 1e-18)

        p_arg_typed = self.correct_arg_with_role / (self.num_pre_arg + 1e-18)
        r_arg_typed = self.correct_arg_with_role / (self.num_gold_arg + 1e-18)
        f_arg_typed = 2 * p_arg_typed * r_arg_typed / (p_arg_typed + r_arg_typed + 1e-18)


        print('num_pre_arg:', self.num_pre_arg)
        print('num_gold_arg:', self.num_gold_arg)
        print('correct_arg_with_role:', self.correct_arg_with_role)

        print('num_tri_error:', self.num_tri_error)
        print('num_ent_bound_error:', self.num_ent_bound_error)
        print('num_arg_error:', self.num_arg_error)
        print('num_arg_error_with_role:', self.num_arg_error_with_role)

        print('num_ent_not_in_arg_error:', self.num_ent_not_in_arg_error)
        print('num_tri_type_not_in_arg_error:', self.num_tri_type_not_in_arg_error)

        # for tri_type, count in self.tri_type_error_count.items():
        #     print(tri_type, ' : ' ,count)
        #
        # print(sum(self.tri_type_error_count.values()))
        #
        # for arg_type, count in self.arg_type_error_count.items():
        #     print(arg_type, ' : ' ,count)
        #
        # for chunk, count in self.arg_error_chunk.items():
        #     print(chunk, count)
        # print(len(self.arg_error_chunk))

        return (p_ent, r_ent, f_ent), (p_ent_typed, r_ent_typed, f_ent_typed), \
               (p_tri, r_tri, f_tri), (p_tri_typed, r_tri_typed, f_tri_typed), \
               (p_arg, r_arg, f_arg), (p_arg_typed, r_arg_typed, f_arg_typed)

