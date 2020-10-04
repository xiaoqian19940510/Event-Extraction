# -*- coding: utf-8 -*-
# AUTHOR: Shun Zheng
# DATE: 19-9-19

import numpy as np


def agg_event_role_tpfpfn_stats(pred_records, gold_records, role_num):
    """
    Aggregate TP,FP,FN statistics for a single event prediction of one instance.
    A pred_records should be formated as
    [(Record Index)
        ((Role Index)
            argument 1, ...
        ), ...
    ], where argument 1 should support the '=' operation and the empty argument is None.
    """
    role_tpfpfn_stats = [[0] * 3 for _ in range(role_num)]

    if gold_records is None:
        if pred_records is not None:  # FP
            for pred_record in pred_records:
                assert len(pred_record) == role_num
                for role_idx, arg_tup in enumerate(pred_record):
                    if arg_tup is not None:
                        role_tpfpfn_stats[role_idx][1] += 1
        else:  # ignore TN
            pass
    else:
        if pred_records is None:  # FN
            for gold_record in gold_records:
                assert len(gold_record) == role_num
                for role_idx, arg_tup in enumerate(gold_record):
                    if arg_tup is not None:
                        role_tpfpfn_stats[role_idx][2] += 1
        else:  # True Positive at the event level
            # sort predicted event records by the non-empty count
            # to remove the impact of the record order on evaluation
            pred_records = sorted(pred_records,
                                  key=lambda x: sum(1 for a in x if a is not None),
                                  reverse=True)
            gold_records = list(gold_records)

            while len(pred_records) > 0 and len(gold_records) > 0:
                pred_record = pred_records[0]
                assert len(pred_record) == role_num

                # pick the most similar gold record
                _tmp_key = lambda gr: sum([1 for pa, ga in zip(pred_record, gr) if pa == ga])
                best_gr_idx = gold_records.index(max(gold_records, key=_tmp_key))
                gold_record = gold_records[best_gr_idx]

                for role_idx, (pred_arg, gold_arg) in enumerate(zip(pred_record, gold_record)):
                    if gold_arg is None:
                        if pred_arg is not None:  # FP at the role level
                            role_tpfpfn_stats[role_idx][1] += 1
                        else:  # ignore TN
                            pass
                    else:
                        if pred_arg is None:  # FN
                            role_tpfpfn_stats[role_idx][2] += 1
                        else:
                            if pred_arg == gold_arg:  # TP
                                role_tpfpfn_stats[role_idx][0] += 1
                            else:
                                role_tpfpfn_stats[role_idx][1] += 1
                                role_tpfpfn_stats[role_idx][2] += 1

                del pred_records[0]
                del gold_records[best_gr_idx]

            # remaining FP
            for pred_record in pred_records:
                assert len(pred_record) == role_num
                for role_idx, arg_tup in enumerate(pred_record):
                    if arg_tup is not None:
                        role_tpfpfn_stats[role_idx][1] += 1
            # remaining FN
            for gold_record in gold_records:
                assert len(gold_record) == role_num
                for role_idx, arg_tup in enumerate(gold_record):
                    if arg_tup is not None:
                        role_tpfpfn_stats[role_idx][2] += 1

    return role_tpfpfn_stats


def agg_event_level_tpfpfn_stats(pred_records, gold_records, role_num):
    """
    Get event-level TP,FP,FN
    """
    # add role-level statistics as the event-level ones
    role_tpfpfn_stats = agg_event_role_tpfpfn_stats(
        pred_records, gold_records, role_num
    )

    return list(np.sum(role_tpfpfn_stats, axis=0))


def agg_ins_event_role_tpfpfn_stats(pred_record_mat, gold_record_mat, event_role_num_list):
    """
    Aggregate TP,FP,FN statistics for a single instance.
    A record_mat should be formated as
    [(Event Index)
        [(Record Index)
            ((Role Index)
                argument 1, ...
            ), ...
        ], ...
    ], where argument 1 should support the '=' operation and the empty argument is None.
    """
    assert len(pred_record_mat) == len(gold_record_mat)
    # tpfpfn_stat: TP, FP, FN
    event_role_tpfpfn_stats = []
    for event_idx, (pred_records, gold_records) in enumerate(zip(pred_record_mat, gold_record_mat)):
        role_num = event_role_num_list[event_idx]
        role_tpfpfn_stats = agg_event_role_tpfpfn_stats(pred_records, gold_records, role_num)
        event_role_tpfpfn_stats.append(role_tpfpfn_stats)

    return event_role_tpfpfn_stats


def agg_ins_event_level_tpfpfn_stats(pred_record_mat, gold_record_mat, event_role_num_list):
    assert len(pred_record_mat) == len(gold_record_mat)
    # tpfpfn_stat: TP, FP, FN
    event_tpfpfn_stats = []
    for event_idx, (pred_records, gold_records, role_num) in enumerate(zip(
            pred_record_mat, gold_record_mat, event_role_num_list)):
        event_tpfpfn = agg_event_level_tpfpfn_stats(pred_records, gold_records, role_num)
        event_tpfpfn_stats.append(event_tpfpfn)

    return event_tpfpfn_stats


def get_prec_recall_f1(tp, fp, fn):
    a = tp + fp
    prec = tp / a if a > 0 else 0
    b = tp + fn
    rec = tp / b if b > 0 else 0
    if prec > 0 and rec > 0:
        f1 = 2.0 / (1 / prec + 1 / rec)
    else:
        f1 = 0
    return prec, rec, f1


def measure_event_table_filling(pred_record_mat_list, gold_record_mat_list, event_type_roles_list, avg_type='micro',
                                dict_return=False):
    """
    The record_mat_list is formated as
    [(Document Index)
        [(Event Index)
            [(Record Index)
                ((Role Index)
                    argument 1, ...
                ), ...
            ], ...
        ], ...
    ]
    The argument type should support the '==' operation.
    Empty arguments and records are set as None.
    """
    event_role_num_list = [len(roles) for _, roles in event_type_roles_list]
    # to store total statistics of TP, FP, FN
    total_event_role_stats = [
        [
            [0]*3 for _ in range(role_num)
        ] for event_idx, role_num in enumerate(event_role_num_list)
    ]

    assert len(pred_record_mat_list) == len(gold_record_mat_list)
    for pred_record_mat, gold_record_mat in zip(pred_record_mat_list, gold_record_mat_list):
        event_role_tpfpfn_stats = agg_ins_event_role_tpfpfn_stats(
            pred_record_mat, gold_record_mat, event_role_num_list
        )
        for event_idx, role_num in enumerate(event_role_num_list):
            for role_idx in range(role_num):
                for sid in range(3):
                    total_event_role_stats[event_idx][role_idx][sid] += \
                        event_role_tpfpfn_stats[event_idx][role_idx][sid]

    per_role_metric = []
    per_event_metric = []

    num_events = len(event_role_num_list)
    g_tpfpfn_stat = [0] * 3
    g_prf1_stat = [0] * 3
    event_role_eval_dicts = []
    for event_idx, role_num in enumerate(event_role_num_list):
        event_tpfpfn = [0] * 3  # tp, fp, fn
        event_prf1_stat = [0] * 3
        per_role_metric.append([])
        role_eval_dicts = []
        for role_idx in range(role_num):
            role_tpfpfn_stat = total_event_role_stats[event_idx][role_idx][:3]
            role_prf1_stat = get_prec_recall_f1(*role_tpfpfn_stat)
            per_role_metric[event_idx].append(role_prf1_stat)
            for mid in range(3):
                event_tpfpfn[mid] += role_tpfpfn_stat[mid]
                event_prf1_stat[mid] += role_prf1_stat[mid]

            role_eval_dict = {
                'RoleType': event_type_roles_list[event_idx][1][role_idx],
                'Precision': role_prf1_stat[0],
                'Recall': role_prf1_stat[1],
                'F1': role_prf1_stat[2],
                'TP': role_tpfpfn_stat[0],
                'FP': role_tpfpfn_stat[1],
                'FN': role_tpfpfn_stat[2]
            }
            role_eval_dicts.append(role_eval_dict)

        for mid in range(3):
            event_prf1_stat[mid] /= role_num
            g_tpfpfn_stat[mid] += event_tpfpfn[mid]
            g_prf1_stat[mid] += event_prf1_stat[mid]

        micro_event_prf1 = get_prec_recall_f1(*event_tpfpfn)
        macro_event_prf1 = tuple(event_prf1_stat)
        if avg_type.lower() == 'micro':
            event_prf1_stat = micro_event_prf1
        elif avg_type.lower() == 'macro':
            event_prf1_stat = macro_event_prf1
        else:
            raise Exception('Unsupported average type {}'.format(avg_type))

        per_event_metric.append(event_prf1_stat)

        event_eval_dict = {
            'EventType': event_type_roles_list[event_idx][0],
            'MacroPrecision': macro_event_prf1[0],
            'MacroRecall': macro_event_prf1[1],
            'MacroF1': macro_event_prf1[2],
            'MicroPrecision': micro_event_prf1[0],
            'MicroRecall': micro_event_prf1[1],
            'MicroF1': micro_event_prf1[2],
            'TP': event_tpfpfn[0],
            'FP': event_tpfpfn[1],
            'FN': event_tpfpfn[2],
        }
        event_role_eval_dicts.append((event_eval_dict, role_eval_dicts))

    micro_g_prf1 = get_prec_recall_f1(*g_tpfpfn_stat)
    macro_g_prf1 = tuple(s / num_events for s in g_prf1_stat)
    if avg_type.lower() == 'micro':
        g_metric = micro_g_prf1
    else:
        g_metric = macro_g_prf1

    g_eval_dict = {
        'MacroPrecision': macro_g_prf1[0],
        'MacroRecall': macro_g_prf1[1],
        'MacroF1': macro_g_prf1[2],
        'MicroPrecision': micro_g_prf1[0],
        'MicroRecall': micro_g_prf1[1],
        'MicroF1': micro_g_prf1[2],
        'TP': g_tpfpfn_stat[0],
        'FP': g_tpfpfn_stat[1],
        'FN': g_tpfpfn_stat[2],
    }
    event_role_eval_dicts.append(g_eval_dict)

    if not dict_return:
        return g_metric, per_event_metric, per_role_metric
    else:
        return event_role_eval_dicts

