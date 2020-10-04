import numpy as np
import random
import tensorflow as tf

def is_NA(x):
    if isinstance(x,tuple):
        return 0 in x
    else:
        return x==0

def f_score(predict,golden,classify='single',mode='f'):
    assert len(predict)==len(golden)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    if isinstance(predict[0],tuple) and classify=='single':
        predict = [e[1] for e in predict]
        golden = [e[1] for e in golden]
    for i in range(len(predict)):
        if predict[i]==golden[i] and not is_NA(predict[i]):
            TP+=1
        elif predict[i]!=golden[i]:
            if is_NA(predict[i]) and not is_NA(golden[i]):
                FN+=1
            elif is_NA(golden[i]) and not is_NA(predict[i]):
                FP+=1
            elif (not is_NA(golden[i])) and (not is_NA(predict[i])):
                FN+=1
                FP+=1
            else:
                TN+=1
        else:
            TN+=1
    try:
        P = TP/(TP+FP)
        R = TP/(TP+FN)
        F = 2*P*R/(P+R)
    except:
        P=R=F=0

    if mode=='f':
        return P,R,F
    else:
        return TP,FN,FP,TN


def get_batch(data,batch_size,shuffle=True):
    assert len(list(set([np.shape(d)[0] for d in data]))) == 1
    num_data = np.shape(data[0])[0]
    indices = list(np.arange(0,num_data))
    if shuffle:
        random.shuffle(indices)
    for i in range((num_data // batch_size)+1):
        select_indices = indices[i*batch_size:(i+1)*batch_size]
        yield [np.take(d,select_indices,axis=0) for d in data]

def get_trigger_feeddict(model,batch,is_train=True):
    posis,sents,maskls,maskrs,event_types,lexical = batch
    return {model.posis:posis,model.sents:sents,model.maskls:maskls,model.maskrs:maskrs,
            model._labels:event_types,model.lexical:lexical,model.is_train:is_train}

def get_argument_feeddict(model,batch,is_train=True,stage='trigger'):
    if stage=='trigger':
        sents,event_types,roles,maskl,maskm,maskr,\
        trigger_lexical,argument_lexical,trigger_maskl,trigger_maskr,trigger_posis,argument_posis = batch
        return get_trigger_feeddict(model,(trigger_posis,sents,trigger_maskl,trigger_maskr,event_types,trigger_lexical),False)
    elif stage=="argument":
        if is_train:
            sents,event_types,roles,maskl,maskm,maskr,\
            trigger_lexical,argument_lexical,trigger_maskl,trigger_maskr,trigger_posis,argument_posis = batch
            return {model.sents:sents,model.trigger_posis:trigger_posis,model.argument_posis:argument_posis,
                    model.maskls:maskl,model.maskms:maskm,model.maskrs:maskr,
                    model.trigger_lexical:trigger_lexical,model.argument_lexical:argument_lexical,
                    model._labels:roles,model.is_train:is_train,model.event_types:event_types}
        else:
            sents,event_types,roles,maskl,maskm,maskr,\
            trigger_lexical,argument_lexical,trigger_maskl,trigger_maskr,trigger_posis,argument_posis,pred_event_types = batch
            return pred_event_types,{model.sents:sents,model.trigger_posis:trigger_posis,model.argument_posis:argument_posis,
                    model.maskls:maskl,model.maskms:maskm,model.maskrs:maskr,
                    model.trigger_lexical:trigger_lexical,model.argument_lexical:argument_lexical,
                    model._labels:roles,model.is_train:is_train,model.event_types:pred_event_types}
    else:
        raise ValueError("stage could only be trigger or argument")
