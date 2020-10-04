import numpy
import time
import sys
import subprocess
import os
import random
import cPickle
import copy

import theano
from theano import tensor as T
from collections import OrderedDict, defaultdict
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
import theano.tensor.shared_randomstreams
from jeeModels import *

dataset_path = '~/projects/jointEE/nn/externalFets/word2vec_jointEE.pkl'
#dataset_path = '../globHead/word2vec_jointEE.pkl'

scoreScript = '~/projects/jointEE/do'

data_sourceDir = '~/projects/jointEE/corpus/qi'
data_fileLists = {'train': '~/projects/jointEE/fileLists/train.txt',
                 'valid': '~/projects/jointEE/fileLists/valid.txt',
                 'test': '~/projects/jointEE/fileLists/test.txt'}
data_predictedFiles = {'train': '',
                       'valid': '',
                       'test': ''}

##################################################################

def setFetVector(index, numDim, binary, fetVec):
    vec = [0] * numDim
    vec[index-1] = 1
    fetVec.append((vec if binary == 1 else index))

def setZeroFetVector(numDim, binary, fetVec):
    vec = [0] * numDim
    fetVec.append((vec if binary == 1 else 0))
    
def produceZeroMatrix(row, col):
    #res = [ [0] * col for i in range(row)]
    return numpy.zeros((row, col), dtype='int32').tolist()

def produceOneMatrix(row, col):
    #res = [ [1] * col for i in range(row)]
    return numpy.ones((row, col), dtype='int32').tolist()

def produceMinusOneMatrix(row, col):
    #res = [ [-1] * col for i in range(row)]
    return (numpy.zeros((row, col), dtype='int32')-1).tolist()

def produceZeroTensor3(dim1, dim2, dim3):
    #res = []
    #for i in range(dim1):
    #    res += [produceZeroMatrix(dim2, dim3)]
    return numpy.zeros((dim1, dim2, dim3), dtype='int32').tolist()
    
def createRelativeDistaceBinaryMapping(mlen, slen):
    res = produceZeroTensor3(mlen, mlen, 2*mlen-1)
    for i in range(slen):
        for j in range(slen):
            pos = mlen + j - i - 1
            res[i][j][pos] = 1.0
    return res
    
def createRelativeDistaceIndexMapping(mlen, slen):
    res = produceZeroMatrix(mlen, mlen)
    for i in range(slen):
        for j in range(slen):
            res[i][j] = mlen + j - i
    return res
        
def generateDataInstance(rev, dictionaries, embeddings, features, idx2Etype, idx2Esubtype, eventEntityType, mLen, mNumEntities, mNodeFets,  mEdgeFets, skipByType):

    numDep = len(dictionaries['dep'])
    numTypeEntity = len(dictionaries['typeEntity'])
    numPossibleNode = len(dictionaries['possibleNode'])
    numPos = len(dictionaries['pos'])
    numChunk = len(dictionaries['chunk'])
    numClause = len(dictionaries['clause'])
    numRefer = len(dictionaries['refer'])
    numTitle = len(dictionaries['title'])
    numTypeOneEntity = len(dictionaries['typeOneEntity'])
    
    numTrigger = len(dictionaries['nodeLabel'])
    numArg = len(dictionaries['edgeLabel'])

    x = []
    dep = []
    ent = []
    possi = []
    pos = []
    chunk = []
    clause = []
    refer = []
    title = []
    oneEnt = []
    
    #typeDic = getTypeDict(numType)
    
    id = -1
    for word, rpos, rchunk, rclause, rrefer, rtitle, rdep, rtypeEntity, rtypeOneEntity, rposType in zip(rev["text"], rev["pos"], rev["chunk"], rev["clause"], rev["refer"], rev["title"], rev["dep"], rev["typeEntity"], rev["typeOneEntity"], rev["posType"]):
        id += 1
        #word = ' '.join(word.split('_'))
        if word in dictionaries["word"]:
            x.append(dictionaries["word"][word])
            
            vdep = [0] * numDep
            for i in rdep:
                vdep[i-1] = 1
            dep.append(vdep)
            
            vtypeEntity = [0] * numTypeEntity
            if i in rtypeEntity:
                vtypeEntity[i-1] = 1
            ent.append(vtypeEntity)
            
            vpossibleNode = [0] * numPossibleNode
            if i in rposType:
                vpossibleNode[i-1] = 1
            possi.append(vpossibleNode)
            
            setFetVector(rpos, numPos, features['pos'], pos)
            setFetVector(rchunk, numChunk, features['chunk'], chunk)
            setFetVector(rclause, numClause, features['clause'], clause)
            setFetVector(rrefer, numRefer, features['refer'], refer)
            setFetVector(rtitle, numTitle, features['title'], title)
            setFetVector(rtypeOneEntity, numTypeOneEntity, features['typeOneEntity'], oneEnt)                
        else:
            print 'unrecognized features '
            exit()
    
    if len(x) > mLen:
        print 'incorrect length!'
        exit()
    
    sentLength = len(x)
    
    if len(x) < mLen:
        vdep = [0] * numDep
        vtypeEntity = [0] * numTypeEntity
        vpossibleNode = [0] * numPossibleNode
        
        while len(x) < mLen:
            x.append(0)
            dep.append(vdep)
            ent.append(vtypeEntity)
            possi.append(vpossibleNode)
            
            setZeroFetVector(numPos, features['pos'], pos)
            setZeroFetVector(numChunk, features['chunk'], chunk)
            setZeroFetVector(numClause, features['clause'], clause)
            setZeroFetVector(numRefer, features['refer'], refer)
            setZeroFetVector(numTitle, features['title'], title)
            setZeroFetVector(numTypeOneEntity, features['typeOneEntity'], oneEnt)
    
    if sentLength != len(rev['nodeFets']):
        print 'length of sentence and feature matrix not the same'
        exit()
        
    revNodeFets = []
    for nfs in rev['nodeFets']:
        onfs = nfs
        while len(onfs) < mNodeFets: onfs += [0]
        revNodeFets += [onfs]
    while len(revNodeFets) < mLen:
        revNodeFets += [[0] * mNodeFets]

    revEdgeFets = []
    for fwid in range(0, sentLength):
        owfs = []
        for feid in range(0, len(rev["entities"])):
            lwfs = rev["edgeFets"][feid][fwid]
            oefs = lwfs
            while len(oefs) < mEdgeFets: oefs += [0]
            owfs += [oefs]
        while len(owfs) < mNumEntities: owfs += [[0] * mEdgeFets]
        revEdgeFets += [owfs]
    while len(revEdgeFets) < mLen:
        revEdgeFets += [produceZeroMatrix(mNumEntities, mEdgeFets)]
    
    fet = {'word' : x, 'pos' : pos, 'chunk' : chunk, 'clause' : clause, 'refer' : refer, 'title' : title, 'posType' : possi, 'dep' : dep, 'typeEntity' : ent, 'typeOneEntity' : oneEnt}
    
    if skipByType:
        skipped_triggerAnn = [0] * mLen
        skipped_triggerMaskTrain, skipped_triggerMaskTest = [], []
        skipped_triggerMaskTrainArg = []
        skipped_triggerMaskTestArg = []
    else:
        triggerAnn = [0] * mLen
        triggerMaskTrain, triggerMaskTest = [], []
        triggerMaskTrainArg = []
        triggerMaskTestArg = []
    for i, v in enumerate(rev["eligible"]):
        mvl = 1 if v == 1 else 0
        #mve = [mvl] * numTrigger
        if skipByType:
            skipped_triggerMaskTrain += [mvl] #mve
            skipped_triggerMaskTest += [mvl]
            skipped_triggerMaskTrainArg += [1] #produceOneMatrix(mNumEntities, numArg)
            skipped_triggerMaskTestArg += [1] #[1] * mNumEntities
        else:
            triggerMaskTrain += [1] #[[1] * numTrigger]
            triggerMaskTest += [1]
            triggerMaskTrainArg += [1] #produceOneMatrix(mNumEntities, numArg)
            triggerMaskTestArg += [1] #[1] * mNumEntities
    while len(triggerMaskTrain if not skipByType else skipped_triggerMaskTrain) < mLen:
        if skipByType:
            skipped_triggerMaskTrain.append(0) #[0] * numTrigger
            skipped_triggerMaskTest += [0]
            skipped_triggerMaskTrainArg += [0] #produceZeroMatrix(mNumEntities, numArg)
            skipped_triggerMaskTestArg += [0] #[0] * mNumEntities
        else:
            triggerMaskTrain.append(0) #[0] * numTrigger
            triggerMaskTest += [0]
            triggerMaskTrainArg += [0] #produceZeroMatrix(mNumEntities, numArg)
            triggerMaskTestArg += [0] #[0] * mNumEntities
        
    entities = [-1] * (1 + mNumEntities)
    for enid, entity in enumerate(rev["entities"]):
        entities[enid+1] = entity[1]
    entities[0] = len(rev["entities"])
    #if entities[0] == 0:
    #    entities[0] = 1
    #    entities[1] = 0
    #    print '***Encounter sentence with no entities'
    
    if not skipByType:
        argumentEntityIdAnn = produceMinusOneMatrix(mLen, mNumEntities)
        argumentPosAnn = produceZeroMatrix(mLen, mNumEntities)
        argumentLabelAnn = produceZeroMatrix(mLen, mNumEntities)
        argumentMaskTrain = produceZeroMatrix(mLen, mNumEntities) #produceZeroTensor3(mLen, mNumEntities, numArg)
    
        for i_pos in range(sentLength):
            for e_id in range(entities[0]):
                argumentEntityIdAnn[i_pos][e_id] = e_id
                argumentPosAnn[i_pos][e_id] = entities[e_id+1]
                argumentMaskTrain[i_pos][e_id] = 1 #[1] * numArg
    else:
        skipped_argumentEntityIdAnn = produceMinusOneMatrix(mLen, mNumEntities)
        skipped_argumentPosAnn = produceZeroMatrix(mLen, mNumEntities)
        skipped_argumentLabelAnn = produceZeroMatrix(mLen, mNumEntities)
        skipped_argumentMaskTrain = produceZeroMatrix(mLen, mNumEntities) #produceZeroTensor3(mLen, mNumEntities, numArg)
    
    for t_pos, t_trigger, t_arg in zip(rev["eventPos"], rev["eventTrigger"], rev["eventArgs"]):
        if not skipByType:
            triggerAnn[t_pos] = t_trigger
        else:
            skipped_triggerAnn[t_pos] = t_trigger
        
        if len(t_arg) == 0: continue
        
        if not skipByType:
            for i_arg in t_arg: argumentLabelAnn[t_pos][i_arg] = t_arg[i_arg]
        else:
            countId = 0
            for i_arg in t_arg:
                skipped_argumentEntityIdAnn[t_pos][countId] = i_arg
                skipped_argumentPosAnn[t_pos][countId] = entities[i_arg+1]
                skipped_argumentLabelAnn[t_pos][countId] = t_arg[i_arg]
                skipped_argumentMaskTrain[t_pos][countId] = 1 #[1] * numArg
                countId += 1
    
    if not skipByType:
        possibleEnityIdByTrigger = produceMinusOneMatrix(1 + len(eventEntityType), mNumEntities)
        possibleEnityPosByTrigger = produceZeroMatrix(1 + len(eventEntityType), mNumEntities)
        argumentMaskTest = produceZeroMatrix(1 + len(eventEntityType), mNumEntities)
        for i_pos in eventEntityType:
            for e_id in range(entities[0]):
                possibleEnityIdByTrigger[i_pos][e_id] = e_id
                possibleEnityPosByTrigger[i_pos][e_id] = entities[e_id+1]
                argumentMaskTest[i_pos][e_id] = 1
    #for e_id in range(entities[0]):
    #    possibleEnityIdByTrigger[0][e_id] = e_id
    #    possibleEnityPosByTrigger[0][e_id] = entities[e_id+1]
    #    argumentMaskTest[0][e_id] = 1
    else:
        skipped_possibleEnityIdByTrigger = produceMinusOneMatrix(1 + len(eventEntityType), mNumEntities)
        skipped_possibleEnityPosByTrigger = produceZeroMatrix(1 + len(eventEntityType), mNumEntities)
        skipped_argumentMaskTest = produceZeroMatrix(1 + len(eventEntityType), mNumEntities)
    
        for i_pos, peet in eventEntityType.items():
            pes = []
            for e_id, e_entity in enumerate(rev["entities"]):
                e_type = idx2Etype[e_entity[4]]
                e_subtype = idx2Esubtype[e_entity[5]]
                ett = e_type
                if e_type == 'VALUE' or e_type == 'TIME': ett = e_subtype
                if ett in peet: pes += [e_id]
        
            for pe_i, pe in enumerate(pes):
                skipped_possibleEnityIdByTrigger[i_pos][pe_i] = pe
                skipped_possibleEnityPosByTrigger[i_pos][pe_i] = entities[pe+1]
                skipped_argumentMaskTest[i_pos][pe_i] = 1
    
    anns, annsType = {}, {}
    
    anns['sentLength'], annsType['sentLength'] = sentLength, 'int32'
    
    if not skipByType:
        anns['triggerAnn'], annsType['triggerAnn'] = triggerAnn, 'int32'
        anns['triggerMaskTrain'], annsType['triggerMaskTrain'] = triggerMaskTrain, 'float32'
        anns['triggerMaskTest'], annsType['triggerMaskTest'] = triggerMaskTest, 'int32'
        anns['triggerMaskTrainArg'], annsType['triggerMaskTrainArg'] = triggerMaskTrainArg, 'float32'
        anns['triggerMaskTestArg'], annsType['triggerMaskTestArg'] = triggerMaskTestArg, 'int32'
    else:
        anns['skipped_triggerAnn'], annsType['skipped_triggerAnn'] = skipped_triggerAnn, 'int32'
        anns['skipped_triggerMaskTrain'], annsType['skipped_triggerMaskTrain'] = skipped_triggerMaskTrain, 'float32'
        anns['skipped_triggerMaskTest'], annsType['skipped_triggerMaskTest'] = skipped_triggerMaskTest, 'int32'
        anns['skipped_triggerMaskTrainArg'], annsType['skipped_triggerMaskTrainArg'] = skipped_triggerMaskTrainArg, 'float32'
        anns['skipped_triggerMaskTestArg'], annsType['skipped_triggerMaskTestArg'] = skipped_triggerMaskTestArg, 'int32'
    
    anns['entities'], annsType['entities'] = entities, 'int32'
    
    if not skipByType:
        anns['argumentEntityIdAnn'], annsType['argumentEntityIdAnn'] = argumentEntityIdAnn, 'int32'
        anns['argumentPosAnn'], annsType['argumentPosAnn'] = argumentPosAnn, 'int32'
        anns['argumentLabelAnn'], annsType['argumentLabelAnn'] = argumentLabelAnn, 'int32'
        anns['argumentMaskTrain'], annsType['argumentMaskTrain'] = argumentMaskTrain, 'float32'
    else:
        anns['skipped_argumentEntityIdAnn'], annsType['skipped_argumentEntityIdAnn'] = skipped_argumentEntityIdAnn, 'int32'
        anns['skipped_argumentPosAnn'], annsType['skipped_argumentPosAnn'] = skipped_argumentPosAnn, 'int32'
        anns['skipped_argumentLabelAnn'], annsType['skipped_argumentLabelAnn'] = skipped_argumentLabelAnn, 'int32'
        anns['skipped_argumentMaskTrain'], annsType['skipped_argumentMaskTrain'] = skipped_argumentMaskTrain, 'float32'
    
    if not skipByType:
        anns['possibleEnityIdByTrigger'], annsType['possibleEnityIdByTrigger'] = possibleEnityIdByTrigger, 'int32'
        anns['possibleEnityPosByTrigger'], annsType['possibleEnityPosByTrigger'] = possibleEnityPosByTrigger, 'int32'
        anns['argumentMaskTest'], annsType['argumentMaskTest'] = argumentMaskTest, 'int32'
    else:
        anns['skipped_possibleEnityIdByTrigger'], annsType['skipped_possibleEnityIdByTrigger'] = skipped_possibleEnityIdByTrigger, 'int32'
        anns['skipped_possibleEnityPosByTrigger'], annsType['skipped_possibleEnityPosByTrigger'] = skipped_possibleEnityPosByTrigger, 'int32'
        anns['skipped_argumentMaskTest'], annsType['skipped_argumentMaskTest'] = skipped_argumentMaskTest, 'int32'
    
    #anns['relDistBinary'], annsType['relDistBinary'] = createRelativeDistaceBinaryMapping(mLen, sentLength), 'float32'
    if not skipByType:
        anns['relDistIdxs'], annsType['relDistIdxs'] = createRelativeDistaceIndexMapping(mLen, sentLength), 'int32'
    else:
        anns['skipped_relDistIdxs'], annsType['skipped_relDistIdxs'] = createRelativeDistaceIndexMapping(mLen, sentLength), 'int32'
    
    if not skipByType:
        anns['NodeFets'], annsType['NodeFets'] = revNodeFets, 'int32'
        anns['EdgeFets'], annsType['EdgeFets'] = revEdgeFets, 'int32'
    else:
        anns['skipped_NodeFets'], annsType['skipped_NodeFets'] = revNodeFets, 'int32'
        anns['skipped_EdgeFets'], annsType['skipped_EdgeFets'] = revEdgeFets, 'int32'
    
    return fet, anns, annsType

def make_data(revs, dictionaries, embeddings, features, eventEntityType, skipByType):

    mLen = -1
    mNumEntities = -1
    mNodeFets = -1
    mEdgeFets = -1
    for rev in revs:
        if len(rev["text"]) > mLen:
            mLen = len(rev["text"])
        if len(rev["entities"]) > mNumEntities:
            mNumEntities = len(rev["entities"])
        for nfs in rev["nodeFets"]:
            if len(nfs) > mNodeFets: mNodeFets = len(nfs)
        for efs in rev["edgeFets"]:
            for wfs in efs:
                if len(wfs) > mEdgeFets: mEdgeFets = len(wfs)
    
    print 'maximum of length, numEntities, mNodeFets, mEdgeFets in the dataset: ', mLen, mNumEntities, mNodeFets, mEdgeFets
    
    idx2Etype = dict((k,v) for v,k in dictionaries['etype'].iteritems())
    idx2Esubtype = dict((k,v) for v,k in dictionaries['esubtype'].iteritems())
    
    #mLen += 1
    
    res = {}
    typeMap = None
    #counter = 0
    for rev in revs:
        #counter += 1
        #if counter % 10 == 0: print counter
        fet, anns, annsType = generateDataInstance(rev, dictionaries, embeddings, features, idx2Etype, idx2Esubtype, eventEntityType, mLen, mNumEntities, mNodeFets,  mEdgeFets, skipByType)
         
        if rev["corpus"] not in res: res[rev["corpus"]] = defaultdict(list)
        
        for kk in fet:
            res[rev["corpus"]][kk] += [fet[kk]]
            
        for kk in anns:
            res[rev["corpus"]][kk] += [anns[kk]]
        res[rev["corpus"]]['id'] += [rev['id']]
        
        typeMap = annsType
        typeMap['id'] = 'int32'
        
    return res, typeMap

def predict(corpus, batch, reModel, features, skipByType):
    evaluateCorpus = {}
    extra_data_num = -1
    nsen = corpus['word'].shape[0]
    if nsen % batch > 0:
        extra_data_num = batch - nsen % batch
        for ed in corpus:  
            extra_data = corpus[ed][:extra_data_num]
            evaluateCorpus[ed] = numpy.append(corpus[ed],extra_data,axis=0)
    else:
        for ed in corpus: 
            evaluateCorpus[ed] = corpus[ed]
        
    numBatch = evaluateCorpus['word'].shape[0] / batch
    
    predictions_tlabel, predictions_apos, predictions_alabel = [], [], []
    
    for ed in reModel.container['setZero']:
        reModel.container['setZero'][ed](reModel.container['zeroVecs'][ed])

    for i in range(numBatch):
        zippedCorpus = [ evaluateCorpus[ed][i*batch:(i+1)*batch] for ed in features if features[ed] >= 0 ]
            
        if skipByType: varPrefix = 'skipped_'
        else: varPrefix = ''
        zippedCorpus += [ evaluateCorpus[varPrefix + vant][i*batch:(i+1)*batch] for vant in reModel.classificationVariables ]
        
        pred = reModel.classify(*zippedCorpus)
        
        reModel.resetGlobalVariables()
        
        predictions_tlabel += [pred[0]]
        predictions_apos += [pred[1]]
        predictions_alabel += [pred[2]]
    
    predictions_tlabel = numpy.concatenate(predictions_tlabel, axis=0)
    predictions_apos = numpy.concatenate(predictions_apos, axis=0)
    predictions_alabel = numpy.concatenate(predictions_alabel, axis=0)
    
    if extra_data_num > 0:
        predictions_tlabel = predictions_tlabel[0:-extra_data_num]
        predictions_apos = predictions_apos[0:-extra_data_num]
        predictions_alabel = predictions_alabel[0:-extra_data_num]
    
    return predictions_tlabel, predictions_apos, predictions_alabel

def score(corpusName, predictions_tlabel, predictions_apos, predictions_alabel, corpus, idx2word, idx2triggerLabel, idx2argLabel, idMap, evaluation_output):

    fout = open(data_predictedFiles[corpusName], 'w')
    
    sidxs, swords, sentities = corpus['id'], corpus['word'], corpus['entities']
    for sid, sword, sentity, s_tlabel, s_apos, s_alabel in zip(sidxs, swords, sentities, predictions_tlabel, predictions_apos, predictions_alabel):
        fout.write(idMap[sid] + '\n')
        for wid, wor in enumerate(sword):
            if wor == 0: break
            fout.write(str(wid) + '\t' + idx2word[wor] + '\n')
        fout.write('--------Entity_Mention--------' + '\n')
        for eid in range(sentity[0]):
            fout.write(str(eid) + '\t' + str(sentity[eid+1]) + '\n')
        fout.write('--------Annotation--------' + '\n')
        
        if len(sword) != len(s_tlabel):
            print 'not matched lengths of words and tlabel'
            exit()
        
        for evid, _tlabel in enumerate(s_tlabel):
            if _tlabel == 0 or sword[evid] == 0: continue
            
            eprint = str(evid) + '\t' + idx2triggerLabel[_tlabel]
            
            _aposs = s_apos[evid]
            _alabels = s_alabel[evid]
            
            if len(_aposs) != len(_alabels):
                print 'not matched pos and argument label lengths'
                exit()
            
            for _apos, _alabel in zip(_aposs, _alabels):
                if _apos < 0: break
                eprint += '\t' + str(_apos) + '\t' + idx2argLabel[_alabel]
            
            fout.write(eprint + '\n')
        
        fout.write('\n')
    
    fout.close()
    
    performance = {}
    
    proc = subprocess.Popen([scoreScript, 'NNScorer', data_sourceDir, data_fileLists[corpusName], data_predictedFiles[corpusName], evaluation_output], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    
    ous, _ = proc.communicate()
    working = False
    identification = False
    for line in ous.split('\n'):
        line = line.strip()
        if line == '----RESULTS----':
            working = True
            continue
        if not working: continue
        if line == 'Identification:':
            identification = True
            continue
        
        if line.startswith('Trigger'):
            els = line.split('\t')
            pers = [els[2], els[4], els[6], els[9], els[11], els[13]]
            tf1, tpre, trec, af1, apre, arec = map(float, pers)
            per_prefix = 'identification-' if identification else ''
            performance[per_prefix + 'trigger'] = {'p' : tpre, 'r' : trec, 'f1' : tf1}
            performance[per_prefix + 'argument'] = {'p' : apre, 'r' : arec, 'f1' : af1}
    
    return performance

def train(model='basic',
          rep='gruBiDirect',
          skipByType=True,
          expected_features = OrderedDict([('pos', -1), ('chunk', -1), ('clause', -1), ('refer', -1), ('title', -1), ('posType', -1), ('dep', -1), ('typeEntity', -1), ('typeOneEntity', -1)]),
          distanceFet=-1,
          triggerGlob=-1,
          argGlob=-1,
          withEmbs=False, # using word embeddings to initialize the network or not
          updateEmbs=True,
          optimizer='adadelta',
          lr=0.01,
          dropoutTrigger=0.05,
          dropoutArg=0.05,
          regularizer=0.5,
          norm_lim = -1.0,
          verbose=1,
          decay=False,
          batch=50,
          winTrigger=-1,
          winArg=-1,
          multilayerTrigger=[1200, 600],
          multilayerArg=[1200, 600],
          multilayerTriggerAtt=[],
          multilayerArgAtt=[],
          multilayerArgExternal=[],
          nhidden=100,
          #nhiddenTrigger=100,
          #nhiddenArg=100,
          conv_feature_map=100,
          conv_win_feature_map=[2,3,4,5],
          seed=3435,
          #emb_dimension=300, # dimension of word embedding
          nepochs=50,
          folder='./res'):
          
    folder = '~/projects/jointEE/res/' + folder
    #folder = './res/storer'

    if not os.path.exists(folder): os.mkdir(folder)
    
    evaluation_output = folder
    for pcpu in data_predictedFiles: data_predictedFiles[pcpu] = folder + '/' + pcpu + '.predicted'

    print 'loading dataset: ', dataset_path, ' ...'
    revs, embeddings, dictionaries, eventEntityType, idMap = cPickle.load(open(dataset_path, 'rb'))
    
    idx2word = dict((k,v) for v,k in dictionaries['word'].iteritems())
    idx2triggerLabel = dict((k,v) for v,k in dictionaries['nodeLabel'].iteritems())
    idx2argLabel = dict((k,v) for v,k in dictionaries['edgeLabel'].iteritems())

    if not withEmbs:
        wordEmbs = embeddings['randomWord']
    else:
        print 'using word embeddings to initialize the network ...'
        wordEmbs = embeddings['word']
        
    emb_dimension = wordEmbs.shape[1]
    
    embs = {'word' : wordEmbs,
            'dist1' : embeddings['dist1'],
            'dist2' : embeddings['dist2'],
            'dist3' : embeddings['dist3'],
            'typeOneEntity' : embeddings['typeOneEntity'],
            'pos' : embeddings['pos'],
            'chunk' : embeddings['chunk'],
            'clause' : embeddings['clause'],
            'refer' : embeddings['refer'],
            'title' : embeddings['title'],
            'trigger' : embeddings['trigger'],
            'arg' : embeddings['arg']}
             
    expected_features['dep'] = 1 if expected_features['dep'] >= 0 else -1
    expected_features['typeEntity'] = 1 if expected_features['typeEntity'] >= 0 else -1
    expected_features['posType'] = 1 if expected_features['posType'] >= 0 else -1
    argGlob = 1 if argGlob >= 0 else -1
    
    #code for the current model only
    triggerGlob=-1
    argGlob=-1
    if distanceFet >= 0: distanceFet = 0
    ###

    features = OrderedDict([('word', 0)])

    for ffin in expected_features:
        features[ffin] = expected_features[ffin]
        if expected_features[ffin] == 0:
            print 'using feature: ', ffin, ' : embeddings'
        elif expected_features[ffin] == 1:
            print 'using feature: ', ffin, ' : binary'
        
    datasets, typeMap = make_data(revs, dictionaries, embeddings, features, eventEntityType, skipByType)
    
    dimCorpus = datasets['train']
    
    maxSentLength = len(dimCorpus['word'][0])
    maxNumEntities = len(dimCorpus['entities'][0])-1
    
    vocsize = len(idx2word)
    numTrigger = len(idx2triggerLabel)
    numArg = len(idx2argLabel)
    nsentences = len(dimCorpus['word'])

    print 'vocabsize = ', vocsize, ', numTrigger = ', numTrigger,  ', numArg = ', numArg, ', nsentences = ', nsentences, ', maxSentLength = ', maxSentLength, ', maxNumEntities = ', maxNumEntities, ', word embeddings dim = ', emb_dimension
    
    features_dim = OrderedDict([('word', emb_dimension)])
    for ffin in expected_features:
        if ffin in embs: cfdim = embs[ffin].shape[1]
        else: cfdim = -1
        features_dim[ffin] = ( len(dimCorpus[ffin][0][0]) if (features[ffin] == 1) else cfdim )
    
    #print '------- length of the instances: ', conv_winre
    
    params = {'model' : model,
              'rep' : rep,
              'nh' : nhidden,
              #'nht' : nhiddenTrigger,
              #'nha' : nhiddenArg,
              'numTrigger' : numTrigger,
              'numArg' : numArg,
              'maxSentLength': maxSentLength,
              'maxNumEntities': maxNumEntities,
              'ne' : vocsize,
              'batch' : batch,
              'embs' : embs,
              'dropoutTrigger' : dropoutTrigger,
              'dropoutArg' : dropoutArg,
              'regularizer': regularizer,
              'norm_lim' : norm_lim,
              'updateEmbs' : updateEmbs,
              'features' : features,
              'features_dim' : features_dim,
              'distanceFet': distanceFet,
              'distanceDim': embs['dist1'].shape[1] if distanceFet == 0 else embs['dist1'].shape[0]-1,
              'triggerGlob' : triggerGlob,
              'triggerDim': embs['trigger'].shape[1] if triggerGlob == 0 else embs['trigger'].shape[0]-1,
              'argGlob' : argGlob,
              'nodeFetDim' : len(dictionaries['nodeFetDict']),
              'edgeFetDim' : len(dictionaries['edgeFetDict']),
              'optimizer' : optimizer,
              'winTrigger' : winTrigger,
              'winArg': winArg,
              'multilayerTrigger' : multilayerTrigger,
              'multilayerArg' : multilayerArg,
              'multilayerTriggerAtt' : multilayerTriggerAtt,
              'multilayerArgAtt' : multilayerArgAtt,
              'multilayerArgExternal' : multilayerArgExternal,
              'conv_feature_map' : conv_feature_map,
              'conv_win_feature_map' : conv_win_feature_map}
    
    for corpus in datasets:
        for ed in datasets[corpus]:
            if ed in typeMap:
                dty = typeMap[ed]
            else:
                dty = 'float32' if numpy.array(datasets[corpus][ed][0]).ndim == 2 else 'int32'
            datasets[corpus][ed] = numpy.array(datasets[corpus][ed], dtype=dty)
    
    trainCorpus = {}
    augt = datasets['train']
    if nsentences % batch > 0:
        extra_data_num = batch - nsentences % batch
        for ed in augt:
            numpy.random.seed(3435)
            permuted = numpy.random.permutation(augt[ed])   
            extra_data = permuted[:extra_data_num]
            trainCorpus[ed] = numpy.append(augt[ed],extra_data,axis=0)
    else:
        for ed in augt:
            trainCorpus[ed] = augt[ed]
    
    number_batch = trainCorpus['word'].shape[0] / batch
    
    print '... number of batches: ', number_batch
    
    # instanciate the model
    print 'building model ...'
    numpy.random.seed(seed)
    random.seed(seed)
    reModel = eval('rnnJoint')(params)
    print 'done'
    
    evaluatingDataset = OrderedDict([#('train', datasets['train']),
                                     ('valid', datasets['valid']),
                                     ('test', datasets['test'])
                                    ])
    
    _perfs = OrderedDict()

    # training model
    best_f1 = -numpy.inf
    clr = lr
    s = OrderedDict()
    for e in xrange(nepochs):
        s['_ce'] = e
        tic = time.time()
        #nsentences = 5
        print '-------------------training in epoch: ', e, ' -------------------------------------'
        # for i in xrange(nsentences):
        miniId = -1
        for minibatch_index in numpy.random.permutation(range(number_batch)):
            miniId += 1
            trainIn = OrderedDict()
            for ed in features:
                if features[ed] >= 0:
                    if ed not in trainCorpus:
                        print 'cannot find data in train for: ', ed
                        exit()
                    
                    trainIn[ed] = trainCorpus[ed][minibatch_index*batch:(minibatch_index+1)*batch]
            
            zippedData = [ trainIn[ed] for ed in trainIn ]

            if skipByType: varPrefix = 'skipped_'
            else: varPrefix = ''
            zippedData += [ trainCorpus[varPrefix + vant][minibatch_index*batch:(minibatch_index+1)*batch] for vant in reModel.trainVariables ]
            for ed in reModel.container['setZero']:
                reModel.container['setZero'][ed](reModel.container['zeroVecs'][ed])
            
            reModel.f_grad_shared(*zippedData)
            reModel.f_update_param(clr)
            
            reModel.resetGlobalVariables()
                
            if verbose:
                if miniId % 10 == 0:
                    print 'epoch %i >> %2.2f%%'%(e,(miniId+1)*100./number_batch),'completed in %.2f (sec) <<'%(time.time()-tic)
                    sys.stdout.flush()

        # evaluation // back into the real world : idx -> words
        print 'evaluating in epoch: ', e
        for elu in evaluatingDataset:
            predictions_tlabel, predictions_apos, predictions_alabel = predict(evaluatingDataset[elu], batch, reModel, features, skipByType)
            _perfs[elu] = score(elu, predictions_tlabel, predictions_apos, predictions_alabel, evaluatingDataset[elu], idx2word, idx2triggerLabel, idx2argLabel, idMap, evaluation_output)

        perPrint(_perfs)
        
        if _perfs['valid']['argument']['f1'] > best_f1:
            #rnn.save(folder)
            best_f1 = _perfs['valid']['argument']['f1']
            print '*************NEW BEST: epoch: ', e
            if verbose:
                perPrint(_perfs, len('Current Performance')*'-')

            for elu in evaluatingDataset: s[elu] = _perfs[elu]
            s['_be'] = e
            
            subprocess.call(['mv', folder + '/test.predicted', folder + '/best.test.txt'])
            subprocess.call(['mv', folder + '/valid.predicted', folder + '/best.valid.txt'])
        else:
            print ''
        
        # learning rate decay if no improvement in 10 epochs
        if decay and abs(s['_be']-s['_ce']) >= 10: clr *= 0.5 
        if clr < 1e-5: break

    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
    print 'BEST RESULT: epoch: ', s['_be']
    perPrint(s, len('Current Performance')*'-')
    print ' with the model in ', folder

def perPrint(perfs, mess='Current Performance'):
    order = ['identification-trigger', 'identification-argument', 'trigger', 'argument']
    print '------------------------------%s-----------------------------'%mess
    for elu in perfs:
        if elu.startswith('_'): continue
        print '***** ' + elu + ' *****'
        for od in order:
            pri = od + ' : ' + str(perfs[elu][od]['p']) + '\t' + str(perfs[elu][od]['r'])+ '\t' + str(perfs[elu][od]['f1'])
            print pri
    
    print '------------------------------------------------------------------------------'
    
if __name__ == '__main__':
    pass
