import numpy as np
import cPickle
from collections import defaultdict, OrderedDict
import sys, re
import pandas as pd
import random

#thien's version
extra = 5
maximumLen = 70
fetCutoff = 1

def build_data(data_file, data_list):
    """
    Loads data.
    """
    nodeDict = {'NONE':0}
    edgeDict = {'NONE':0}
    
    etypeDict = {'NONE':0}
    esubtypeDict = {'NONE':0}
    
    vocab = defaultdict(float)
    
    depRelDict = {'NONE':1}
    typeDict = {'NONE':1}
    typeOneDict = {'NONE':1}
    posDict = {}
    chunkDict = {'O':1}
    clauseDict = {}
    referDict = {'false':1}
    titleModifierDict = {'false':1}
    possibleNodeDict = {'NONE':1}
    
    nodeFetDict = {'':0}
    edgeFetDict = {'':0}
    
    nodeFetCounter = defaultdict(int)
    edgeFetCounter = defaultdict(int)
    
    revs = []
    
    corpusCountIns = defaultdict(int)
    maxLength = -1
    lengthCounter = defaultdict(int)
    tooLong = 0
    idMap = {}
    
    corpusMap = loadCorpusMap(data_list)
    
    inst = []
    entId, edgeId, annId = -1, -1, -1
    
    idid = -1
    
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line:
                inst += [line]
                
                if line == '--------Entity_Mention--------': entId = len(inst)
                if line == '--------Edge_Features--------': edgeId = len(inst)
                if line == '--------Annotation--------': annId = len(inst)
                
                continue
            
            id = inst[0]
            docId = id[(id.find('=')+1):]
            docId = docId[0:docId.rfind('#')]
            if docId not in corpusMap:
                print 'cannot find ', docId, ' in corpusMap'
                exit()
            corpus = corpusMap[docId]
            
            sentence, pos, chunk, clause, posType, grs, ets, ref, title, eligible, nodeFets, entities, edgeFets, eventPos, eventTrigger, eventArgs = parseInst(inst, entId, edgeId, annId)
            
            inst = []
            
            if len(sentence) > maximumLen:
                tooLong += 1
                continue
            if not eventPos and corpus == 'train': continue
            
            entId, annId = -1, -1
            
            for i, trigger in enumerate(eventTrigger):
                lookup('trigger', trigger, nodeDict, False)
                eventTrigger[i] = nodeDict[trigger]
                
                for arg_pos in eventArgs[i]:
                    arg_label = eventArgs[i][arg_pos]
                    lookup('argument', arg_label, edgeDict, False)
                    eventArgs[i][arg_pos] = edgeDict[arg_label]
            
            for i, entity in enumerate(entities):
                etype = entity[4]
                lookup('entityType', etype, etypeDict, False)
                entities[i][4] = etypeDict[etype]
                
                esubtype = entity[5]
                lookup('entitySubType', esubtype, esubtypeDict, False)
                entities[i][5] = esubtypeDict[esubtype]
                
            words = set(sentence)
            for word in words:
                #word = ' '.join(word.split('_'))
                vocab[word] += 1
            
            for i, pos_i in enumerate(pos):
                lookup('POS', pos_i, posDict, True)
                pos[i] = posDict[pos_i]
            
            for i, chunk_i in enumerate(chunk):
                lookup('CHUNK', chunk_i, chunkDict, True)
                chunk[i] = chunkDict[chunk_i]
            
            for i, clause_i in enumerate(clause):
                clauseDict[clause_i] = int(clause_i)
                clause[i] = int(clause_i)
            
            for pts in posType:
                for pt in pts:
                    lookup('possibleTriggerType', pt, possibleNodeDict, True)
            nposType = []
            for pts in posType:
                npt = [ possibleNodeDict[pt] for pt in pts ]
                nposType += [npt]
            posType = nposType
            
            for gs in grs:
                for g in gs:
                    lookup('depRelType', g, depRelDict, True)
            nngs = []
            for gs in grs:
                nng = [ depRelDict[g] for g in gs ]
                nngs += [nng]
            grs = nngs
            
            oneEts = []
            for et in ets: oneEts += [et[0]]
            for i, oneEts_i in enumerate(oneEts):
                lookup('entityOneTypeSequence', oneEts_i, typeOneDict, True)
                oneEts[i] = typeOneDict[oneEts_i]
            
            for et in ets:
                for e in et:
                    lookup('entityTypeSequence', e, typeDict, True)
            nets = []
            for et in ets:
                net = [ typeDict[e] for e in et ]
                nets += [net]
            ets = nets
            
            for i, ref_i in enumerate(ref):
                lookup('REFERENCE', ref_i, referDict, True)
                ref[i] = referDict[ref_i]
            
            for i, title_i in enumerate(title):
                lookup('TITLE', title_i, titleModifierDict, True)
                title[i] = titleModifierDict[title_i]
            
            for nfs in nodeFets:
                for nf in nfs:
                    if not nf: continue
                    nodeFetCounter[nf] += 1
            
            for eefs in edgeFets:
                for wefs in eefs:
                    for ef in wefs:
                        if not ef: continue
                        edgeFetCounter[ef] += 1
                
            if len(sentence) > maxLength:
                maxLength = len(sentence)
                    
            lengthCounter[len(sentence)] += 1
            
            corpusCountIns[corpus] += 1
            
            idid += 1
            idMap[idid] = id
            
            datum = {"id": idid,
                         
                     "text": sentence,
                     "pos": pos,
                     "chunk": chunk,
                     "clause": clause,
                     "posType": posType,
                     "dep": grs,
                     "typeEntity": ets,
                     "typeOneEntity": oneEts,
                     "refer": ref,
                     "title": title,
                     "eligible": eligible,
                     "nodeFets": nodeFets,
                     
                     "entities": entities,
                     "edgeFets": edgeFets,
                     
                     "eventPos": eventPos,
                     "eventTrigger": eventTrigger,
                     "eventArgs": eventArgs,
                     
                     "corpus": corpus}
            revs.append(datum)
    
    for mf in nodeFetCounter:
        if nodeFetCounter[mf] >= fetCutoff:
            nodeFetDict[mf] = len(nodeFetDict)
    for mf in edgeFetCounter:
        if edgeFetCounter[mf] >= fetCutoff:
            edgeFetDict[mf] = len(edgeFetDict)
    
    for rev in revs:
        nnodeFets = []
        for nfs in rev["nodeFets"]:
            nnfs = [ nodeFetDict[nf] for nf in nfs if nf in nodeFetDict ]
            nnodeFets += [nnfs]
        rev["nodeFets"] = nnodeFets
        
        nedgeFets = []
        for eefs in rev["edgeFets"]:
            neefs = []
            for wefs in eefs:
                nwefs = [ edgeFetDict[ef] for ef in wefs if ef in edgeFetDict ]
                neefs += [nwefs]
            nedgeFets += [neefs]
        rev["edgeFets"] = nedgeFets
    
    print 'instances in corpus'
    for corpus in corpusCountIns:
    	print corpus, ' : ', corpusCountIns[corpus]
    print '---------------'
    print 'length distribution'
    for le in lengthCounter:
    	print le, ' : ', lengthCounter[le]
    print '---------------'
    print "maximum length of sentences: ", maxLength
    print "number of too long: ", tooLong
    print '----------------'
    print 'total node features: ', len(nodeFetDict)
    print 'total edge features: ', len(edgeFetDict)
    
    return idMap, maxLength, revs, vocab, nodeDict, edgeDict, etypeDict, esubtypeDict, depRelDict, typeDict, typeOneDict, posDict, chunkDict, clauseDict, referDict, titleModifierDict, possibleNodeDict, nodeFetDict, edgeFetDict

def lookup(mess, key, gdict, addOne):
    if key not in gdict:
        nk = len(gdict)
        if addOne: nk += 1
        gdict[key] = nk
        if mess: print mess, ': ', key, ' --> id = ', gdict[key]

def loadCorpusMap(data_list):
    print 'loading corpusMap ...'
    res = {}
    for dl in data_list:
        with open(data_list[dl], 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                res[line] = dl
    print 'loaded: ', len(res), ' files'
    return res

def parseInst(inst, entId, edgeId, annId):
    
    sentence, pos, chunk, clause, posType, grs, ets, ref, title, eligible, nodeFets = [], [], [], [], [], [], [], [], [], [], []
    
    for line in inst[1:entId-1]:
        tokens = line.split('\t')
        if len(tokens) != 16:
            print 'not have 16 elements: ', line
            exit()
        
        sentence += [tokens[1]]
        pos += [tokens[3]]
        chunk += [tokens[4]]
        clause += [tokens[6]]
        posType += [tokens[7].split()]
        grs += [tokens[10].split()]
        #ets += [tokens[11].split()]
        ets += [[tokens[11].split()[0]]]
        ref += [tokens[12]]
        title += [tokens[13]]
        eligible += [int(tokens[14])]
        nodeFets += [tokens[15].split()]
    psentLen = len(sentence)
    
    entities = []
    for line in inst[entId:edgeId-1]:
        mentions = line.split('\t')
        if len(mentions) != 7 and len(mentions) != 8:
            print 'not 7 or 8 elements'
            exit()        
        entities += [[int(mentions[1]), int(mentions[2]), int(mentions[3]), int(mentions[4]), mentions[5], mentions[6]]]
    pnumEntities = len(entities)
        
    edgeFets = []
    for lid in range(pnumEntities):
        leid = edgeId + lid*(1+psentLen)
        if int(inst[leid]) != lid:
            print 'wrong entity id: ', leid, inst[leid]
            exit()
        oneWordEdgeFets = []
        for sid in range(1, 1+psentLen):
            lsid = leid + sid
            edgeEls = inst[lsid].split('\t')
            if len(edgeEls) != 2 or int(edgeEls[0]) != (sid-1):
                print 'wrong token id: ', lsid, inst[lsid]
                exit()
            oneWordEdgeFets += [edgeEls[1].split()]
        edgeFets += [oneWordEdgeFets]
    
    if (edgeId + pnumEntities*(1+psentLen)) != (annId-1):
        print 'wrong positions for annotation and edge features: ', edgeId + pnumEntities*(1+psentLen), annId-1
        exit()
        
    eventPos, eventTrigger, eventArgs = [], [], []
    for line in inst[annId:]:
        event = line.split('\t')
        eventPos += [int(event[0])]
        eventTrigger += [event[1]]
        
        argm = {}
        for i in range(1,(len(event)/2)):
            argm[int(event[2*i])] = event[2*i+1]
        argm_sorted = sorted(argm)
        ars = OrderedDict()
        for eid in argm_sorted:
            ars[eid] = argm[eid]
        eventArgs += [ars]
    
    return sentence, pos, chunk, clause, posType, grs, ets, ref, title, eligible, nodeFets, entities, edgeFets, eventPos, eventTrigger, eventArgs

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k))
    W[0] = np.zeros(k)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    dim = 0
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
               dim = word_vecs[word].shape[0]
            else:
                f.read(binary_len)
    print 'dim: ', dim
    return dim, word_vecs
    
def load_text_vec(fname, vocab):
    word_vecs = {}
    count = 0
    dim = 0
    with open(fname, 'r') as f:
        for line in f:
            count += 1
            line = line.strip()
            if count == 1:
                if len(line.split()) < 10:
                    dim = int(line.split()[1])
                    print 'dim: ', dim
                    continue
                else:
                    dim = len(line.split()) - 1
                    print 'dim: ', dim
            word = line.split()[0]
            emStr = line[(line.find(' ')+1):]
            if word in vocab:
                word_vecs[word] = np.fromstring(emStr, dtype='float32', sep=' ')
                if word_vecs[word].shape[0] != dim:
                    print 'mismatch dimensions: ', dim, word_vecs[word].shape[0]
                    exit()
    print 'loaded ', len(word_vecs), ' words in word embeddings'
    return dim, word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)

def loadEventEntityType(file, nodeDict):
    res = {}
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            els = line.split('\t')
            ev = els[0]
            if ev not in nodeDict:
                print 'cannot find event type: ', ev, ' in nodeDict'
                exit()
            res[nodeDict[ev]] = els[1:]
    return res

if __name__=="__main__":
    np.random.seed(3435)
    random.seed(3435)
    embType = sys.argv[1]
    w2v_file = sys.argv[2]
    data_file = sys.argv[3]
    srcDir = sys.argv[4]
    eventEntityTypeFile = sys.argv[5]
    
    dataCorpus = ["train", "valid", "test"]
    data_list = {}
    for d in dataCorpus: data_list[d] = srcDir + "/" + d + ".txt"
    
    print "loading data...\n"
    idMap, maxLength, revs, vocab, nodeDict, edgeDict, etypeDict, esubtypeDict, depRelDict, typeDict, typeOneDict, posDict, chunkDict, clauseDict, referDict, titleModifierDict, possibleNodeDict, nodeFetDict, edgeFetDict = build_data(data_file, data_list)
    
    eventEntityType = loadEventEntityType(eventEntityTypeFile, nodeDict)
    
    #print "max distance between entities: " + str(maxDist)
    print "data loaded!"
    print "vocab size: " + str(len(vocab))
    print "loading word embeddings...",
    dimEmb = 300
    if embType == 'word2vec':
    	dimEmb, w2v = load_bin_vec(w2v_file, vocab)
    else:
    	dimEmb, w2v = load_text_vec(w2v_file, vocab)
    print "word embeddings loaded!"
    print "num words already in word embeddings: " + str(len(w2v))
    add_unknown_words(w2v, vocab, 1, dimEmb)
    W1, word_idx_map = get_W(w2v, dimEmb)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab, 1, dimEmb)
    W2, _ = get_W(rand_vecs, dimEmb)
    
    dictionaries = {}
    dictionaries['word'] = word_idx_map
    dictionaries['nodeLabel'] = nodeDict
    dictionaries['edgeLabel'] = edgeDict
    dictionaries['etype'] = etypeDict
    dictionaries['esubtype'] = esubtypeDict
    dictionaries['dep'] = depRelDict
    dictionaries['typeEntity'] = typeDict
    dictionaries['typeOneEntity'] = typeOneDict
    dictionaries['pos'] = posDict
    dictionaries['chunk'] = chunkDict
    dictionaries['clause'] = clauseDict
    dictionaries['refer'] = referDict
    dictionaries['title'] = titleModifierDict
    dictionaries['possibleNode'] = possibleNodeDict
    dictionaries['nodeFetDict'] = nodeFetDict
    dictionaries['edgeFetDict'] = edgeFetDict
    
    embeddings = {}
    
    dist_size = 2*maxLength - 1
    dist_dim = 50
    D1 = np.random.uniform(-0.25,0.25,(dist_size+1,dist_dim))
    D2 = np.random.uniform(-0.25,0.25,(dist_size+1,dist_dim))
    D3 = np.random.uniform(-0.25,0.25,(dist_size+1,dist_dim))
    D1[0] = np.zeros(dist_dim)
    D2[0] = np.zeros(dist_dim)
    D3[0] = np.zeros(dist_dim)
    
    type_dim = 50
    TYPE = np.random.uniform(-0.25,0.25,(len(typeOneDict)+1,type_dim))
    TYPE[0] = np.zeros(type_dim)
    
    pos_dim = 50
    POS = np.random.uniform(-0.25,0.25,(len(posDict)+1,pos_dim))
    POS[0] = np.zeros(pos_dim)
    
    chunk_dim = 50
    CHUNK = np.random.uniform(-0.25,0.25,(len(chunkDict)+1,chunk_dim))
    CHUNK[0] = np.zeros(chunk_dim)
    
    clause_dim = 50
    CLAUSE = np.random.uniform(-0.25,0.25,(len(clauseDict)+1,clause_dim))
    CLAUSE[0] = np.zeros(clause_dim)
    
    refer_dim = 50
    REFER = np.random.uniform(-0.25,0.25,(2+1,refer_dim))
    REFER[0] = np.zeros(refer_dim)
    
    title_dim = 50
    TITLE = np.random.uniform(-0.25,0.25,(2+1,title_dim))
    TITLE[0] = np.zeros(title_dim)
    
    trigger_dim = 50
    TRIGGER = np.random.uniform(-0.25,0.25,(len(nodeDict)+1,trigger_dim))
    TRIGGER[0] = np.zeros(trigger_dim)
    
    arg_dim = 50
    ARG = np.random.uniform(-0.25,0.25,(len(edgeDict)+1,arg_dim))
    ARG[0] = np.zeros(arg_dim)
    
    embeddings['word'] = W1
    embeddings['randomWord'] = W2
    embeddings['dist1'] = D1
    embeddings['dist2'] = D2
    embeddings['dist3'] = D3
    embeddings['typeOneEntity'] = TYPE
    embeddings['pos'] = POS
    embeddings['chunk'] = CHUNK
    embeddings['clause'] = CLAUSE
    embeddings['refer'] = REFER
    embeddings['title'] = TITLE
    embeddings['trigger'] = TRIGGER
    embeddings['arg'] = ARG
    
    for di in dictionaries:
        print 'size of ', di, ': ', len(dictionaries[di])
    
    print 'dumping ...'
    cPickle.dump([revs, embeddings, dictionaries, eventEntityType, idMap], open('cut_' + str(fetCutoff) + '.' + embType + "_jointEE.pkl", "wb"))
    print "dataset created!"   
