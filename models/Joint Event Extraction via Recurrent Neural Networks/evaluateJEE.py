from jointEE import train
from collections import OrderedDict

def main(params):
    print params
    train(model = params['model'],
          rep = params['rep'],
          skipByType = params['skipByType'],
          expected_features = params['expected_features'],
          distanceFet = params['distanceFet'],
          triggerGlob = params['triggerGlob'],
          argGlob = params['argGlob'],
          winTrigger = params['winTrigger'],
          winArg = params['winArg'],
          withEmbs = params['withEmbs'],
          updateEmbs = params['updateEmbs'],
          optimizer = params['optimizer'],
          lr = params['lr'],
          dropoutTrigger = params['dropoutTrigger'],
          dropoutArg = params['dropoutArg'],
          regularizer = params['regularizer'],
          norm_lim = params['norm_lim'],
          verbose = params['verbose'],
          decay = params['decay'],
          batch = params['batch'],
          multilayerTrigger = params['multilayerTrigger'],
          multilayerArg = params['multilayerArg'],
          multilayerTriggerAtt = params['multilayerTriggerAtt'],
          multilayerArgAtt = params['multilayerArgAtt'],
          multilayerArgExternal = params['multilayerArgExternal'],
          nhidden = params['nhidden'],
          conv_feature_map = params['conv_feature_map'],
          conv_win_feature_map = params['conv_win_feature_map'],
          seed = params['seed'],
          #emb_dimension=300, # dimension of word embedding
          nepochs = params['nepochs'],
          folder = params['folder'])
def fetStr(ef):
    res = ''
    for f in ef:
        res += str(ef[f])
    return res

def fmStr(ft):
    res = ''
    for f in ft:
        res += str(f) + ' '
    return res.strip().replace(' ', '_')

if __name__=='__main__':
    pars={'model' : 'basic',
          'rep' : 'gruBiDirect', # gruBiDirect, gruForward, gruBackward, ffBiDirect, ffForward, ffBackward
          'skipByType' : True,
          'expected_features' : OrderedDict([('pos', -1),
                                             ('chunk', -1),
                                             ('clause', -1),
                                             ('refer', -1),
                                             ('title', -1),
                                             ('posType', -1),
                                             ('dep', 1),
                                             ('typeEntity', -1),
                                             ('typeOneEntity', 0)]),
                                              
          'distanceFet' : -1,
          'triggerGlob' : -1,
          'argGlob' : -1,
          'winTrigger' : 2,
          'winArg' : 2,
          'withEmbs' : True,
          'updateEmbs' : True,
          'optimizer' : 'adadelta',
          'lr' : 0.01,
          'dropoutTrigger' : 0.0,
          'dropoutArg' : 0.0,
          'regularizer' : 0.0,
          'norm_lim' : 9.0,
          'verbose' : 1,
          'decay' : False,
          'batch' : 50,
          'multilayerTrigger' : [600],
          'multilayerArg' : [600],
          'multilayerTriggerAtt' : [],
          'multilayerArgAtt' : [],
          'multilayerArgExternal' : [300],
          'nhidden' : 300,
          'conv_feature_map' : 150,
          'conv_win_feature_map' : [2,3,4,5],
          'seed' : 3435,
          'nepochs' : 20,
          'folder' : './res'}
    folder = 'model_' + pars['model'] \
             + '.rep_' + pars['rep'] \
             + '.skip_' + ('1' if pars['skipByType'] else '0') \
             + '.h_' + str(pars['nhidden']) \
             + '.wt_' + str(pars['winTrigger']) \
             + '.wa_' + str(pars['winArg']) \
             + '.emb_' + ('1' if pars['withEmbs'] else '0') \
             + '.upd_' + ('1' if pars['updateEmbs'] else '0') \
             + '.bat_' + str(pars['batch']) \
             + '.mulT_' + fmStr(pars['multilayerTrigger']) \
             + '.mulA_' + fmStr(pars['multilayerArg']) \
             + '.mulTA_' + fmStr(pars['multilayerTriggerAtt']) \
             + '.mulAA_' + fmStr(pars['multilayerArgAtt']) \
             + '.mulAE' + fmStr(pars['multilayerArgExternal']) \
             + '.opt_' + pars['optimizer'] \
             + '.drt_' + str(pars['dropoutTrigger']) \
             + '.dra_' + str(pars['dropoutArg']) \
             + '.fet_' + fetStr(pars['expected_features']) \
             + '.dif_' + str(pars['distanceFet']) \
             + '.tg_' + str(pars['triggerGlob']) \
             + '.ag_' + str(pars['argGlob']) \
             + '.cvft_' + str(pars['conv_feature_map']) \
             + '.cvfm_' + fmStr(pars['conv_win_feature_map']) \
             + '.lr_' + str(pars['lr']) \
             + '.nrm_' + str(pars['norm_lim'])
    pars['folder'] =  'NoWin.concat.A-GlobTri.' + folder
    main(pars)
