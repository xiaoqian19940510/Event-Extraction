import os
import math

from model import *

from io_utils import read_pickle, write_lines, read_lines
from vocab import Vocab


class Trainer(object):

    def __init__(self):
        logger.info('Loading data...')
        self.train_list, self.dev_list, self.test_list, self.token_vocab, self.char_vocab, self.ent_type_vocab, \
        self.ent_ref_vocab, self.tri_type_vocab, self.arg_type_vocab, self.action_vocab, \
        self.pos_vocab = read_pickle(data_config['inst_pl_file'])
        logger.info('Sent size train: %d, dev: %d, test:%d'%(len(self.train_list), len(self.dev_list), len(self.test_list)))
        logger.info('Loading pretrained from: %s'%data_config['vec_npy'])
        pretrained_vec = np.load(data_config['vec_npy'])

        self.unk_idx = self.token_vocab[Vocab.UNK]
        joint_config['n_chars'] = len(self.char_vocab)
        self.trans_model = MainModel(self.token_vocab.get_vocab_size(),
                                       self.action_vocab,
                                       self.ent_type_vocab,
                                       self.tri_type_vocab,
                                       self.arg_type_vocab,
                                       self.pos_vocab,
                                       pretrained_vec=pretrained_vec)
        logger.info("Model:%s"%type(self.trans_model))

        self.event_eval = EventEval()


        # model = pm.global_collection()
        # for param in model.parameters_list():
        #     print(param.name(), param.dim())
        #
        # for param in model.lookup_parameters_list():
        #     print(param.name(), param.dim())


    def unk_replace_singleton(self, unk_idx, unk_ratio, words):
        noise = words[:]
        bernoulli = np.random.binomial(n=1, p=unk_ratio, size=len(words))
        for i, idx in enumerate(words):
            if self.token_vocab.is_singleton(idx) and bernoulli[i] == 1:
                noise[i] = unk_idx
        return noise


    def iter_batch(self, inst_list, shuffle=True):
        batch_size = joint_config['batch_size']
        if shuffle:
            random.shuffle(inst_list)
        inst_len = len(inst_list)
        plus_n = 0 if (inst_len % batch_size) == 0 else 1
        num_batch = (len(inst_list) // batch_size) + plus_n

        start = 0
        for i in range(num_batch):
            batch_inst = inst_list[start: start + batch_size]
            start += batch_size
            yield batch_inst


    def train_batch(self):
        loss_all = 0.
        batch_num = 0
        for batch_inst in self.iter_batch(self.train_list, shuffle=True):
            dy.renew_cg()
            loss_minibatch = []

            for inst in batch_inst:
                words = inst['word_indices']
                if joint_config['unk_ratio'] > 0:
                    words = self.unk_replace_singleton(self.unk_idx, joint_config['unk_ratio'], words)
                loss_rep = self.trans_model(words, inst['char_indices'], inst['action_indices'],
                                            inst['actions'], inst['tri_indices'], inst['ent_indices'],
                                            inst['arg_indices'], inst['sent_range'],
                                            inst['pos_indices'])
                loss_minibatch.append(loss_rep)

            batch_loss = dy.esum(loss_minibatch) / len(loss_minibatch)
            loss_all += batch_loss.value()
            batch_loss.backward()
            self.trans_model.update()
            batch_num += 1

        logger.info('loss %.5f ' % (loss_all / float(batch_num)))

    def eval(self, inst_list, write_ent_file=None, is_write_ent=False, mtype='dev'):
        #logger.info('~~~~~~Empty buffer times %d'%self.trans_model.shift_reduce.empty_times)
        self.event_eval.reset()
        sent_num_eval = 0
        total_loss = 0.

        ent_lines = []
        eval_lines = []
        for inst in inst_list:
            _, pred_ents, pred_tris, pred_args = self.trans_model.decode(
                                            inst['word_indices'], inst['char_indices'], inst['action_indices'],
                                            inst['actions'], inst['tri_indices'], inst['ent_indices'],
                                            inst['arg_indices'], inst['sent_range'],
                                            inst['pos_indices'], mtype=mtype)

            # total_loss += loss
            self.event_eval.update(pred_ents, inst['ent_indices'],
                                   pred_tris, inst['tri_indices'],
                                   pred_args, inst['arg_indices'],
                                   eval_arg=True, words=inst['nlp_words'])

            ent_line = str(sent_num_eval) + ' '
            ent_str_list = []
            for ent in pred_ents:
                ent_str_list.append(str(ent[0])+'#'+str(ent[1])+'#'+self.ent_type_vocab.get_token(ent[2]))
            ent_line += ','.join(ent_str_list)
            ent_lines.append(ent_line)
            sent_num_eval += 1

        #     eval_lines.append(' '.join(inst['nlp_words']))
        #     eval_lines.append(str(pred_ents))
        #     eval_lines.append(str(pred_tris))
        #     eval_lines.append(str(pred_args))
        # write_lines('data_files/dev_test_prediction/test_pred.txt',eval_lines)

        if write_ent_file is not None and is_write_ent:
            write_lines(write_ent_file, ent_lines)
        # file.flush()

        #logger.info('~~~~~~Empty buffer times %d' % self.trans_model.shift_reduce.empty_times)


    def train(self, save_model=True):
        logger.info(joint_config['msg_info'])
        best_f1_tri_typed, best_f1_arg_typed = 0, 0
        best_epoch = 0

        adjust_lr = False
        stop_patience = joint_config['patience']
        stop_count = 0
        eval_best_arg = True # for other task than event
        t_cur = 1
        t_i = 4
        t_mul = 2
        lr_max, lr_min = joint_config['init_lr'], joint_config['minimum_lr']
        for epoch in range(joint_config['num_epochs']):
            anneal_lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t_cur / t_i))
            self.trans_model.set_lr(anneal_lr)

            logger.info('--------------------------------------')
            logger.info('Epoch : %d'%epoch)
            logger.info('LR : %.5f' % self.trans_model.get_lr())

            self.train_batch()

            self.eval(self.dev_list, mtype='dev')
            (p_ent, r_ent, f_ent), (p_ent_typed, r_ent_typed, f_ent_typed), \
            (p_tri, r_tri, f_tri), (p_tri_typed, r_tri_typed, f_tri_typed), \
            (p_arg, r_arg, f_arg), (p_arg_typed, r_arg_typed, f_arg_typed) = self.event_eval.report()

            logger.info('Entity         P:%.5f, R:%.5f, F:%.5f' % (p_ent, r_ent, f_ent))
            logger.info('Entity typed   P:%.5f, R:%.5f, F:%.5f' % (p_ent_typed, r_ent_typed, f_ent_typed))
            logger.info('Trigger        P:%.5f, R:%.5f, F:%.5f' % (p_tri, r_tri, f_tri))
            logger.info('Trigger typed  P:%.5f, R:%.5f, F:%.5f' % (p_tri_typed, r_tri_typed, f_tri_typed))
            logger.info('Argument       P:%.5f, R:%.5f, F:%.5f' % (p_arg, r_arg, f_arg))
            logger.info('Argument typed P:%.5f, R:%.5f, F:%.5f' % (p_arg_typed, r_arg_typed, f_arg_typed))

            if t_cur == t_i:
                t_cur = 0
                t_i *= t_mul

            t_cur += 1

            if not eval_best_arg:
                continue


            if f_arg_typed >= best_f1_arg_typed :
                best_f1_arg_typed = f_arg_typed
                best_f1_tri_typed = f_tri_typed
                best_epoch = epoch

                stop_count = 0
                #logger.info('Saving model %s'%model_save_file)
                #self.trans_model.save_model(model_save_file)

                if save_model:
                    logger.info('Saving model %s' % data_config['model_save_file'])
                    self.trans_model.save_model(data_config['model_save_file'])

            else:
                stop_count += 1
                if stop_count >= stop_patience:
                    logger.info('Stop training, Arg performance did not improved for %d epochs'%stop_count)
                    break

                if adjust_lr:
                    self.trans_model.decay_lr(joint_config['decay_lr'])
                    logger.info('@@@@  Adjusting LR: %.5f  @@@@@@' % self.trans_model.get_lr())

                if self.trans_model.get_lr() < joint_config['minimum_lr']:
                    adjust_lr = False

            # if epoch > 0 and epoch % 6 == 0:
            #     self.trans_model.decay_lr(joint_config['decay_lr'])
            #     logger.info('@@@@  Adjusting LR: %.5f  @@@@@@' % self.trans_model.get_lr())

            best_msg = '*****Best epoch: %d Tri and Arg typed F:%.5f, F:%.5f ******' % (best_epoch,
                                                                                    best_f1_tri_typed,
                                                                                    best_f1_arg_typed)
            logger.info(best_msg)

        return best_msg, best_f1_arg_typed

    def test(self, fname):
        self.trans_model.load_model(fname)
        self.eval(self.test_list, mtype='test')
        (p_ent, r_ent, f_ent), (p_ent_typed, r_ent_typed, f_ent_typed), \
        (p_tri, r_tri, f_tri), (p_tri_typed, r_tri_typed, f_tri_typed), \
        (p_arg, r_arg, f_arg), (p_arg_typed, r_arg_typed, f_arg_typed) = self.event_eval.report()

        logger.info('Entity         P:%.5f, R:%.5f, F:%.5f' % (p_ent, r_ent, f_ent))
        logger.info('Entity typed   P:%.5f, R:%.5f, F:%.5f' % (p_ent_typed, r_ent_typed, f_ent_typed))
        logger.info('Trigger        P:%.5f, R:%.5f, F:%.5f' % (p_tri, r_tri, f_tri))
        logger.info('Trigger typed  P:%.5f, R:%.5f, F:%.5f' % (p_tri_typed, r_tri_typed, f_tri_typed))
        logger.info('Argument       P:%.5f, R:%.5f, F:%.5f' % (p_arg, r_arg, f_arg))
        logger.info('Argument typed P:%.5f, R:%.5f, F:%.5f' % (p_arg_typed, r_arg_typed, f_arg_typed))


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train(save_model=True)

    logger.info('---------------Test Results---------------')
    ckp_path = data_config['model_save_file']
    trainer.test(ckp_path)
