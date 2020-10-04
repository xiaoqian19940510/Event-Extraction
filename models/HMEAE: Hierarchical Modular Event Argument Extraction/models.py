import tensorflow as tf
import constant
from func import get_batch,get_trigger_feeddict,f_score,get_argument_feeddict
import numpy as np
class DMCNN():
    def __init__(self,t_data,a_data,maxlen,max_argument_len,wordemb,stage="trigger",classify='single'):
        self.t_train,self.t_dev,self.t_test = t_data
        self.a_train,self.a_dev,self.a_test = a_data
        self.maxlen = maxlen
        self.wordemb = wordemb
        self.stage = stage
        self.max_argument_len = max_argument_len
        self.classify = classify
        self.build_graph()
    
    def build_graph(self):
        if self.stage=='trigger':
            print('--Building Trigger Graph--')
            self.build_trigger()
        elif self.stage=="DMCNN":
            print('--Building DMCNN Graph--')
            self.build_argument()
        elif self.stage=="HMEAE":
            print('--Building HMEAE Graph--')
            self.build_HMEAE()
        else:
            raise ValueError("stage could only be trigger or DMCNN or HMEAE")

    def build_trigger(self,scope='DMCNN_Trigger'):
        maxlen = self.maxlen
        num_class = len(constant.EVENT_TYPE_TO_ID)
        keepprob = constant.t_keepprob
        with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
            with tf.variable_scope('Initialize'):
                posi_mat = tf.concat(
                            [tf.zeros([1,constant.posi_embedding_dim],tf.float32),
                            tf.get_variable('posi_emb',[2*maxlen,constant.posi_embedding_dim],tf.float32,initializer=tf.contrib.layers.xavier_initializer())],axis=0)
                word_mat = tf.concat([
                            tf.zeros((1, constant.embedding_dim),dtype=tf.float32),
                            tf.get_variable("unk_word_embedding", [1, constant.embedding_dim], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer()),
                            tf.get_variable("wordemb", initializer=self.wordemb,trainable=True)], axis=0)

            with tf.variable_scope('placeholder'):
                self.sents = sents = tf.placeholder(tf.int32,[None,maxlen],'sents')
                self.posis = posis = tf.placeholder(tf.int32,[None,maxlen],'posis')
                self.maskls = maskls = tf.placeholder(tf.float32,[None,maxlen],'maskls')
                self.maskrs = maskrs = tf.placeholder(tf.float32,[None,maxlen],'maskrs')
                self._labels = _labels = tf.placeholder(tf.int32,[None],'labels')
                labels = tf.one_hot(_labels,num_class)
                self.is_train = is_train = tf.placeholder(tf.bool,[],'is_train')
                self.lexical = lexical = tf.placeholder(tf.int32,[None,3],'lexicals')

                sents_len = tf.reduce_sum(tf.cast(tf.cast(sents,tf.bool),tf.int32),axis=1)
                sents_mask = tf.expand_dims(tf.sequence_mask(sents_len,maxlen,tf.float32),axis=2)
            with tf.variable_scope('embedding'):
                sents_emb = tf.nn.embedding_lookup(word_mat,sents)
                posis_emb  = tf.nn.embedding_lookup(posi_mat,posis)
                lexical_emb = tf.nn.embedding_lookup(word_mat,lexical)
            with tf.variable_scope('lexical_feature'):
                lexical_feature = tf.reshape(lexical_emb,[-1,3*constant.embedding_dim])
            with tf.variable_scope('encoder'):
                emb = tf.concat([sents_emb,posis_emb],axis=2)
                emb_shape = tf.shape(emb)
                pad = tf.zeros([emb_shape[0],1,emb_shape[2]],tf.float32)
                conv_input = tf.concat([pad,emb,pad],axis=1)
                conv_res = tf.layers.conv1d(
                        inputs=conv_input,
                        filters=constant.t_filters, kernel_size=3,
                        strides=1,
                        padding='valid',
                        activation=tf.nn.relu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        name='convlution_layer')
                conv_res = tf.reshape(conv_res,[-1,maxlen,constant.t_filters])
            with tf.variable_scope('maxpooling'):
                maskl = tf.tile(tf.expand_dims(maskls,axis=2),[1,1,constant.t_filters])
                left = maskl*conv_res
                maskr = tf.tile(tf.expand_dims(maskrs,axis=2),[1,1,constant.t_filters])
                right = maskr*conv_res
                sentence_feature = tf.concat([tf.reduce_max(left,axis=1),tf.reduce_max(right,axis=1)],axis=1)
            with tf.variable_scope('classifier'):
                feature = tf.concat([sentence_feature,lexical_feature],axis=1)
                feature = tf.layers.dropout(feature,1-constant.t_keepprob,training=is_train)
                self.logits = logits = tf.layers.dense(feature,num_class,kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.contrib.layers.xavier_initializer())
                self.pred = pred = tf.nn.softmax(logits,axis=1)
                self.pred_label = pred_label = tf.argmax(pred,axis=1)
                self.loss = loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=logits),axis=0)
                self.train_op = train_op = tf.train.AdamOptimizer(constant.t_lr).minimize(loss)

    def train_trigger(self):
        train,dev,test = self.t_train,self.t_dev,self.t_test
        saver = tf.train.Saver()
        print('--Training Trigger--')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            devbest = 0
            testbest = (0,0,0)
            for epoch in range(constant.t_epoch):
                loss_list =[]
                for batch in get_batch(train,constant.t_batch_size,True):
                    loss,_ = sess.run([self.loss,self.train_op],feed_dict=get_trigger_feeddict(self,batch))
                    loss_list.append(loss)
                print('epoch:{}'.format(str(epoch)))
                print('loss:',np.mean(loss_list))

                pred_labels = []
                for batch in get_batch(dev,constant.t_batch_size,False):
                    pred_label = sess.run(self.pred_label,feed_dict=get_trigger_feeddict(self,batch,is_train=False))
                    pred_labels.extend(list(pred_label))
                golds = list(dev[4])
                dev_p,dev_r,dev_f = f_score(pred_labels,golds)
                print("dev_Precision: {} dev_Recall:{} dev_F1:{}".format(str(dev_p),str(dev_r),str(dev_f)))

                pred_labels = []
                for batch in get_batch(test,constant.t_batch_size,False):
                    pred_label = sess.run(self.pred_label,feed_dict=get_trigger_feeddict(self,batch,is_train=False))
                    pred_labels.extend(list(pred_label))
                golds = list(test[4])
                test_p, test_r, test_f = f_score(pred_labels, golds)
                print("test_Precision: {} test_Recall:{} test_F1:{}\n".format(str(test_p), str(test_r), str(test_f)))

                if dev_f>devbest:
                    devbest = dev_f
                    testbest = (test_p, test_r, test_f)
                    saver.save(sess,"saved_models/trigger.ckpt")
            test_p, test_r, test_f = testbest
            print("test best Precision: {} test best Recall:{} test best F1:{}".format(str(test_p), str(test_r), str(test_f)))
            a_data_process = self.predict_trigger(sess)
        return a_data_process

    def predict_trigger(self,sess):
        print('--Predict Trigger For Argument Stage--')
        train,dev,test = self.a_train,self.a_dev,self.a_test
        saver = tf.train.Saver()
        saver.restore(sess,"saved_models/trigger.ckpt")
        dev_pred_event_types = []
        test_pred_event_types = []
        for batch in get_batch(dev,constant.t_batch_size,False):
            pred_label = list(sess.run(self.pred_label,feed_dict=get_argument_feeddict(self,batch,False,'trigger')))
            dev_pred_event_types.extend(pred_label)
        for batch in get_batch(test,constant.t_batch_size,False):
            pred_label = list(sess.run(self.pred_label,feed_dict=get_argument_feeddict(self,batch,False,'trigger')))
            test_pred_event_types.extend(pred_label)
        return self.process_data_for_argument(np.array(dev_pred_event_types,np.int32),np.array(test_pred_event_types,np.int32))
    
    def process_data_for_argument(self,dev_pred_event_types,test_pred_event_types):
        print('--Preprocess Data For Argument Stage--')
        train,dev,test = list(self.a_train),list(self.a_dev),list(self.a_test)

        dev = list(dev)
        dev.append(dev_pred_event_types)
        dev = tuple(dev)

        test = list(test)
        test.append(test_pred_event_types)
        test = tuple(test)
        
        dev_slices = []
        for idx,event_type in enumerate(list(dev_pred_event_types)):
            if event_type!=0:
                dev_slices.append(idx)
        
        test_slices = []
        for idx,event_type in enumerate(list(test_pred_event_types)):
            if event_type!=0:
                test_slices.append(idx)

        dev = [np.take(d,dev_slices,axis=0) for d in dev]
        test = [np.take(d,test_slices,axis=0) for d in test]
        return train,dev,test

    def build_argument(self,scope="DMCNN_argument"):
        maxlen = self.maxlen
        num_class = len(constant.ROLE_TO_ID)
        with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
            with tf.variable_scope('Initialize'):
                posi_mat = tf.concat(
                            [tf.zeros([1,constant.posi_embedding_dim],tf.float32),
                            tf.get_variable('posi_emb',[2*maxlen,constant.posi_embedding_dim],tf.float32,initializer=tf.contrib.layers.xavier_initializer())],axis=0)
                word_mat =  tf.concat([
                            tf.zeros((1, constant.embedding_dim),dtype=tf.float32),
                            tf.get_variable("unk_word_embedding", [1, constant.embedding_dim], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer()),
                            tf.get_variable("word_emb", initializer=self.wordemb,trainable=True)], axis=0)

                event_mat = tf.concat([
                            tf.zeros((1, constant.event_type_embedding_dim),dtype=tf.float32),
                            tf.get_variable("event_emb", [len(constant.EVENT_TYPE_TO_ID)-1,constant.event_type_embedding_dim],initializer=tf.contrib.layers.xavier_initializer(),trainable=True)], axis=0)
            with tf.variable_scope('placeholder'):
                self.sents = sents = tf.placeholder(tf.int32,[None,maxlen],'sents')
                self.trigger_posis = trigger_posis = tf.placeholder(tf.int32,[None,maxlen],'trigger_posis')
                self.argument_posis = argument_posis = tf.placeholder(tf.int32,[None,maxlen],'argument_posis')
                self.maskls = maskls = tf.placeholder(tf.float32,[None,maxlen],'maskls')
                self.maskms = maskms = tf.placeholder(tf.float32,[None,maxlen],'maskms')
                self.maskrs = maskrs = tf.placeholder(tf.float32,[None,maxlen],'maskrs')
                self.event_types = event_types = tf.placeholder(tf.int32,[None],'event_types')
                self.trigger_lexical = trigger_lexical = tf.placeholder(tf.int32,[None,3],'trigger_lexicals')
                self.argument_lexical = argument_lexical = tf.placeholder(tf.int32,[None,2+self.max_argument_len],'argument_lexicals')
                self._labels = _labels = tf.placeholder(tf.int32,[None],'labels')
                labels = tf.one_hot(_labels,num_class)
                self.is_train = is_train = tf.placeholder(tf.bool,[],'is_train')
                
                sents_len = tf.reduce_sum(tf.cast(tf.cast(sents,tf.bool),tf.int32),axis=1)
                sents_mask = tf.sequence_mask(sents_len,maxlen,tf.float32)
                event_types = tf.tile(tf.expand_dims(event_types,axis=1),[1,maxlen])*tf.cast(sents_mask,tf.int32)
            with tf.variable_scope('embedding'):
                sents_emb = tf.nn.embedding_lookup(word_mat,sents)
                trigger_posis_emb  = tf.nn.embedding_lookup(posi_mat,trigger_posis)
                trigger_lexical_emb = tf.nn.embedding_lookup(word_mat,trigger_lexical)
                argument_posis_emb = tf.nn.embedding_lookup(posi_mat,argument_posis)
                argument_lexical_emb = tf.nn.embedding_lookup(word_mat,argument_lexical)
                event_type_emb = tf.nn.embedding_lookup(event_mat,event_types)
            with tf.variable_scope('lexical_feature'):
                trigger_lexical_feature = tf.reshape(trigger_lexical_emb,[-1,3*constant.embedding_dim])
                argument_len = tf.reduce_sum(tf.cast(tf.cast(argument_lexical[:,1:-1],tf.bool),tf.float32),axis=1,keepdims=True)
                argument_lexical_mid = tf.reduce_sum(argument_lexical_emb[:,1:-1,:],axis=1)/argument_len
                argument_lexical_feature = tf.concat([argument_lexical_emb[:,0,:],argument_lexical_mid,argument_lexical_emb[:,-1,:]],axis=1)
                lexical_feature = tf.concat([trigger_lexical_feature,argument_lexical_feature],axis=1)
            with tf.variable_scope('encoder'):
                emb = tf.concat([sents_emb,trigger_posis_emb,argument_posis_emb,event_type_emb],axis=2)
                emb_shape = tf.shape(emb)
                pad = tf.zeros([emb_shape[0],1,emb_shape[2]],tf.float32)
                conv_input = tf.concat([pad,emb,pad],axis=1)
                conv_res = tf.layers.conv1d(
                        inputs=conv_input,
                        filters=constant.a_filters, kernel_size=3,
                        strides=1,
                        padding='valid',
                        activation=tf.nn.relu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        name='convlution_layer')
                conv_res = tf.reshape(conv_res,[-1,maxlen,constant.a_filters])
            with tf.variable_scope('maxpooling'):
                maskl = tf.tile(tf.expand_dims(maskls,axis=2),[1,1,constant.a_filters])
                left = maskl*conv_res
                maskm = tf.tile(tf.expand_dims(maskms,axis=2),[1,1,constant.a_filters])
                mid = maskm*conv_res
                maskr = tf.tile(tf.expand_dims(maskrs,axis=2),[1,1,constant.a_filters])
                right = maskr*conv_res
                sentence_feature = tf.concat([tf.reduce_max(left,axis=1),tf.reduce_max(mid,axis=1),tf.reduce_max(right,axis=1)],axis=1)
            with tf.variable_scope('classifier'):
                feature = tf.concat([sentence_feature,lexical_feature],axis=1)
                feature = tf.layers.dropout(feature,1-constant.a_keepprob,training=is_train)
                self.logits = logits = tf.layers.dense(feature,num_class,kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.contrib.layers.xavier_initializer())
                self.pred = pred = tf.nn.softmax(logits,axis=1)
                self.pred_label = pred_label = tf.argmax(pred,axis=1)
                self.loss = loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=logits),axis=0)
                self.train_op = train_op = tf.train.AdamOptimizer(constant.a_lr).minimize(loss)

    def build_HMEAE(self,scope="HMEAE"):
        maxlen = self.maxlen
        num_class = len(constant.ROLE_TO_ID)
        with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
            with tf.variable_scope('Initialize'):
                posi_mat = tf.concat(
                            [tf.zeros([1,constant.posi_embedding_dim],tf.float32),
                            tf.get_variable('posi_emb',[2*maxlen,constant.posi_embedding_dim],tf.float32,initializer=tf.contrib.layers.xavier_initializer())],axis=0)
                word_mat =  tf.concat([
                            tf.zeros((1, constant.embedding_dim),dtype=tf.float32),
                            tf.get_variable("unk_word_embedding", [1, constant.embedding_dim], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer()),
                            tf.get_variable("word_emb", initializer=self.wordemb,trainable=True)], axis=0)

                event_mat = tf.concat([
                            tf.zeros((1, constant.event_type_embedding_dim),dtype=tf.float32),
                            tf.get_variable("event_emb", [len(constant.EVENT_TYPE_TO_ID)-1,constant.event_type_embedding_dim],initializer=tf.contrib.layers.xavier_initializer(),trainable=True)], axis=0)
            
                u_c = tf.get_variable('feat_vatiable',[constant.module_num,1,constant.a_u_c_dim],initializer=tf.contrib.layers.xavier_initializer())   
                module_design = tf.constant(constant.module_design,tf.float32)
            with tf.variable_scope('placeholder'):
                self.sents = sents = tf.placeholder(tf.int32,[None,maxlen],'sents')
                self.trigger_posis = trigger_posis = tf.placeholder(tf.int32,[None,maxlen],'trigger_posis')
                self.argument_posis = argument_posis = tf.placeholder(tf.int32,[None,maxlen],'argument_posis')
                self.maskls = maskls = tf.placeholder(tf.float32,[None,maxlen],'maskls')
                self.maskms = maskms = tf.placeholder(tf.float32,[None,maxlen],'maskms')
                self.maskrs = maskrs = tf.placeholder(tf.float32,[None,maxlen],'maskrs')
                self.event_types = event_types = tf.placeholder(tf.int32,[None],'event_types')
                self.trigger_lexical = trigger_lexical = tf.placeholder(tf.int32,[None,3],'trigger_lexicals')
                self.argument_lexical = argument_lexical = tf.placeholder(tf.int32,[None,2+self.max_argument_len],'argument_lexicals')
                self._labels = _labels = tf.placeholder(tf.int32,[None],'labels')
                labels = tf.one_hot(_labels,num_class)
                self.is_train = is_train = tf.placeholder(tf.bool,[],'is_train')
                
                sents_len = tf.reduce_sum(tf.cast(tf.cast(sents,tf.bool),tf.int32),axis=1)
                sents_mask = tf.sequence_mask(sents_len,maxlen,tf.float32)
                event_types = tf.tile(tf.expand_dims(event_types,axis=1),[1,maxlen])*tf.cast(sents_mask,tf.int32)
                batch_size = tf.shape(sents)[0]
            with tf.variable_scope('embedding'):
                sents_emb = tf.nn.embedding_lookup(word_mat,sents)
                trigger_posis_emb  = tf.nn.embedding_lookup(posi_mat,trigger_posis)
                trigger_lexical_emb = tf.nn.embedding_lookup(word_mat,trigger_lexical)
                argument_posis_emb = tf.nn.embedding_lookup(posi_mat,argument_posis)
                argument_lexical_emb = tf.nn.embedding_lookup(word_mat,argument_lexical)
                event_type_emb = tf.nn.embedding_lookup(event_mat,event_types)
            with tf.variable_scope('lexical_feature'):
                trigger_lexical_feature = tf.reshape(trigger_lexical_emb,[-1,3*constant.embedding_dim])
                argument_len = tf.reduce_sum(tf.cast(tf.cast(argument_lexical[:,1:-1],tf.bool),tf.float32),axis=1,keepdims=True)
                argument_lexical_mid = tf.reduce_sum(argument_lexical_emb[:,1:-1,:],axis=1)/argument_len
                argument_lexical_feature = tf.concat([argument_lexical_emb[:,0,:],argument_lexical_mid,argument_lexical_emb[:,-1,:]],axis=1)
                lexical_feature = tf.concat([trigger_lexical_feature,argument_lexical_feature],axis=1)
            with tf.variable_scope('encoder'):
                emb = tf.concat([sents_emb,trigger_posis_emb,argument_posis_emb,event_type_emb],axis=2)
                emb_shape = tf.shape(emb)
                pad = tf.zeros([emb_shape[0],1,emb_shape[2]],tf.float32)
                conv_input = tf.concat([pad,emb,pad],axis=1)
                conv_res = tf.layers.conv1d(
                        inputs=conv_input,
                        filters=constant.a_filters, kernel_size=3,
                        strides=1,
                        padding='valid',
                        activation=tf.nn.relu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        name='convlution_layer')
                conv_res = tf.reshape(conv_res,[-1,maxlen,constant.a_filters])
            with tf.variable_scope("attention"):
                conv_res_extend = tf.tile(tf.expand_dims(conv_res,axis=1),[1,constant.module_num,1,1])
                u_c_feat = tf.tile(tf.expand_dims(u_c,axis=0),[batch_size,1,maxlen,1])*tf.tile(tf.expand_dims(tf.expand_dims(sents_mask,axis=2),axis=1),[1,constant.module_num,1,1])
                hidden_state = tf.layers.dense(tf.concat([conv_res_extend,u_c_feat],axis=3),constant.a_W_a_dim,use_bias=False,kernel_initializer=tf.contrib.layers.xavier_initializer(),activation=tf.nn.tanh)
                score_logit = tf.reshape(tf.layers.dense(hidden_state,1,use_bias=False,kernel_initializer=tf.contrib.layers.xavier_initializer()),[batch_size,constant.module_num,maxlen])
                score_mask = tf.tile(tf.expand_dims(sents_mask,axis=1),[1,constant.module_num,1])
                score_logit = score_logit*score_mask-(1-score_mask)*constant.INF
                module_score = tf.nn.softmax(score_logit,axis=2)
                module_mask = tf.tile(tf.expand_dims(tf.expand_dims(module_design,axis=0),axis=3),[batch_size,1,1,maxlen])
                score_mask = tf.tile(tf.expand_dims(module_score,axis=1),[1,len(constant.ROLE_TO_ID),1,1])*module_mask
                module_of_role = tf.expand_dims(tf.expand_dims(tf.reduce_sum(module_design,axis=1),axis=0),axis=2)
                role_score = tf.reduce_sum(score_mask,axis=2)/module_of_role
                role_oriented_emb = tf.reduce_sum(tf.expand_dims(role_score,axis=3)*tf.tile(tf.expand_dims(conv_res,axis=1),[1,len(constant.ROLE_TO_ID),1,1]),axis=2)
            with tf.variable_scope('maxpooling'):
                maskl = tf.tile(tf.expand_dims(maskls,axis=2),[1,1,constant.a_filters])
                left = maskl*conv_res
                maskm = tf.tile(tf.expand_dims(maskms,axis=2),[1,1,constant.a_filters])
                mid = maskm*conv_res
                maskr = tf.tile(tf.expand_dims(maskrs,axis=2),[1,1,constant.a_filters])
                right = maskr*conv_res
                sentence_feature = tf.concat([tf.reduce_max(left,axis=1),tf.reduce_max(mid,axis=1),tf.reduce_max(right,axis=1)],axis=1)
            with tf.variable_scope('classifier'):
                dmcnn_feature = tf.concat([sentence_feature,lexical_feature],axis=1)
                hmeae_feature = tf.concat([tf.tile(tf.expand_dims(dmcnn_feature,axis=1),[1,len(constant.ROLE_TO_ID),1]),role_oriented_emb],axis=2)
                feature = tf.layers.dropout(hmeae_feature,1-constant.a_keepprob,training=is_train)
                eye_mask = tf.tile(tf.expand_dims(tf.eye(num_class),axis=0),[batch_size,1,1])
                dense_res = tf.layers.dense(feature,num_class,kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.contrib.layers.xavier_initializer())
                self.logits = logits = tf.reduce_max(eye_mask*dense_res-(1-eye_mask)*constant.INF,axis=2)
                self.pred = pred = tf.nn.softmax(logits,axis=1)
                self.pred_label = pred_label = tf.argmax(pred,axis=1)
                self.loss = loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=logits),axis=0)
                self.train_op = train_op = tf.train.AdamOptimizer(constant.a_lr).minimize(loss)
            
    def train_argument(self):
        print('--Training Argument--')
        train,dev,test = self.a_train,self.a_dev,self.a_test
        with tf.Session() as sess:
            devbest = 0
            testbest=(0,0,0)
            sess.run(tf.global_variables_initializer())
            for epoch in range(constant.a_epoch):
                loss_list = []
                for batch in get_batch(train,constant.a_batch_size,shuffle=True):
                    loss,_ = sess.run([self.loss,self.train_op],feed_dict=get_argument_feeddict(self,batch,True,"argument"))
                    loss_list.append(loss)
                print('epoch:{}'.format(str(epoch)))
                print('loss:',np.mean(loss_list))

                pred_labels = []
                for batch in get_batch(dev,constant.a_batch_size,False):
                    pred_event_types,feed_dict = get_argument_feeddict(self,batch,False,"argument")
                    pred_label = sess.run(self.pred_label,feed_dict=feed_dict)
                    pred_labels.extend(list(zip(list(pred_event_types),list(pred_label))))
                golds = list(zip(list(dev[1]),list(dev[2])))
                dev_p,dev_r,dev_f = f_score(pred_labels,golds,self.classify)
                print("dev_Precision: {} dev_Recall:{} dev_F1:{}".format(str(dev_p),str(dev_r),str(dev_f)))

                pred_labels = []
                for batch in get_batch(test,constant.a_batch_size,False):
                    pred_event_types,feed_dict = get_argument_feeddict(self,batch,False,"argument")
                    pred_label = sess.run(self.pred_label,feed_dict=feed_dict)
                    pred_labels.extend(list(zip(list(pred_event_types),list(pred_label))))
                golds = list(zip(list(test[1]),list(test[2])))
                test_p, test_r, test_f = f_score(pred_labels, golds,self.classify)
                print("test_Precision: {} test_Recall:{} test_F1:{}\n".format(str(test_p), str(test_r), str(test_f)))

                if dev_f>devbest:
                    devbest = dev_f
                    testbest = (test_p, test_r, test_f)
            test_p, test_r, test_f = testbest
            print("test best Precision: {} test best Recall:{} test best F1:{}".format(str(test_p), str(test_r), str(test_f)))
