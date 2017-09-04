# -*- coding: utf-8 -*-
import numpy as np
import time
from util import load_user_data, load_validation_set, load_task2_tags, load_task2_train_set,load_test_set, read_lines_as_list
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from lda_model import LDA_Model
import random
import codecs
from d2v_model import Doc2Vec_Model
np.random.seed(2264)

'''
从训练集中划出验证集，训练集所占的比重
'''
sp = 1.0






class Task2_Model():
    
    def __init__(self):
        '''
        model1: doc2vec
        '''
        self.model1 = Sequential()
        self.model1.add(Dense(30, activation='relu', input_dim=100))
        self.model1.add(Dense(42, activation='sigmoid'))
        
        self.model1.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['acc'])
        
        '''
        model2: lda
        '''
        self.model2 = Sequential()
        self.model2.add(Dense(64, activation='relu', input_dim=100))
        self.model2.add(Dense(42, activation='sigmoid'))
        
        self.model2.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['acc'])
        
        
        
    def train_model(self, x_train1, x_train2, y_train):
        epochs1 = 50
        epochs2 = 100
        
        if sp < 1.0:
            self.model1.fit(x_train1[0:int(len(x_train1)*sp), :], y_train[0:int(len(y_train)*sp), :],
                  batch_size=5,
                  epochs=epochs1,
                  validation_data=(x_train1[int(len(x_train1)*sp):len(x_train1), :], y_train[int(len(y_train)*sp):len(y_train), :]))
            
            self.model2.fit(x_train2[0:int(len(x_train2)*sp), :], y_train[0:int(len(y_train)*sp), :],
                  batch_size=5,
                  epochs=epochs2,
                  validation_data=(x_train2[int(len(x_train2)*sp):len(x_train2), :], y_train[int(len(y_train)*sp):len(y_train), :]))
            
            
            
        elif sp == 1.0:
            self.model1.fit(x_train1, y_train,
                  batch_size=5,
                  epochs=epochs1,
                  validation_data=(x_train1, y_train))
            self.model2.fit(x_train2, y_train,
                  batch_size=5,
                  epochs=epochs2,
                  validation_data=(x_train2, y_train))
            
            
            
             
    def save_model(self, path):
        self.model1.save(path + '_1_d2v.h5')
        self.model2.save(path + '_2_lda.h5')
        
    def load_model(self, path):
        self.model1.load_weights(path + '_1_d2v.h5')
        self.model2.load_weights(path + '_2_lda.h5')
        
    def predict(self, fea1, fea2):
        return self.model1.predict(fea1), self.model2.predict(fea2)
        
    def summary(self):
        print(self.model1.summary())
        print(self.model2.summary())
        
        
def get_doc_feature_lda(doc_topic, K):
    res = np.zeros((K), dtype=np.float64)
    z_total = sum(doc_topic.values())
    for z in doc_topic.keys():
        res[z] = doc_topic[z] / z_total if z_total > 0 else 0.0
    return res

def get_user_feature(user_activity, user_id, d2v_model, doc_spec_z_dist, K):
    
    '''
    对不同类型的document以不同的权重加权主题特征，不考虑多次
    '''
    keys = ['post', 'browse', 'comment', 'vote_up', 'favorite']  # 'vote_down'
    weights = [20, 5, 10, 10, 20]
    weight = {}
    for i in range(len(keys)):
        key = keys[i]
        if user_id in user_activity[key].keys():
            for doc_id in user_activity[key][user_id]:
                if doc_id in weight.keys():
                    weight[doc_id] = max(weight[doc_id], weights[i])
                else:
                    weight[doc_id] = weights[i]


    user_feature1 = np.zeros((d2v_model.dim), dtype=np.float64)
    weight_sum = 0.0
    for doc_id in weight.keys():
        user_feature1 += d2v_model.doc_vec(doc_id) * weight[doc_id]
        weight_sum += weight[doc_id]
    user_feature1 /= weight_sum
    
    
    user_feature2 = np.zeros((K), dtype=np.float64)
    weight_sum = 0.0
    for doc_id in weight.keys():
        int_id = int(doc_id[1:])
        user_feature2 += get_doc_feature_lda(doc_spec_z_dist[int_id], K) * weight[doc_id]
        weight_sum += weight[doc_id]
    user_feature2 /= weight_sum
    return user_feature1, user_feature2

'''
ground truth以概率形式给出
'''
def get_label(label, tag2id):
    label_num = 42
    res = np.zeros((label_num), dtype=np.float)
    ids = [tag2id[tag] for tag in label]
    res[ids[0]] = 0.95
    res[ids[1]] = 0.85
    res[ids[2]] = 0.75
    for i in range(0, label_num):
        if i not in ids:
            res[i] = 0.2
    return res
        
    
        
if __name__ =='__main__':
        
    tag2id, id2tag = load_task2_tags()
    task2_train_ids, userid2label = load_task2_train_set()
    
    task1_valid_ids, task2_valid_ids, task3_valid_ids = load_validation_set()
    task1_test_ids, task2_test_ids, task3_test_ids = load_test_set()
    user_activity = load_user_data()
    
    d2v_model = Doc2Vec_Model('d2v/100/csdn.d2v.100d.model')
    lda_model = LDA_Model('lda/model/k_100')
    
    x_train1 = []
    x_train2 = []
    y_train = []
    for user_id in task2_train_ids:
        
        label = userid2label[user_id]
        y_train_cur = get_label(label, tag2id)
        user_feature1, user_feature2 = get_user_feature(user_activity, user_id, d2v_model, lda_model.doc_spec_z_dist, lda_model.K)
        x_train1.append(user_feature1)
        x_train2.append(user_feature2)
        y_train.append(y_train_cur)
    x_train1 = np.array(x_train1, dtype=float)
    x_train2 = np.array(x_train2, dtype=float)
    y_train = np.array(y_train, dtype=float)

    
    '''
    模型训练与保存
    '''
#    model = Task2_Model()
#    model.train_model(x_train1, x_train2, y_train)
#    model.save_model('model/task2_new/' + time.strftime('%m%d_%H_%M',time.localtime(time.time())))
#    model.summary()
    
    model = Task2_Model()
    model.load_model('model/task2_new/' + '0810_09_57')
    
    '''
    预测训练集
    '''
    if sp < 1.0:
        y_train_set = y_train[0:int(len(x_train1)*sp), :]
        pred1, pred2 = model.predict(x_train1[0:int(len(x_train1)*sp), :], x_train2[0:int(len(x_train1)*sp), :])
    else:
        y_train_set = y_train
        pred1, pred2 = model.predict(x_train1, x_train2)
        
    
    total_score1 = 0.0
    total_score2 = 0.0
    total_score_combine = 0.0
    for i in range(len(pred1)):
        
        a1 = np.argsort(pred1[i])
        b1 = a1[-3:]
        predict_tags1 = [id2tag[tagid] for tagid in b1]
        score1 = 0.0
        for tag_id in b1:
            if y_train_set[i, tag_id] >= 0.75:
                score1 += 1.0
        total_score1 += score1
        
        a2 = np.argsort(pred2[i])
        b2 = a2[-3:]
        predict_tags2 = [id2tag[tagid] for tagid in b2]
        score2 = 0.0
        for tag_id in b2:
            if y_train_set[i, tag_id] >= 0.75:
                score2 += 1.0
        total_score2 += score2
        
        
        
        
    print('train set: d2v ' + str(total_score1 / (len(pred1) * 3)))
    print('train set: lda ' + str(total_score2 / (len(pred2) * 3)))
    
    
    
    
    total_score1 = 0.0
    total_score2 = 0.0
    total_score_combine = 0.0
    
    for k in range(0, 11):
        ratio = 0.1 * k
        total_score1 = 0.0
        pred = ratio * pred1 + (1-ratio) * pred2
        
        for i in range(len(pred)):
            
            a1 = np.argsort(pred[i])
            b1 = a1[-3:]
            predict_tags1 = [id2tag[tagid] for tagid in b1]
            score1 = 0.0
            
            for tag_id in b1:
                if y_train_set[i, tag_id] >= 0.75:
                    score1 += 1.0
            total_score1 += score1
        print('train set -> ratio = 0.' + str(k) + ', score = ' + str(total_score1 / (len(pred1) * 3)))
    
    
        
        
    '''
    从训练集中划出的验证集
    '''
    if sp < 1.0:
        y_valid_set = y_train[int(len(y_train)*sp):len(y_train)]
        pred1, pred2 = model.predict(x_train1[int(len(x_train1)*sp):len(x_train1), :], x_train2[int(len(x_train2)*sp):len(x_train2), :])
    else:
        y_valid_set = y_train
        pred1, pred2 = model.predict(x_train1, x_train2) 
        
    for k in range(0, 11):
        ratio = 0.1 * k
        total_score1 = 0.0
        pred = ratio * pred1 + (1-ratio) * pred2
        
        for i in range(len(pred)):
            
            a1 = np.argsort(pred[i])
            b1 = a1[-3:]
            predict_tags1 = [id2tag[tagid] for tagid in b1]
            score1 = 0.0
            
            for tag_id in b1:
                if y_valid_set[i, tag_id] >= 0.75:
                    score1 += 1.0
            total_score1 += score1
        print('validate set -> ratio = 0.' + str(k) + ', score = ' + str(total_score1 / (len(pred) * 3)))
        
        
        
        
    
    test_1 = []
    test_2 = []
#    for user_id in task2_valid_ids:
    for user_id in task2_test_ids:
        user_feature1, user_feature2 = get_user_feature(user_activity, user_id, d2v_model, lda_model.doc_spec_z_dist, lda_model.K)
        test_1.append(user_feature1)
        test_2.append(user_feature2)
    test_1 = np.array(test_1, dtype=float)
    test_2 = np.array(test_2, dtype=float)
    
    pred1, pred2 = model.predict(test_1, test_2)
    pred = 0.1 * pred1 + 0.9 * pred2
    
    out = codecs.open('test_task2_2.txt', 'w', 'utf-8')
    
    for i in range(len(pred)):
        a = np.argsort(pred[i])
        b = a[len(a)-3:]
        tags = [id2tag[tagid] for tagid in b]

        out.write(str(task2_test_ids[i]) + ',' + ','.join(tags) + '\n')
    out.close()
    
