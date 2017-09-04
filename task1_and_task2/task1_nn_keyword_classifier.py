# -*- coding: utf-8 -*-
import numpy as np
import re
from keras.models import Sequential
from keras.layers import Dense, Dropout
from util import read_doc_by_id, load_word_count, load_title_word_count, read_lines_as_set, load_ngram_count
from lda_model import LDA_Model
from d2v_model import Doc2Vec_Model
import doc_preprocess
np.random.seed(2264)

class NeuralNetworkKeywordClassifier():
    
    def __init__(self):
        self.word_count = load_word_count()
        self.model = Sequential()
        self.model.add(Dense(128, activation='relu', input_dim=13))
        
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        
    def train_model(self, x_train, y_train):
        self.model.fit(x_train, y_train,
                  batch_size=32,
                  epochs=200,
                  validation_data=(x_train, y_train))
        
    def save_model(self, path):
        self.model.save(path)
        
    def load_model(self, path):
        self.model.load_weights(path)
        
    def predict(self, x_test):
        return self.model.predict(x_test)
        
        
    def summary(self):
        print(self.model.summary())
        
    def extractKeywords(self, x, keywords):
        pred = self.predict(x)
                            
        if len(keywords) < 3:
            new_add = {}
            
            for i in range(len(keywords)):
                if len(keywords[i]) == 4:
                    w1 = keywords[i][0:2]
                    w2 = keywords[i][2:4]
                    new_add[w1] = self.word_count[w1] if w1 in self.word_count.keys() else 1
                    new_add[w2] = self.word_count[w2] if w2 in self.word_count.keys() else 1
            new_add = (sorted(new_add.items(), key=lambda d: d[1], reverse=True))
            for i in range(len(new_add)):
                if len(keywords) < 3:
                    keywords.append(new_add[i][0])
                else:
                    break
            
        res = []
        if len(keywords) <= 3:
            for i in range(0, 3):
                if len(keywords) > i:
                    res.append(keywords[i])
                else:
                    res.append('nan')
        else:
            pred_sort = sorted(pred, reverse=True)
            res = [keywords[a] for a in range(0, len(pred)) if pred[a] >= pred_sort[2]]
            if len(res) > 3:
                res = res[0:3]

        return res
        
    def tmp(self, x, keywords, num):
        pred = self.predict(x)
                            
        if len(keywords) < 3:
            new_add = {}
            
            for i in range(len(keywords)):
                if len(keywords[i]) == 4:
                    w1 = keywords[i][0:2]
                    w2 = keywords[i][2:4]
                    new_add[w1] = self.word_count[w1] if w1 in self.word_count.keys() else 1
                    new_add[w2] = self.word_count[w2] if w2 in self.word_count.keys() else 1
            new_add = (sorted(new_add.items(), key=lambda d: d[1], reverse=True))
            for i in range(len(new_add)):
                if len(keywords) < 3:
                    keywords.append(new_add[i][0])
                else:
                    break
            
        res = []
        if len(keywords) <= num:
            for i in range(0, num):
                if len(keywords) > i:
                    res.append(keywords[i])
        else:
            pred_sort = sorted(pred, reverse=True)
            res = [keywords[a] for a in range(0, len(pred)) if pred[a] >= pred_sort[num-1]]
            if len(res) > num:
                res = res[0:num]

        return res



class NeuralNetworkFeatureGenerator():
    
    def __init__(self, it_lexicons, word_df):
        
        self.document_preprocessor = doc_preprocess.DocumentPreprocessor()
        
        self.it_lexicons = it_lexicons
        self.word_df = word_df
        self.word_count = load_word_count()
        self.lda_model = LDA_Model('lda/model/k_100')
        self.title_word_count = load_title_word_count()
        self.ngram = load_ngram_count()
        self.two_unary = read_lines_as_set('dicts/two_unary.txt', skip=1) 
        self.bigram_lexicons = read_lines_as_set('dicts/bigram_dict.txt', skip=1) 
        self.trigram_lexicons = read_lines_as_set('dicts/trigram_dict.txt', skip=1) 
        self.d2v_model = Doc2Vec_Model('d2v/100/csdn.d2v.100d.model')
    
    def get_dist_word(self, title_word, content_word, word):
        total_num = len(title_word) + len(content_word)
        for i in range(0, len(title_word)):
            if title_word[i] == word:
                return (i+1) / total_num
        for i in range(0, len(content_word)):
            if content_word[i] == word:
                return (i+1+len(title_word)) / total_num
        return 1.0
    
    # 计算两个numpy array之间的余弦相似度
    def cosine_similarity(self, vec1, vec2):
        numerator = np.dot(vec1, vec2)
        denominator = np.sqrt(np.square(vec1).sum()) * np.sqrt(np.square(vec2).sum())
        return numerator / denominator

    def degree(self, vec1, vec2):
        a = vec1 / sum(vec1)
        b = vec2 / sum(vec2)
        return np.dot(a, b)
        
#    def count_alpha_num(self, word):
#        count_alpha = 0
#        count_num = 0
#        for i in range(len(word)):
#            if word[i] >= 'a' and word[i] <= 'z':
#                count_alpha += 1
#            if word[i] >= '0' and word[i] <= '9':
#                count_num += 1
#        return count_alpha, count_num
        
    def contain_stop_char(self, word):
        for i in range(len(word)):
            if word[i] in self.document_preprocessor.stopch:
                return True
        return False
        
        
        
    '''
    instance is a (doc_id, [word1, word2]) tuple
    '''
    def _gen_features_of_doc_instance(self, doc_id, words, title_word, content_word):
        
        feas = []
        
        word_tf = {}
        word_set = set(title_word + content_word)

        for word in title_word:
            if word in word_tf.keys():
                word_tf[word] += 1
            else:
                word_tf[word] = 1
        for word in content_word:
            if word in word_tf.keys():
                word_tf[word] += 1
            else:
                word_tf[word] = 1

                
        MAX_TF = 0
        MAX_IDF = 0
        for word in word_set:
            MAX_TF = max(MAX_TF, word_tf[word])
            df = self.word_df[word] if word in self.word_df.keys() else 1
            df = self.ngram[word] if word in self.ngram.keys() and df == 1 else df
            idf = np.log2(1000001 / (df+1))
            MAX_IDF = max(MAX_IDF, idf)
            
            
        doc_topic = self.lda_model.doc_spec_z_dist[int(doc_id[1:])]
        doc_topic_dist = np.zeros((self.lda_model.K), dtype=np.float32)
        for topic in doc_topic.keys():
            doc_topic_dist[topic] = doc_topic[topic]
            
        for i in range(len(words)):
            word = words[i]
            fea = []
            fea.append(len(word) / 10.0)     # 词语长度
            
            fea.append(1.0 if re.search('[a-zA-Z0-9]+', word) else 0.0)   # 是否包含数字字母
            
            fea.append(1.0 if word in title_word else 0.0)         #  是否出现在标题中
            fea.append(1.0 if word in content_word else 0.0)      # 是否出现在正文中
            tf = word_tf[word] / MAX_TF if word in word_tf.keys() else 1.0 / MAX_TF    
            fea.append(tf)                                          # 词频特征
            df = self.word_df[word] if word in self.word_df.keys() else 1    
            idf = np.log2(1000001 / (df+1))  
            fea.append(idf / MAX_IDF)                                # 逆文档频率
            fea.append(tf * idf)                                       # tfidf
            fea.append(self.get_dist_word(title_word, content_word, word))      #  首次出现的position
            if word in self.word_count.keys():
                word_count = self.word_count[word]
            elif word in self.ngram.keys():
                word_count = self.ngram[word]
            elif word.find(' ')!=-1:
                word_count = 50
            else:
                word_count = 1
            fea.append(word_count / 27852.0)                # 计数
            
            # No.10
            if word not in self.lda_model.word_spec_z_dist.keys():
#                pass
                fea.append(0.0)
            else:
                word_topic = self.lda_model.word_spec_z_dist[word]
                word_topic_dist = np.zeros((self.lda_model.K), dtype=np.float32)
                for topic in word_topic.keys():
                    word_topic_dist[topic] = word_topic[topic]
                cosine_simi = self.cosine_similarity(word_topic_dist, doc_topic_dist)
                fea.append(cosine_simi)                      #  单词主题与文档主题分布的余弦相似度
                
            # No.11
            if word not in self.lda_model.word_spec_z_dist.keys():
                fea.append(0.0)
            else:
                fea.append(self.degree(word_topic_dist, doc_topic_dist))        # 单词表达文档主题的程度
                
            # No.12
            if self.d2v_model.has_key(word):
                word_embedding = self.d2v_model.word_vec(word)
                doc_embedding = self.d2v_model.doc_vec(doc_id)
                fea.append(self.cosine_similarity(word_embedding,doc_embedding))
            elif len(word) == 3:
                if self.d2v_model.has_key(word[0:1]) and self.d2v_model.has_key(word[1:]):
                    word_embedding = self.d2v_model.get_average_vector([word[0:1], word[1:]])
                    doc_embedding = self.d2v_model.doc_vec(doc_id)
                    fea.append(self.cosine_similarity(word_embedding,doc_embedding))
                elif self.d2v_model.has_key(word[0:2]) and self.d2v_model.has_key(word[2:]):
                    word_embedding = self.d2v_model.get_average_vector([word[0:2], word[2:]])
                    doc_embedding = self.d2v_model.doc_vec(doc_id)
                    fea.append(self.cosine_similarity(word_embedding,doc_embedding))
                else:
                    fea.append(0.0)
            elif len(word) == 4:
                if self.d2v_model.has_key(word[0:2]) and self.d2v_model.has_key(word[2:]):
                    word_embedding = self.d2v_model.get_average_vector([word[0:2], word[2:]])
                    doc_embedding = self.d2v_model.doc_vec(doc_id)
                    fea.append(self.cosine_similarity(word_embedding,doc_embedding))
                else:
                    fea.append(0.0)
            else:
                fea.append(0.0)

#            # No.13 No.14
#            if re.search('[a-zA-Z0-9]+', word):
#                count_alpha, count_num = self.count_alpha_num(word)
#                fea.append(count_alpha / 10.0)
#                fea.append(count_num / 10.0)
#            else:
#                fea.append(0.0)
#                fea.append(0.0)
                
            # No.15
            if self.contain_stop_char(word):
                fea.append(1.0)
            else:
                fea.append(0.0)
                
            feas.append(fea)
    
        return feas
        
    def _gen_labels_of_doc_instance(self, doc_id, words, title_word, content_word, docid2label):
        
        labs = []
        for i in range(len(words)):
            if words[i] in docid2label[doc_id]:
                labs.append(1)
            else:
                labs.append(0)
        return labs
        
    def gen_feature_of_doc_instance(self, doc_id, words, title_word, content_word):
        feas = self._gen_features_of_doc_instance(doc_id, words, title_word, content_word)
        return np.array(feas, dtype=float)
        
    def gen_feature_of_doc_instances(self, instances):
        features = []
        for k in instances:
            
            feas = self._gen_features_of_doc_instance(k[0], k[1], k[2], k[3])
            for fea in feas:
                features.append(fea)
        return np.array(features, dtype=float)
        
    def gen_labels_of_doc_instances(self, instances, docid2label):
        labels = []
        for k in instances:
            labs = self._gen_labels_of_doc_instance(k[0], k[1], k[2], k[3], docid2label)
            for lab in labs:
                labels.append(lab)
        return np.array(labels, dtype=float)
        


        
if __name__ =='__main__':
    
    print('nn_keyword_classifier')
    

