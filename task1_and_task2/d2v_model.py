# -*- coding: utf-8 -*-
import numpy as np
import gensim


class Doc2Vec_Model():
    
    def __init__(self, model_path):
        
        print('start loading doc2vec model ' + model_path)
        self.dim = int(((model_path.split('.'))[-2])[0:-1])
        self.model = gensim.models.Doc2Vec.load(model_path)
        self.vectors = self.model.wv
        print('complete!')
        
    def has_key(self, word):
        return True if word in self.vectors.vocab.keys() else False
        
    def doc_vec(self, docId):
        return self.model.docvecs[docId]

    def word_vec(self, word):
        return self.vectors[word] if self.has_key(word) else np.zeros((self.dim), np.float)

    '''
    对word_list中的词的词向量求平均，不包含不计算在length中
    '''
    def get_average_vector(self, word_list = []):
        
        vector = np.zeros((self.dim), np.float)
        
        has_key_num = 0
        not_exist_list = []
        
        for word in word_list:
            if self.has_key(word):
                vector += self.vectors[word]
                has_key_num += 1
            else:
                not_exist_list.append(word)
        
        print('#Doc2Vec_Model: calculate average vector of ' + ','.join(word_list) + ' .')
        
        if has_key_num != len(word_list):
            print('#Doc2Vec_Model: vector(s) of ' + ','.join(not_exist_list) + ' not found in the vocabulary!')
                
        return vector / has_key_num if has_key_num > 0 else vector
    
        
    def most_similar_average_vector(self, word_list = [], topn = 10):
        return self.vectors.similar_by_vector(self.get_average_vector(word_list), topn=topn)
    
        
    def most_similar(self, word_list = [], topn = 10):
        return self.vectors.most_similar(word_list, topn=topn)
    
        
        
if __name__ =='__main__':
        
    d2v_model = Doc2Vec_Model('d2v/100/csdn.d2v.100d.model')
    print(d2v_model.doc_vec('D0000002'))
    print(d2v_model.model.docvecs.most_similar('D0000003'))
    print(d2v_model.model.docvecs.most_similar('D0000002'))
#    print(d2v_model.most_similar_average_vector(['网络管理', '智能机器人'], topn=20))
    
    
    
    
    
    
    
    
    
    
    


    
    
