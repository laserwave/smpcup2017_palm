# -*- coding: utf-8 -*-
import pickle


class LDA_Model():
    
    def __init__(self, model_path):
        
        print('start loading lda model ' + model_path)
        
        '''
        document specific topic distribution
        dict: 
            key: doc id (1 ~ 1000000)
            value: dict (key: topic id, value: count)
        '''
        self.doc_spec_z_dist = pickle.load(open(model_path + '/pickle.doc_spec_z_dist', 'rb'))
        
        '''
        count of words
        dict:
            key: the word
            value: count
        '''
        self.nw = pickle.load(open(model_path + '/pickle.nw', 'rb'))
        
        '''
        count of topic
        dict:
            key: topic id (0 ~ K-1)
            value: count
        '''
        self.nz = pickle.load(open(model_path + '/pickle.nz', 'rb'))
        
        '''
        word specific topic distribution
        dict: 
            key: word
            value: dict (key: topic id, value: count)
        '''
        self.word_spec_z_dist = pickle.load(open(model_path + '/pickle.word_spec_z_dist', 'rb'))
        
        self.word2id = pickle.load(open(model_path + '/pickle.word2id', 'rb'))
        
        self.id2word = pickle.load(open(model_path + '/pickle.id2word', 'rb'))
        
        '''
        K * V array
        '''
        self.zw = pickle.load(open(model_path + '/pickle.zw', 'rb'))
        
        self.K = len(self.nz.keys())
        
        self.V = len(self.word2id.keys())
        
        print('complete!')
    
    
if __name__ =='__main__':
    
    lda_model = LDA_Model('lda/model/k_100')
    print('K', lda_model.K)
    print('V', lda_model.V)
    
    
    
    
    
    
    
    
    
    
