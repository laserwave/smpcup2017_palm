# -*- coding: utf-8 -*-
import numpy as np
import re
from util import read_lines_as_list, read_lines_as_set, load_word_df
import util

class DocumentPreprocessor():
    def __init__(self):
        self.word_df = load_word_df()   # 文档频率
        self.stopwords = read_lines_as_set('dicts/stopwords.txt') # 停用词
        self.stopch = read_lines_as_set('dicts/stopch.txt')
        self.surnames = read_lines_as_set('dicts/surnames.txt') # 姓
        self.stop_lexicons = read_lines_as_set('dicts/stop_lexicons.txt', skip=1) # 包含该词的也视为停用词
        self.it_lexicons = read_lines_as_list('dicts/it_lexicons.txt', skip=1) # it专业名词
        self.two_unary = read_lines_as_set('dicts/two_unary.txt', skip=1) 
        self.bigram_lexicons = read_lines_as_set('dicts/bigram_dict.txt', skip=1) 
        self.trigram_lexicons = read_lines_as_set('dicts/trigram_dict.txt', skip=1) 
        self.quatgram_lexicons = read_lines_as_set('dicts/quatgram_dict.txt', skip=1)
    '''
    专有名词转化
    '''
    def convert_it_lexicons(self, word_list):
        new_list = []
        for word in word_list:
            flag = 0
            for lexicon in self.it_lexicons:
                if len(lexicon) < 2:
                    continue
                if word.find(lexicon) != -1:
                    new_list.append(lexicon)
                    flag = 1
                    break
            if flag == 0:
                new_list.append(word)
        return new_list
        
    def combine(self, word_list):
        if len(word_list) < 2:
            return word_list
        delete = []
        
        for i in range(1, len(word_list)):

            if i >= 3 and i-3 not in delete and i-2 not in delete and i-1 not in delete:
                quatgram = word_list[i-3] + word_list[i-2] + word_list[i-1] + word_list[i]
                if quatgram in self.quatgram_lexicons:
                    word_list[i-3] = quatgram
                    delete.append(i-2)
                    delete.append(i-1)
                    delete.append(i)
                    continue
            if i >= 2 and i-2 not in delete and i-1 not in delete:
                trigram = word_list[i-2] + word_list[i-1] + word_list[i]
                if trigram in self.trigram_lexicons:
                    word_list[i-2] = trigram
                    delete.append(i-1)
                    delete.append(i)
                    continue
            if i >= 1 and i-1 not in delete:
                bigram =  word_list[i-1] + word_list[i]
                if bigram in self.two_unary or bigram in self.bigram_lexicons:
                    word_list[i-1] = bigram
                    delete.append(i)
                bigram =  word_list[i-1] + ' ' + word_list[i]  # bigram with blank
                if bigram in self.bigram_lexicons:
                    word_list[i-1] = bigram
                    delete.append(i)
            
                
        for i in range(len(delete)-1, -1, -1):
            word_list.pop(delete[i])
        return word_list
        
if __name__ =='__main__':
        
    print('')
    
    
