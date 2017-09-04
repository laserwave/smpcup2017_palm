import codecs
import numpy as np
import os
import pickle


'''
加载单词的文档频率
'''
def load_word_df():
    path = 'dicts/csdn.word.df.pickle'
    if os.path.exists(path):
        word_df = pickle.load(open(path, 'rb'))
    else:
        raise Exception(path + ' not found!')
    return word_df
    
'''
加载单词计数
'''
def load_word_count():
    path = 'dicts/word_count.pickle'
    if os.path.exists(path):
        word_count = pickle.load(open(path, 'rb'))
    else:
        raise Exception(path + ' not found!')
    return word_count
    
'''
加载ngram
'''
def load_ngram_count():
    
    ngram_count = {}
    lines = read_lines_as_list('dicts/ngram_count.txt')
    for line in lines:
        c = line.split()
        word = c[0]
        count = int(c[1])
        ngram_count[word] = count
    
    return ngram_count
    
'''
加载单词在标题中的计数
'''
def load_title_word_count():
    path = 'dicts/title_word_count.pickle'
    if os.path.exists(path):
        title_word_count = pickle.load(open(path, 'rb'))
    else:
        raise Exception(path + ' not found!')
    return title_word_count

    

'''
加载词典(list)
'''
def read_lines_as_list(path, skip=0):
    return [line.strip() for line in codecs.open(path, 'r', 'utf-8') if len(line.strip()) > 0][skip:]

'''
加载词典(set)
'''
def read_lines_as_set(path, skip=0):
    return set(read_lines_as_list(path, skip))
 
'''
文档全名
'''
def doc_name(num):
    s = ""
    for i in range(7 - len(str(num))):
        s += '0'
    return  "D" + s + str(num);

'''
加载task1训练集
'''
def load_task1_train_set():

    train_set_path = 'data/train_task_1/SMPCUP2017_TrainingData_Task1.txt'
    train_ids = []
    docid2label = {}
    for line in codecs.open(train_set_path, 'r', 'utf-8'):
        line = line.strip()
        if len(line) < 1:
            continue
        split = line.split('\001')
        doc_id = split[0]
        label = [keyword for keyword in split[1:]]
        docid2label[doc_id] = label
        train_ids.append(doc_id)
    return train_ids, docid2label

'''
加载task2训练集
'''
def load_task2_train_set():
    
    train_set_path = 'data/train_task_2/SMPCUP2017_TrainingData_Task2.txt'
    train_ids = []
    userid2label = {}
    for line in codecs.open(train_set_path, 'r', 'utf-8'):
        line = line.strip()
        if len(line) < 1:
            continue
        split = line.split('\001')
        user_id = split[0]
        label = [tag for tag in split[1:]]
        userid2label[user_id] = label
        train_ids.append(user_id)
    return train_ids, userid2label
    
'''
加载task2标签
'''
def load_task2_tags():
    tag2id, id2tag = pickle.load(open('data/tags.pickle', 'rb'))
    return tag2id, id2tag
    
'''
加载3个任务的验证集编号
'''
def load_validation_set():
    root_path = 'data/validation_set/SMPCUP2017_ValidationSet_Task'
    task1 = root_path + '1.txt'
    task2 = root_path + '2.txt'
    task3 = root_path + '3.txt'
    
    task1_ids = read_lines_as_list(task1)
    task2_ids = read_lines_as_list(task2)
    task3_ids = read_lines_as_list(task3)
        
    return task1_ids, task2_ids, task3_ids

    
def load_test_set():
    root_path = 'data/test_set/SMPCUP2017_TestSet_Task'
    task1 = root_path + '1.txt'
    task2 = root_path + '2.txt'
    task3 = root_path + '3.txt'
    
    task1_ids = read_lines_as_list(task1)
    task2_ids = read_lines_as_list(task2)
    task3_ids = read_lines_as_list(task3)
        
    return task1_ids, task2_ids, task3_ids
    
'''
读取文章的标题和正文
'''
def read_doc_by_id(doc_id):
    file_path = 'data/seg_data/' + doc_id + '.txt'
    lines = read_lines_as_list(file_path)
    title = lines[0] if len(lines) > 0 else ''
    content = ' '.join(lines[1:]) if len(lines) > 1 else ''
    return title, content
    


    
'''
加载用户行为数据
'''
def load_user_data():
    post = pickle.load(open('data/post.pickle', 'rb'))
    browse = pickle.load(open('data/browse.pickle', 'rb'))
    comment = pickle.load(open('data/comment.pickle', 'rb'))
    vote_up = pickle.load(open('data/vote_up.pickle', 'rb'))
    vote_down = pickle.load(open('data/vote_down.pickle', 'rb'))
    favorite = pickle.load(open('data/favorite.pickle', 'rb'))
    user_activity = {'post':post, 'browse':browse, 'comment':comment, 'vote_up':vote_up, 'vote_down':vote_down, 'favorite':favorite}
    
    return user_activity
    
