# -*- coding: utf-8 -*-
import numpy as np
import codecs
from util import load_task1_train_set, read_doc_by_id
import task1_tfidf_keyword_extractor
import task1_nn_keyword_classifier
import time
import doc_preprocess
#np.random.seed(2264)  #2264   

    
    
'''
固定测试集的参数，验证训练集构造候选词参数的影响
'''
def validateTrainParams():
    task1_train_ids, docid2label = load_task1_train_set()
    keywordExtractor = task1_tfidf_keyword_extractor.KeywordExtractor()
    
    document_preprocessor = doc_preprocess.DocumentPreprocessor()    

    nnFeatureGenerator = task1_nn_keyword_classifier.NeuralNetworkFeatureGenerator(document_preprocessor.it_lexicons, document_preprocessor.word_df)
    
    # 使用固定的测试集参数
    id2test = {}
    for doc_id in task1_train_ids:
        title, content = read_doc_by_id(doc_id)
        #######################   False
        title_word, content_word, keywords = keywordExtractor._extract(title, content, alpha=0.22, num1=12, num2=16, delete=False, withscore=False, enrich=False)
        x = nnFeatureGenerator.gen_feature_of_doc_instance(doc_id, keywords, title_word, content_word)
        id2test[doc_id] = (x, keywords)
        
    
    neuralNetworkKeywordClassifier = task1_nn_keyword_classifier.NeuralNetworkKeywordClassifier()
    positive_instance = 0
    negative_instance = 0
    '''
    for alpha in [0.21, -15, -10]:
        for num1 in [7, 10, 12]:
            for num2 in [7, 10, 12]:
    '''
    # 训练集处理的参数组合
    for alpha in [0.21]:
        for num1 in [12]:
            for num2 in [16]:
                if num2 < num1:
                    continue
                for delete in [False]:         #######################################################################################
                    for append_groundtruth in [False]:
                        total_score = 0.0
                        instances = []
                        print('generate candidate keywords...')
                        for doc_id in task1_train_ids:
                            title, content = read_doc_by_id(doc_id)
                            score = 0.0
                            title_word, content_word, keywords = keywordExtractor._extract(title, content, alpha=alpha, num1=num1, num2=num2, delete=delete, withscore=False, enrich=False)
                            
                            for i in range(len(keywords)):
                                if keywords[i] in docid2label[doc_id]:
                                    score += 1 if score < 3 else 0
                            for word in keywords:
                                assert word != 'nan'
                                if word in docid2label[doc_id]:
                                    positive_instance += 1
                                else:
                                    negative_instance += 1
#                            if append_groundtruth:
#                                for word in docid2label[doc_id]:
#                                    if word in word_set and word not in keywords:
#                                        keywords.append(word)
#                                        score += 1 if score < 3 else 0
#                                        positive_instance += 1
                            total_score += score   
                            instances.append((doc_id, keywords, title_word, content_word))
                        
                        print('end')
                        print('alpha=' + str(alpha) + ', num1=' + str(num1) + ', num2=' + str(num2) + ', delete=' + str(delete) + ', append_groundtruth=' + str(append_groundtruth) + ', upper_bound_score=' + str(total_score / (3.0 * len(task1_train_ids))))
                        model_name = time.strftime('%m%d_%H_%M',time.localtime(time.time()))
                        out = codecs.open('debug/' + model_name + '.txt', 'w', 'utf-8')
                        out.write('------------------------------\n')
                        out.write('alpha=' + str(alpha) + ', num1=' + str(num1) + ', num2=' + str(num2) + ', delete=' + str(delete) + ', append_groundtruth=' + str(append_groundtruth) + ', upper_bound_score=' + str(total_score / (3.0 * len(task1_train_ids))) + '\n')
                        print('positive_instance', positive_instance)
                        print('negative_instance', negative_instance)
                        out.write('positive_instance' + ' ' + str(positive_instance) + '\n')
                        out.write('negative_instance' + ' ' + str(negative_instance) + '\n')
                        out.write('model_name = ' + ' ' + model_name + '\n')
                        x_train = nnFeatureGenerator.gen_feature_of_doc_instances(instances)
                        y_train = nnFeatureGenerator.gen_labels_of_doc_instances(instances, docid2label)
                        
                        
                        
                        neuralNetworkKeywordClassifier = task1_nn_keyword_classifier.NeuralNetworkKeywordClassifier()
                        neuralNetworkKeywordClassifier.train_model(x_train, y_train)
                        neuralNetworkKeywordClassifier.save_model('model/task1/' + model_name + '.h5')
                        neuralNetworkKeywordClassifier.summary()
                        total_score = 0.0
                        for doc_id in task1_train_ids:
                            x_test, keywords = id2test[doc_id]
                            keywords = neuralNetworkKeywordClassifier.extractKeywords(x_test, keywords)
#                            print(keywords)
                            score = 0.0
                            for i in range(len(keywords)):
                                if keywords[i] in docid2label[doc_id]:
                                    score += 1
                            print(score)
                            total_score += score
                        print('total_score=' + str(total_score / (3.0 * len(task1_train_ids))))
                        out.write('#test:' + ' ' + 'total_score=' + str(total_score / (3.0 * len(task1_train_ids))) + '\n')
                        out.close()
    
def validate():
    task1_train_ids, docid2label = load_task1_train_set()
    keywordExtractor = task1_tfidf_keyword_extractor.KeywordExtractor()
    
    document_preprocessor = doc_preprocess.DocumentPreprocessor()    

    nnFeatureGenerator = task1_nn_keyword_classifier.NeuralNetworkFeatureGenerator(document_preprocessor.it_lexicons, document_preprocessor.word_df)
    neuralNetworkKeywordClassifier = task1_nn_keyword_classifier.NeuralNetworkKeywordClassifier()
    neuralNetworkKeywordClassifier.load_model('F:/#smp17/#palm/model/task1/0809_20_57_65.36_62.16.h5')
    total_score = 0.0
    
    out = codecs.open('train_set_result.txt', 'w', 'utf-8')
    
    for doc_id in task1_train_ids:
        title, content = read_doc_by_id(doc_id)
        title_word, content_word, keywords = keywordExtractor._extract(title, content, alpha=0.22, num1=12, num2=16, delete=False, withscore=False, enrich=False)
        x = nnFeatureGenerator.gen_feature_of_doc_instance(doc_id, keywords, title_word, content_word)
        keywords = neuralNetworkKeywordClassifier.extractKeywords(x, keywords)        
        score = 0
        for i in range(len(keywords)):
            if keywords[i] in docid2label[doc_id]:
                score += 1
        total_score += score
        out.write(str(doc_id) + '\n')
        out.write(','.join(docid2label[doc_id]) + '\n')
        out.write(','.join(keywords) + '\n')
        out.write(str(score) + '\n')
        
        
    print('total_score=' + str(total_score / (3.0 * len(task1_train_ids))))
#    out.close()



def validate_real():
    task1_train_ids, docid2label = load_task1_train_set()
    keywordExtractor = task1_tfidf_keyword_extractor.KeywordExtractor()
    
    document_preprocessor = doc_preprocess.DocumentPreprocessor()    

    nnFeatureGenerator = task1_nn_keyword_classifier.NeuralNetworkFeatureGenerator(document_preprocessor.it_lexicons, document_preprocessor.word_df)
    
    # 使用固定的测试集参数
    id2test = {}
    for doc_id in task1_train_ids:
        title, content = read_doc_by_id(doc_id)
        #######################   False
        title_word, content_word, keywords = keywordExtractor._extract(title, content, alpha=0.22, num1=12, num2=16, delete=False, withscore=False, enrich=False)
        x = nnFeatureGenerator.gen_feature_of_doc_instance(doc_id, keywords, title_word, content_word)
        id2test[doc_id] = (x, keywords)
    sp = 0.8
    train_split = task1_train_ids[0:int(len(task1_train_ids) * sp)]
    test_split = task1_train_ids[int(len(task1_train_ids) * sp):]
    
    neuralNetworkKeywordClassifier = task1_nn_keyword_classifier.NeuralNetworkKeywordClassifier()
    positive_instance = 0
    negative_instance = 0
    total_score = 0.0
    instances = []
    print('generate candidate keywords...')
    for doc_id in train_split:
        title, content = read_doc_by_id(doc_id)
        score = 0.0
        title_word, content_word, keywords = keywordExtractor._extract(title, content, alpha=0.21, num1=12, num2=16, delete=False, withscore=False, enrich=False)
        
        for i in range(len(keywords)):
            if keywords[i] in docid2label[doc_id]:
                score += 1 if score < 3 else 0
        for word in keywords:
            assert word != 'nan'
            if word in docid2label[doc_id]:
                positive_instance += 1
            else:
                negative_instance += 1
        total_score += score   
        instances.append((doc_id, keywords, title_word, content_word))
    
    print('positive_instance', positive_instance)
    print('negative_instance', negative_instance)
    x_train = nnFeatureGenerator.gen_feature_of_doc_instances(instances)
    y_train = nnFeatureGenerator.gen_labels_of_doc_instances(instances, docid2label)
    
    
    
    neuralNetworkKeywordClassifier = task1_nn_keyword_classifier.NeuralNetworkKeywordClassifier()
    neuralNetworkKeywordClassifier.train_model(x_train, y_train)
    neuralNetworkKeywordClassifier.summary()
    total_score = 0.0
    
    for doc_id in test_split:
        x_test, keywords = id2test[doc_id]
        keywords = neuralNetworkKeywordClassifier.extractKeywords(x_test, keywords)
        score = 0.0
        for i in range(len(keywords)):
            if keywords[i] in docid2label[doc_id]:
                score += 1
        print(score)
        total_score += score
    print('total_score=' + str(total_score / (3.0 * len(test_split))))


if __name__ =='__main__': 
#    validateTrainParams()
    validate()
#    validate_real()
    
    
    
        
        
        

        
    

