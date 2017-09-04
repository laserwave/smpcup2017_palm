import codecs
import task1_tfidf_keyword_extractor
from util import load_validation_set,load_test_set, load_word_df, read_lines_as_list, read_doc_by_id, doc_name, load_word_count, load_task1_train_set
import task1_nn_keyword_classifier


def do_task_1_nn(task1_ids):
    keywordExtractor = task1_tfidf_keyword_extractor.KeywordExtractor()
    word_df = load_word_df()
    it_lexicons = read_lines_as_list('dicts/it_lexicons.txt', skip=1) # it专业名词
    neuralNetworkKeywordClassifier = task1_nn_keyword_classifier.NeuralNetworkKeywordClassifier()
    neuralNetworkKeywordClassifier.load_model('model/task1/0810_02_19_64.54.h5')
    nnFeatureGenerator = task1_nn_keyword_classifier.NeuralNetworkFeatureGenerator(it_lexicons, word_df)
    res = {}
    for doc_id in task1_ids:
        title, content = read_doc_by_id(doc_id)
        title_word, content_word, keywords = keywordExtractor._extract(title, content, alpha=0.22, num1=12, num2=16, delete=False, withscore=False, enrich=False)
        x = nnFeatureGenerator.gen_feature_of_doc_instance(doc_id, keywords, title_word, content_word)
        res[doc_id] = neuralNetworkKeywordClassifier.extractKeywords(x, keywords)
    return res
        
if __name__ =='__main__':
    task1_ids, task2_ids, task3_ids = load_test_set()
    res_task1 = do_task_1_nn(task1_ids)
    
    
    out = codecs.open('test_task1_2.txt', 'w', 'utf-8')
    for doc_id in task1_ids:
        keywords = res_task1[doc_id]
        sb = str(doc_id)
        for word in keywords:
            sb += ',' + word
        out.write(sb + '\n')
    out.close()




    
    
    
    
    
    
    
    
    
    
    
    

        
        
    
        
        
        