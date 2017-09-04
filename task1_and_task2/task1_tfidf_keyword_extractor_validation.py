# -*- coding: utf-8 -*-
import tfidf_keyword_extractor
from util import read_doc_by_id, load_task1_train_set

'''
验证tfidf关键词提取器在训练集上的效果
超参数：
    alpha: [0.21, -15, -10]
    num1: 3
    num2: [6, 10, 15]
    delete: [True, False]
'''    
if __name__ =='__main__':

    task1_train_ids, docid2label = load_task1_train_set()
    keywordExtractor = tfidf_keyword_extractor.KeywordExtractor()
    for alpha in [0.21, -15, -10]:
        for num2 in [6, 10, 15]:
            for delete in [True, False]:
                
                total_score = 0.0
                for doc_id in task1_train_ids:
                    title, content = read_doc_by_id(doc_id)
                    score = 0.0
                    keywords = keywordExtractor.extract(title, content, alpha=alpha, num1=3, num2=num2, delete=delete, withscore=False, enrich=True)
                    assert len(keywords) == 3
                    for i in range(len(keywords)):
                        if keywords[i] in docid2label[doc_id]:
                            score += 1
                    total_score += score
                print('alpha=' + str(alpha) + ', num2=' + str(num2) + ', delete=' + str(delete) + ', score=' + str(total_score / (3.0 * len(task1_train_ids))))

'''
alpha -> 0.21  num2 -> 10  delete -> True  score=0.5391389432485323

2017-07-17 19:41
alpha=0.21, num2=6, delete=True, score=0.5371819960861057
alpha=0.21, num2=6, delete=False, score=0.5283757338551859
alpha=0.21, num2=10, delete=True, score=0.5391389432485323
alpha=0.21, num2=10, delete=False, score=0.530658838878017
alpha=0.21, num2=15, delete=True, score=0.5352250489236791
alpha=0.21, num2=15, delete=False, score=0.5296803652968036
alpha=-15, num2=6, delete=True, score=0.5290280495759948
alpha=-15, num2=6, delete=False, score=0.5215264187866928
alpha=-15, num2=10, delete=True, score=0.5319634703196348
alpha=-15, num2=10, delete=False, score=0.5234833659491194
alpha=-15, num2=15, delete=True, score=0.5296803652968036
alpha=-15, num2=15, delete=False, score=0.5257664709719504
alpha=-10, num2=6, delete=True, score=0.5264187866927593
alpha=-10, num2=6, delete=False, score=0.5176125244618396
alpha=-10, num2=10, delete=True, score=0.5287018917155903
alpha=-10, num2=10, delete=False, score=0.5198956294846706
alpha=-10, num2=15, delete=True, score=0.5267449445531638
alpha=-10, num2=15, delete=False, score=0.5221787345075016
'''