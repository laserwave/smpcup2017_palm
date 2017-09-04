from __future__ import print_function
import time
from keras.callbacks import Callback
import numpy as np

def score(target, pred):
    max_v = np.max(np.concatenate((target, pred), axis=-1),axis=-1)+0.0000000001
    cha = np.fabs(target-pred)
    cha = cha.reshape(-1)
    n = target.shape[0]
    return 1-np.sum(cha/max_v, axis=0)/n

class CallBackMy(Callback):
    def __init__(self, test_array, test_label, path, log_dict={}
                 ):
        super(CallBackMy, self).__init__()
        self.test_array = test_array
        self.test_label = test_label
        self.path = path
        self.log_dict = log_dict
        self.best_socre = 0.0

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.test_array)
        # print('the shape of pred: ', pred.shape)
        score3 = score(self.test_label, pred)
        print(str(epoch+1) + '/epoch\n')
        print('----score:\t', score3, '  \t----best score:\t', self.best_socre)
        if float(score3) > self.best_socre:
            self.model.save(self.log_dict['path'])
            self.best_socre = float(score3)

        filename = str(self.path) + 'epochs_result/'+ str(self.log_dict['model']) +'_seed_' \
                   + str(self.log_dict['seed']) +'_batch_'+str(self.log_dict['batch_size'])+'.txt'
        with open(filename, 'a')as wr:
            wr.write(str(epoch+1) + '/epoch\n')
            wr.write(str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))+'\n')
            wr.write('Score3: '+str(score3)+'\n\n')
