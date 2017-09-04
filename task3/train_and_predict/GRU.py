import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.engine import Input
from keras.engine import Model
from keras.engine import merge
from keras.layers import GRU, Dense, Dropout, Convolution1D, MaxPooling1D, Flatten, Bidirectional, regularizers, LSTM
from keras.models import Sequential
from CallBackMy import CallBackMy

TIMESEED = 8946
np.random.seed(TIMESEED)

max_len = 12
win = 3
batch_size = 15
data_path = '/home/ml/SMPCUP2017/'
BEST_MODEL_PATH = data_path + 'Model/best_model.h5'
SCORE_BEST_MODEL_PATH = data_path + 'Model/score_best_model_rnn_1.h5'

def score(target, pred):
    max_v = np.max(np.concatenate((target, pred), axis=-1),axis=-1)
    cha = np.fabs(target-pred)
    cha = cha.reshape(-1)
    n = target.shape[0]
    return 1-np.sum(cha/max_v, axis=0)/n

# def normalize(train_x1):
#     train_x1_max = np.max(train_x1, axis=0) + 0.0001
#     train_x1_max = train_x1_max.reshape(1, -1)
#     train_x1_max = np.repeat(train_x1_max, train_x1.shape[0], axis=0)
#     train_x = train_x1 / train_x1_max  # normalization
#     return train_x

def normalize(x):
    x_sum = np.sum(x, axis=0)+0.0001
    x_sum = x_sum.reshape(12,x_sum.shape[-1]/12)
    x_sum = np.sum(x_sum, axis=0)
    x_average = x_sum/(x.shape[0]*12)
    x_average = x_average.reshape(1,-1)
    x_average = np.repeat(x_average, 12, axis=0)
    x_average = x_average.reshape(1,-1)
    x_average = np.repeat(x_average, x.shape[0], axis=0)
    x_n = x / x_average  # normalization
    return x_n

train_x1 = np.loadtxt(data_path + 'task3_train_feature.txt')
# train_x1 = normalize(train_x1)
dim = train_x1.shape[1]/12
train_x1 = train_x1.reshape(-1, 12, dim)

train_x2 = np.loadtxt(data_path + 'task3_train_feature2.txt')
train_x =  [train_x1, train_x2] #train_x1#[train_x1, train_x2]



train_y = np.loadtxt(data_path + 'task3_train_tag.txt')
train_y = train_y.reshape(-1,1)

validation_x1 = np.loadtxt(data_path + 'task3_validation_feature.txt')
# validation_x1 = normalize(validation_x1)
validation_x1 = validation_x1.reshape(-1, 12, dim)

validation_x2 = np.loadtxt(data_path + 'task3_validation_feature2.txt')
validation_x = [validation_x1, validation_x2] # validation_x1 #  [validation_x1, validation_x2]

val_x1 = np.loadtxt(data_path + 'task3_train_feature_split.txt')
# val_x1 = normalize(val_x1)
val_x1 = val_x1.reshape(-1, 12, dim)
val_x2 = np.loadtxt(data_path + 'task3_train_feature2_split.txt')
val_x = [val_x1, val_x2]

val_y = np.loadtxt(data_path + 'task3_train_tag_split.txt')
val_y = val_y.reshape(-1,1)

test_x1 = np.loadtxt(data_path + 'task3_test_feature.txt')
# validation_x1 = normalize(validation_x1)
test_x1 = test_x1.reshape(-1, 12, dim)

test_x2 = np.loadtxt(data_path + 'task3_test_feature2.txt')
test_x = [test_x1, test_x2] # validation_x1 #  [validation_x1, validation_x2]

print train_x1.shape
print train_x2.shape
print train_y.shape

feature2_size = train_x2.shape[-1]

'''
#gru
'''
train_x = train_x1
val_x = train_x1 # train_x1  #  val_x1
val_y = train_y # train_y  # val_y
validation_x = validation_x1
test_x = test_x1
sequence_input = Input(shape=(max_len,dim))
seq = LSTM(150, input_shape=(max_len,dim),dropout_U=0.1, dropout_W=0.1, b_regularizer=regularizers.l2(0.0001))(sequence_input)  #,  b_regularizer=regularizers.l2(0.0001)
seq = Dense(100, activation = 'tanh')(seq)
# seq = Dropout(0.1)(seq)
out = Dense(1, activation = 'sigmoid')(seq)
model = Model(sequence_input, out)

'''
#gru two
'''
# sequence_input = Input(shape=(max_len,dim))
# feature_input = Input(shape=(feature2_size,))
# sequence = LSTM(150, input_shape=(max_len,dim),dropout_U=0.1, dropout_W=0.1, b_regularizer=regularizers.l2(0.0001))(sequence_input)
# feature2 = Dense(50, activation='relu', input_shape=(feature2_size,))(feature_input)
# feature2 = Dense(100, activation='relu')(feature2)
# feature2 = Dense(50, activation='tanh')(feature2)
# feature2 = Dense(100, activation='tanh')(feature2)
# # model.add(Dropout(0.5))
# x = merge([sequence, feature2], mode='concat', concat_axis=-1)
# x = Dense(100, activation = 'tanh')(x)
# out = Dense(1, activation = 'sigmoid')(x)
# model = Model([sequence_input, feature_input], out)

'''
#three layers for feature2
'''
# train_x = train_x2
# val_x = val_x2
# validation_x = validation_x2
# feature_input = Input(shape=(feature2_size,))
# x = Dense(50, activation='relu', input_shape=(feature2_size,))(feature_input)
# x = Dense(100,activation='relu')(x)
# x = Dense(50, activation='tanh')(x)
# x = Dense(100,activation='tanh')(x)
# out = Dense(1, activation = 'sigmoid')(x)
# model = Model(feature_input, out)

'''
#cnn
'''
# sequence_input = Input(shape=(max_len,dim))
# seq = Convolution1D(nb_filter=150,
#                             filter_length=win,
#                             activation='tanh')(sequence_input)
# seq = MaxPooling1D(pool_length = max_len - win +1)(seq)
# seq = Flatten()(seq)
# out = Dense(1, activation = 'sigmoid')(seq)
# model = Model(sequence_input, out)


model.compile(optimizer='adamax',
              loss = 'binary_crossentropy',   #binary_crossentropy
              metrics=['accuracy'])
model.summary()
model_name= 'GRU'
checkpointer = ModelCheckpoint(filepath=BEST_MODEL_PATH, monitor='loss', verbose=0, save_best_only=True)
mycallback = CallBackMy(test_array=val_x, #val_x1 #train_x
                        test_label=val_y, #val_y #train_y
                        path=data_path,
                        log_dict={'seed':TIMESEED, 'batch_size':batch_size,
                                'model':model_name, 'path':SCORE_BEST_MODEL_PATH})
# model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=500, verbose=0,
#           callbacks=[checkpointer, mycallback])

model.load_weights(SCORE_BEST_MODEL_PATH)

pred_validation = model.predict(validation_x, batch_size=5)
print pred_validation

validation_list = open(data_path + 'SMPCUP2017_ValidationSet_Task3.txt', 'r')
task3_validation_result = open(data_path + '/result/task3_validation_result.txt', 'w')
lines = validation_list.readlines()
if len(lines) != pred_validation.shape[0]:
    print 'wrong with length of validation set'
for i in range(len(lines)):
    task3_validation_result.write(lines[i].strip() + ',' + str(pred_validation[i][0]) + '\n')
task3_validation_result.close()
validation_list.close()

pred_test = model.predict(test_x, batch_size=5)

test_list = open(data_path + 'SMPCUP2017_TestSet_Task3.txt', 'r')
task3_test_result = open(data_path + '/result/task3_test_result.txt', 'w')
lines = test_list.readlines()
print(pred_test.shape)
print(len(lines))
if len(lines) != pred_test.shape[0]:
    print 'wrong with length of test set'
for i in range(len(lines)):
    # print(str(i) +'\t'+ lines[i] + '\t' + str(pred_test[i]))
    task3_test_result.write(lines[i].strip() + ',' + str(pred_test[i][0]) + '\n')
task3_test_result.close()
test_list.close()

pred = model.predict(train_x, batch_size=5)
result = score(train_y, pred)
print result

print 'code end'