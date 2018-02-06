#对预测模块进行预先训练，并参与全局训练

from keras.layers import Input, Embedding, LSTM, Dense,CuDNNLSTM,Dropout
from keras.models import Model
import keras,os
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
dataFile = 'sinr.mat' #包含细粒度的SINR数据(每个速度点仿真5s)      sinrlstm2.wts.h5
data = scio.loadmat(dataFile)
sinr_temp = data['sinr']

data_dim = 1
timesteps = 20

data_dim2 = 10
pre_step = 10

batch_size = 256
epochs = 100

num_classes = 5 #总类别数 可调参数
class_range = int(150/num_classes) #每个类别的速度范围
num_samples = class_range*(5000-data_dim*timesteps-pre_step+1) #每个类别的样本数

sinr_input1 = np.zeros([num_classes*num_samples, timesteps, data_dim])
sinr_input2 = np.zeros([num_classes*num_samples, data_dim2])
sinr_output = np.zeros([num_classes*num_samples, 1])
sinr_class = np.zeros([num_classes*num_samples, 1])

for i in range(0, num_classes):
    for j in range(0, class_range):
        for k in range(0, (5000-data_dim*timesteps-pre_step+1)):
            data1 = sinr_temp[((i * class_range + j) * 5000 + k): ((i * class_range + j) * 5000 + k + data_dim * timesteps)]
            data2 = sinr_temp[((i * class_range + j) * 5000 + k+data_dim*timesteps-data_dim2): ((i * class_range + j) * 5000 + k + data_dim*timesteps)]
            sinr_input1[(i*class_range+j)*(5000-data_dim*timesteps-pre_step+1)+k, :, :] = np.reshape(data1, [timesteps, data_dim])
            sinr_input2[(i*class_range+j)*(5000-data_dim*timesteps-pre_step+1)+k, :] = np.reshape(data2, [1, data_dim2])
            sinr_output[(i*class_range+j)*(5000-data_dim*timesteps-pre_step+1)+k] = sinr_temp[(i*class_range+j)*5000+k+data_dim*timesteps-1+pre_step]
            sinr_class[(i * class_range + j) * (5000 - data_dim * timesteps - pre_step + 1) + k] = i

# Generate dummy training data
np.random.seed(2016)
n_examples = sinr_output.shape[0]
n_train = int(0.8*n_examples)  # 三分之二的数据用于训练
train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
test_idx = list(set(range(0, n_examples)) - set(train_idx))
# train_idx = np.arange(n_train,n_examples)
# test_idx = np.arange(0,n_train)

x_train1 = sinr_input1[train_idx, :, :]
x_train2 = sinr_input2[train_idx, :]
y_train = sinr_output[train_idx, :]
y_train_class = sinr_class[train_idx, :]# 训练的时候，使用类别标签

x_test1 = sinr_input1[test_idx, :, :]
x_test2 = sinr_input2[test_idx, :]
y_test = sinr_output[test_idx, :]
y_test_class = sinr_class[test_idx, :]# 测试的时候，使用类别标签

# classifaction module
input1 = Input(shape=(timesteps, data_dim, ), dtype='float32', name='input1')
lstm_hidden1 = LSTM(100, kernel_initializer='glorot_uniform', return_sequences=False, stateful=False, input_shape=sinr_input1.shape[1:])(input1)
# lstm_hidden2 = CuDNNLSTM(16, kernel_initializer='glorot_uniform', return_sequences=True, stateful=False)(lstm_hidden1)
# lstm_hidden3 = CuDNNLSTM(8, kernel_initializer='glorot_uniform', return_sequences=False, stateful=False)(lstm_hidden2)
lstm_hidden2 = Dense(64, activation='relu', kernel_initializer='he_normal')(lstm_hidden1)
lstm_hidden3 = Dense(32, activation='relu', kernel_initializer='he_normal')(lstm_hidden2)
lstm_hidden4 = Dropout(0.5)(lstm_hidden3)
lstm_out = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(lstm_hidden4)
print(lstm_out.shape)

# 5 prediction modules
input2 = Input(shape=(data_dim2, ), dtype='float32', name='input2')
sinr_hidden1 = Dense(15,activation='tanh')(input2)
sinr_hidden2 = Dense(10,activation='tanh')(sinr_hidden1)
sinr_hidden3 = Dense(5,activation='tanh')(sinr_hidden2)
sinr_output2_1 = Dense(1,activation='linear')(sinr_hidden3)
model1 = Model(inputs=input2, outputs=sinr_output2_1)
# model1.load_weights(filepath='sinrprediction-fnn3kmph.wts.h5')
model1.load_weights(filepath='models/sinrprediction-fnn1.5~4.5kmph.wts.h5')

sinr_hidden1 = Dense(15,activation='tanh')(input2)
sinr_hidden2 = Dense(10,activation='tanh')(sinr_hidden1)
sinr_hidden3 = Dense(5,activation='tanh')(sinr_hidden2)
sinr_output2_2 = Dense(1,activation='linear')(sinr_hidden3)
model2 = Model(inputs=input2, outputs=sinr_output2_2)
# model2.load_weights(filepath='sinrprediction-fnn6kmph.wts.h5')
model2.load_weights(filepath='models/sinrprediction-fnn4.5~7.5kmph.wts.h5')

sinr_hidden1 = Dense(15,activation='tanh')(input2)
sinr_hidden2 = Dense(10,activation='tanh')(sinr_hidden1)
sinr_hidden3 = Dense(5,activation='tanh')(sinr_hidden2)
sinr_output2_3 = Dense(1,activation='linear')(sinr_hidden3)
model3 = Model(inputs=input2, outputs=sinr_output2_3)
# model3.load_weights(filepath='sinrprediction-fnn9kmph.wts.h5')
model3.load_weights(filepath='models/sinrprediction-fnn7.5~10.5kmph.wts.h5')

sinr_hidden1 = Dense(15,activation='tanh')(input2)
sinr_hidden2 = Dense(10,activation='tanh')(sinr_hidden1)
sinr_hidden3 = Dense(5,activation='tanh')(sinr_hidden2)
sinr_output2_4 = Dense(1,activation='linear')(sinr_hidden3)
model4 = Model(inputs=input2, outputs=sinr_output2_4)
# model4.load_weights(filepath='sinrprediction-fnn12kmph.wts.h5')
model4.load_weights(filepath='models/sinrprediction-fnn10.5~13.5kmph.wts.h5')

sinr_hidden1 = Dense(15,activation='tanh')(input2)
sinr_hidden2 = Dense(10,activation='tanh')(sinr_hidden1)
sinr_hidden3 = Dense(5,activation='tanh')(sinr_hidden2)
sinr_output2_5 = Dense(1,activation='linear')(sinr_hidden3)
model5 = Model(inputs=input2, outputs=sinr_output2_5)
# model5.load_weights(filepath='sinrprediction-fnn15kmph.wts.h5')
model5.load_weights(filepath='models/sinrprediction-fnn13.5~16.5kmph.wts.h5')

sinr_output2_ = keras.layers.concatenate([sinr_output2_1, sinr_output2_2, sinr_output2_3, sinr_output2_4, sinr_output2_5], axis=1)
print(sinr_output2_.shape)

# 软合并预测结果
sinr_prediction = keras.layers.dot([lstm_out, sinr_output2_], axes=1)
model = Model(inputs=[input1, input2], outputs=sinr_prediction)

# 冻结预测模块
# layeridx = np.arange(3, 8)
# layeridx = np.append(layeridx, np.arange(9, 14))
# layeridx = np.append(layeridx, np.arange(15, 20))
# layeridx = np.append(layeridx, np.arange(21, 26))
# # layeridx = np.arange(26)
# for i in layeridx:
#     model.layers[i].trainable = False

# 打印模型概况
model.summary()

# 为样本赋予新的预测模块标签，并进行分类模块的与训练
def to_onehot(y):
    y = list(y)
    y_onehot = np.zeros([len(y), max(y) + 1])
    y_onehot[np.arange(len(y)), y] = 1
    return y_onehot


model_prediction = Model(inputs=input2, outputs=sinr_output2_)
a_ = model_prediction.predict(x_train2, batch_size=batch_size)
error_ = abs(a_ - np.repeat(y_train, num_classes, axis=1))
idx = np.argmin(error_, axis=1)
y_train_classidx = to_onehot(idx)

# 训练分类模块
model_class = Model(inputs=input1, outputs=lstm_out)

model_class.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history_class = model_class.fit(x_train1, y_train_classidx,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    shuffle=False,
                    validation_split=0.2,
                    callbacks=[keras.callbacks.ModelCheckpoint('models/sinr_classification_'+str(num_classes)+'classes.h5', monitor='val_acc', verbose=0,
                                                               save_best_only=True,
                                                               mode='auto')]
                    )
model_class.load_weights('models/sinr_classification_'+str(num_classes)+'classes.h5')

#全局训练
modelmark = '5-8-6'
filepath = 'models/sinr_classification_prediction'+modelmark+'.wts.h5'
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mse', optimizer=adam)
history = model.fit([x_train1, x_train2], y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    shuffle=False,
                    validation_split=0.2,
                    callbacks=[keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
                                                               save_best_only=True,
                                                               mode='auto')]
                    )
model.load_weights(filepath)
predicted_output = model.predict([x_test1, x_test2], batch_size=batch_size)

# 记录中间结果
np.savetxt('record/y_test-'+modelmark+'.csv', y_test, delimiter=',')#记录sinr
np.savetxt('record/predicted_output-'+modelmark+'.csv', predicted_output, delimiter=',')#记录模型输出
np.savetxt('record/y_test_class-'+modelmark+'.csv', y_test_class, delimiter=',')#记录类别

classfiy_module_output = model_class.predict(x_test1, batch_size=batch_size)
predict_module_output = model_prediction.predict(x_test2, batch_size=batch_size)
np.savetxt('record/classfiy_module_output-'+modelmark+'.csv', classfiy_module_output, delimiter=',')#记录分类结果
np.savetxt('record/predict_module_output-'+modelmark+'.csv', predict_module_output, delimiter=',')#记录分类结果

# Show loss curves
# plt.figure()
# # plt.subplot(3, 1, 1)
# plt.title('Training performance')
# plt.plot(history.epoch, history.history['loss'], label='train loss')
# plt.plot(history.epoch, history.history['val_loss'], label='validation loss')
# plt.legend()
# plt.xlabel('epoch')

plt.figure()
# plt.subplot(3, 1, 2)
plt.scatter(y_test, predicted_output)
plt.xlabel('true sinr')
plt.ylabel('predicted sinr')

plt.figure()
error = (y_test - predicted_output)
print('MAE',np.mean(np.abs(error)))
print('RMSE',np.sqrt(np.mean(np.power(error, 2))))
# plt.subplot(3, 1, 3)
plt.plot(y_test,label='true')
plt.plot(predicted_output,label='predicted')
plt.legend()

plt.figure()
plt.hist(error, bins=1000, cumulative=True, normed=1, histtype='step', label='model3')
plt.xlabel('error')
plt.legend()

plt.show()




