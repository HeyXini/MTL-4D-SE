from netCDF4 import Dataset
from utils import *
from conv4d import *
from keras.optimizers import Adam
import os
import numpy as np

def get_list(path):  # 此函数读取特定文件夹下的文件，返回图片所在路径的列表
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.nc')]


c = get_list(r"temperature")
a = get_list(r"salt")
# b = get_list(r"geostrophic current".)
cd = len(c)
cd1 = len(a)
# cd2=len(b)
d = cd
d1 = cd1
# d2 = cd2
map_height, map_width = 25,20

# data_T = np.empty((d,  map_height,3, map_width), dtype=float)
# data_S = np.empty((d1, map_height,3, map_width), dtype=float)
data_T = np.empty((d1, 25, 20,3), dtype=float)
data_S = np.empty((d1, 25, 20,3), dtype=float)

while d > 0:
    dataset = Dataset(c[cd - d], mode='r', format="NETCDF4")
    dataset1 = Dataset(a[cd1 - d1], mode='r', format="NETCDF4")
    # dataset2 = Dataset(b[cd2 - d2], mode='r', format="NETCDF4")
    T = dataset.variables['T'][235:260, 325:345, 0:3]#P1 0 2:10 8:50 10:100 18:400 20:500 25:1000 42:5000
    # print(dataset.variables['latitude'][165:185])390:400, 185:188, 0:37
    # print(dataset.variables['longitude'][368:388])
    # print(dataset1.variables['depth'][34:37])
    S = dataset1.variables['Salinity'][235:260, 325:345, 0:3]#14:17(200-300),20:23(500-700),25:28(1k-1.2k),34:37(1.8k-2k)
    # U = dataset2.variables['U'][216:241, 265:285,0]#p1:221:241, 255:275 p2:300:320, 220:240 p3:317:337, 389:409 p4:240:260, 325:345  p5:368:388, 165:185
    # V = dataset2.variables['V'][216:241, 265:285,0]
    # # for i in range(0, 409):
    #     for j in range(0,497):
    #             if T[i,j] ==T[i,j]:
    #                 print(i,j,T[i,j])
    #             if S[i,j]==S[i,j]:
    #                 print('V',i,j,S[i,j])
    T = np.asarray(T)
    S = np.asarray(S)
    # U = np.asarray(U)
    # V = np.asarray(V)
    data_T[cd - d] = T
    data_S[cd1 - d1] = S
    # data_U[cd2- d2] = U
    # data_V[cd2 - d2] = V
    print(cd - d, '% 546')
    d = d-1
    d1 = d1-1
    # d2 = d2-1


len_closeness = 6
len_test = 100
trend_l = 539


mmn = MinMaxNormalization()
data_T_train = data_T[:-len_test]
mmn.fit(data_T_train)
data_T_all = [data_T]
data_T_mmn = []
for d in data_T_all:
    data_T_mmn.append(mmn.transform(d))
data_T_mmn = np.asarray(data_T_mmn)
# data_T_mmn = data_T_mmn.reshape((546, map_height,map_width,3))
data_T_mmn = data_T_mmn.reshape((546, map_height,3,map_width))

mmn1 = MinMaxNormalization()
data_S_train = data_S[:-len_test]
mmn1.fit(data_S_train)
data_S_all = [data_S]
data_S_mmn = []
for d in data_S_all:
    data_S_mmn.append(mmn1.transform(d))
data_S_mmn = np.asarray(data_S_mmn)
# data_S_mmn = data_S_mmn.reshape((546, map_height,map_width,3))
data_S_mmn = data_S_mmn.reshape((546, map_height,3,map_width))

A = []
A1 = []
B = []
B1 = []
for k in range(0,trend_l):
    a = np.empty((len_closeness, map_height, 3, map_width), dtype=float)
    b = np.empty((len_closeness, map_height, 3, map_width), dtype=float)
    for j in range(1, len_closeness+1):
        a[len_closeness - j] = data_T_mmn[len_closeness+k-j]
        b[len_closeness - j] = data_S_mmn[len_closeness + k - j]
    a = a[np.newaxis, :]
    a=np.transpose(a,(0,1,2,4,3))
    b = b[np.newaxis, :]
    b = np.transpose(b,(0,1,2,4,3))
    A.append(a)
    B.append(b)
    a=[]
    b = []

    a1 = []
    b1 = []
    a1.append(data_T_mmn[len_closeness + k:len_closeness+k+2])
    b1.append(data_S_mmn[len_closeness + k:len_closeness+k+2])
    a1 = np.asarray(a1)
    a1 = np.transpose(a1,(0,1,2,4,3))
    b1 = np.asarray(b1)
    b1 = np.transpose(b1,(0,1,2,4,3))
    A1.append(a1)
    B1.append(b1)

A = np.asarray(A)
A1 = np.asarray(A1)
B = np.asarray(B)
B1 = np.asarray(B1)

T_train,TY_train = A[:-len_test],   A1[:-len_test]
T_test, TY_test = A[-len_test:],   A1[-len_test:]
TY_test1 = mmn.inverse_transform(TY_test)

S_train,SY_train = B[:-len_test],   B1[:-len_test]
S_test, SY_test = B[-len_test:],   B1[-len_test:]
SY_test1 = mmn1.inverse_transform(SY_test)

xtrain = np.concatenate((T_train, S_train),axis=1)
xtest = np.concatenate((T_test, S_test),axis=1)
ytrain = np.concatenate((TY_train, SY_train),axis=1)
ytest = np.concatenate((TY_test, SY_test),axis=1)
ytest1 = np.concatenate((TY_test1, SY_test1),axis=1)

print('train shape:', xtrain.shape, ytrain.shape,'test shape:', xtest.shape,ytest.shape)



epochs = 2000  # number of epoch at training stage
epoch_cont = 150  # number of epoch at training (cont) stage
batch_size = 32  # batch size
lr = 0.0005# learning rate
len_closeness = 6  # length of closeness dependent sequence
nb_flow = 2  # there are two types of flows: new-flow and end-flow
d = 3
# map_height, map_width = 25,25  # grid size


c_conf = (nb_flow, len_closeness,map_height, map_width,d) if len_closeness > 0 else None
model = conv4d(c_conf=c_conf)
adam = Adam(lr=lr)
model.compile(loss='mae', optimizer=adam, metrics=[rmse])
# model.compile(loss={'reshape_5':'mae','reshape_6':'mae','reshape_7':'mse','reshape_8':'mse'},loss_weights={'reshape_5':1,'reshape_6':1,'reshape_7':1,'reshape_8':1}, optimizer=adam, metrics=[rmse])
model.summary()

from keras.callbacks import EarlyStopping, ModelCheckpoint
hyperparams_name = 'lr{}'.format(lr)
fname_param = '{}.best.h5'.format(hyperparams_name)

early_stopping = EarlyStopping(monitor='val_rmse', patience=50, mode='min')#mode: 就’auto’, ‘min’, ‘,max’三个可能,按上升下降选择
model_checkpoint = ModelCheckpoint(fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')
#
print('=' * 10)
print("training model...")
history = model.fit(xtrain, ytrain,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,#用来指定训练集的一定比例数据作为验证集
                    callbacks=[early_stopping, model_checkpoint],
                    verbose=1)
model.save_weights('{}.h5'.format(hyperparams_name), overwrite=True)
#
print('=' * 10)
print('evaluating using the model that has the best loss on the valid set')
score = model.evaluate(xtrain, ytrain,  verbose=0)
print(score)
print('Train loss: %.6f  rmse (real): %.6f' %(score[0], score[1] ))
score = model.evaluate(xtest, ytest, verbose=0)
print('Test loss: %.6f  rmse (real): %.6f' %(score[0], score[1] ))

y_pred = model.predict((xtest))
y1 = y_pred[:,0,:,:]
y1 = np.ravel(y1,'F')
y1 = mmn.inverse_transform(y1)
y2 = y_pred[:,1,:,:]
y2 = np.ravel(y2,'F')
y2 = mmn1.inverse_transform(y2)
yt1 = ytest1[:,0,:,:]
yt1 = np.ravel(yt1,'F')
yt2 = ytest1[:,1,:,:]
yt2 = np.ravel(yt2,'F')
from sklearn.metrics import mean_absolute_error,mean_squared_error
rmse1 = mean_squared_error(yt1, y1) ** 0.5
rmse2 = mean_squared_error(yt2, y2) ** 0.5
print('rmse temp',rmse1)
print('rmse salt',rmse2)
mae1 = mean_absolute_error(yt1, y1)
mae2 = mean_absolute_error(yt2, y2)
print('mae temp',mae1)
print('mae salt',mae2)
def mean_relative_error(y_true, y_pred):
    import numpy as np
    relative_error = np.average(np.abs(y_true - y_pred) / y_true)
    return relative_error
re1 = mean_relative_error(yt1,y1)
re2 = mean_relative_error(yt2,y2)
acc1 = 1-re1
acc2 = 1-re2
print("acc temp",acc1)
print("acc salt",acc2)
from sklearn.metrics import r2_score
R21 = r2_score( yt1,y1)
R22 = r2_score( yt2,y2)
print('TR2',R21)
print('SR2',R22)

print('=' * 10)
print("training model (cont)...")
fname_param = os.path.join('D:\学习资料\温度预测\MODEL', '{}.cont.best.h5'.format(hyperparams_name))
model_checkpoint = ModelCheckpoint(fname_param, monitor='rmse', verbose=0, save_best_only=True, mode='min')
history = model.fit(xtrain, ytrain, epochs=epoch_cont, verbose=1, batch_size=batch_size, callbacks=[model_checkpoint], validation_data=(xtest, ytest))
model.save_weights('{}_cont.h5'.format(hyperparams_name), overwrite=True)
# fname_param ='{}.h5'.format(hyperparams_name)
# fname_param ='{}_cont.h5'.format(hyperparams_name)
# model.load_weights(fname_param)
print('=' * 10)
print('evaluating using the final model')
score = model.evaluate(xtrain, ytrain, verbose=0)
print('Train loss: %.6f  rmse (real): %.6f' %
       (score[0], score[1]))
score = model.evaluate(xtest, ytest, verbose=0)
print('Test loss: %.6f  rmse (real): %.6f' %
       (score[0], score[1]))



y_pred = model.predict((xtest))
y1 = y_pred[:,0]
y1 = np.ravel(y1,'F')
y1 = mmn.inverse_transform(y1)
y2 = y_pred[:,1]
y2 = np.ravel(y2,'F')
y2 = mmn1.inverse_transform(y2)
# y3 = y_pred[:,2,:,:]
# y3 = np.ravel(y3,'F')
# y3 = mmn2.inverse_transform(y3)
# y4 = y_pred[:,3,:,:]
# y4 = np.ravel(y4,'F')
# y4 = mmn3.inverse_transform(y4)
yt1 = ytest1[:,0]
yt1 = np.ravel(yt1,'F')
yt2 = ytest1[:,1]
yt2 = np.ravel(yt2,'F')
# yt3 = ytest1[:,2,:,:]
# yt3 = np.ravel(yt3,'F')
# yt4 = ytest1[:,3,:,:]
# yt4 = np.ravel(yt4,'F')


from sklearn.metrics import mean_absolute_error,mean_squared_error

rmse1 = mean_squared_error(yt1, y1) ** 0.5
rmse2 = mean_squared_error(yt2, y2) ** 0.5
# rmse3 = mean_squared_error(yt3, y3) ** 0.5
# rmse4 = mean_squared_error(yt4, y4) ** 0.5
print('rmse temp',rmse1)
print('rmse salt',rmse2)
# print('rmse U',rmse3)
# print('rmse V',rmse4)

mae1 = mean_absolute_error(yt1, y1)
mae2 = mean_absolute_error(yt2, y2)
# mae3 = mean_absolute_error(yt3, y3)
# mae4 = mean_absolute_error(yt4, y4)
print('mae temp',mae1)
print('mae salt',mae2)
# print('mae U',mae3)
# print('mae V',mae4)

def mean_relative_error(y_true, y_pred):
    import numpy as np
    relative_error = np.average(np.abs(y_true - y_pred) / y_true)
    return relative_error
re1 = mean_relative_error(yt1,y1)
re2 = mean_relative_error(yt2,y2)
acc1 = 1-re1
acc2 = 1-re2
print("acc temp",acc1)
print("acc salt",acc2)

from sklearn.metrics import r2_score
R21 = r2_score( yt1,y1)
R22 = r2_score( yt2,y2)
# R23 = r2_score( yt3,y3)
# R24 = r2_score( yt4,y4)
print('TR2',R21)
print('SR2',R22)
# print('UR2',R23)
# print('VR2',R24)
# # #
# dataset = Dataset(r'201708-201808\Temperature_20180801.nc', mode='r', format="NETCDF4")
# lat = dataset.variables['latitude'][150:153]
# lon = dataset.variables['longitude'][399:409]
# dep = dataset.variables['depth'][0:37]
# lon2 = dataset.variables['longitude'][389:409]
# #
# for j in range(0, 99):
#     f_w = Dataset('D/pm-{}.predict.nc'.format(timestamp_test[-2-j]), 'w',
#                   format='NETCDF4')  # 创建一个格式为.nc的，名字为 ‘hecheng.nc’的文件
#     f_r = Dataset('D/pm-{}.real.nc'.format(timestamp_test[-2-j]), 'w',
#                   format='NETCDF4')
#     f_h = Dataset('D/pm-{}.p-r.nc'.format(timestamp_test[-2 - j]), 'w',
#                   format='NETCDF4')
# #
#     f_w.createDimension('latitude', 3)
#     f_w.createDimension('longitude', 10)
#     f_w.createDimension('depth', 37)
#     f_r.createDimension('latitude', 3)
#     f_r.createDimension('longitude', 10)
#     f_r.createDimension('depth', 37)
#     f_h.createDimension('latitude', 3)
#     f_h.createDimension('longitude', 20)
#     f_h.createDimension('depth', 37)
# # #
# # #
#     f_w.createVariable('latitude', np.float64, ('latitude'))
#     f_w.createVariable('longitude', np.float64, ('longitude'))
#     f_w.createVariable('depth', np.float64, ('depth'))
#     f_r.createVariable('latitude', np.float64, ('latitude'))
#     f_r.createVariable('longitude', np.float64, ('longitude'))
#     f_r.createVariable('depth', np.float64, ('depth'))
#     f_h.createVariable('latitude', np.float64, ('latitude'))
#     f_h.createVariable('longitude', np.float64, ('longitude'))
#     f_h.createVariable('depth', np.float64, ('depth'))
# #
# #
#     predict = model.predict(xtest[-2 - j:-1 - j])
#     predict = predict.reshape(2,3, 10, 37)
#     predict = np.transpose(predict, (0,2,3,1))
#     predT = predict[0]
#     predictT = mmn.inverse_transform(predT)
#     predS = predict[1]
#     predictS = mmn1.inverse_transform(predS)
#     predict1T = predictT[tf.newaxis,:]
#     predict1S = predictS[tf.newaxis,:]
#     predict = np.concatenate((predict1T,predict1S),axis=0)
#     real = ytest1[-2 - j:-1 - j]
#     real = real.reshape(2,3,10,37)
#     real = np.transpose(real, (0, 2,3, 1))
#     realT = real[0]
#     realS = real[1]
#     ttt = np.concatenate((predict,real),axis=1)
#     T = ttt[0]
#     S = ttt[1]
# #
# #
#     f_w.createVariable('T', np.float64, ('longitude','depth', 'latitude'))
#     f_w.createVariable('S', np.float64, ('longitude','depth', 'latitude'))
#     f_r.createVariable('T', np.float64, ('longitude','depth', 'latitude'))
#     f_r.createVariable('S', np.float64, ('longitude','depth', 'latitude'))
#     f_h.createVariable('T', np.float64, ('longitude','depth', 'latitude'))
#     f_h.createVariable('S', np.float64, ('longitude','depth', 'latitude'))
# #
#     f_w.variables['latitude'][:]  = lat
#     f_r.variables['latitude'][:]  = lat
#     f_h.variables['latitude'][:]  = lat
#
#     f_w.variables['depth'][:] = dep
#     f_r.variables['depth'][:] = dep
#     f_h.variables['depth'][:] = dep
# #
#     f_w.variables['longitude'][:]  = lon
#     f_r.variables['longitude'][:]  = lon
#     f_h.variables['longitude'][:]  = lon2
#
#     f_w.variables['T'][:] = predictT
#     f_w.variables['S'][:] = predictS
#     f_r.variables['T'][:] = realT
#     f_r.variables['S'][:] = realS
#     f_h.variables['T'][:]  = T
#     f_h.variables['S'][:]  = S
