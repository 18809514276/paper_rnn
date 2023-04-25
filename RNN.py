# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.layers.rnn.simple_rnn import SimpleRNN

from tools import getMaxMinData as gmm

plt.rcParams['font.sans-serif']=['SimHei']      # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False        # 用来正常显示负号
from keras.backend import clear_session



#箱型异常值监测
# def outLiersWithBox(disp,time_disp):
#     statistics = disp.describe() #保存基本统计量
#     IQR = statistics.loc['75%']-statistics.loc['25%']   #四分位数间距
#     QL = statistics.loc['25%']  #下四分位数
#     QU = statistics.loc['75%']  #上四分位数
#     threshold1 = QL - 1.5 * IQR #下阈值
#     threshold2 = QU + 1.5 * IQR #上阈值
#     outlier = [] #将异常值保存
#     outlier_x = []
#     for i in range(0, len(disp)):
#         if (disp[i] < threshold1)|(disp[i] > threshold2):
#             outlier.append(disp[i])
#             outlier_x.append(time_disp[i])
#         else:
#             continue
#     return outlier,outlier_x




#读取数据文件

# def readData(load_dict):
#     # print(load_dict)
#     # water = pd.read_csv("waterLevel.csv")
#     # time_water =  water['日期']
#     # water_high = water['测值']
#     # value = pd.read_csv("monitorData.csv",encoding = "GB2312")
#     # print(value)
#     # disp = value['测值']
#     # time_disp = value['日期']
#     #data_y = disp
#     #data_x = time_disp
#     dam_bot = load_dict["damBase"]
#     load_data = load_dict['data']['data']
#     disp_data = load_data[0]['value']
#     water_data = load_data[1]['value']
#     disp = np.array(list(disp_data.values()))
#     time_disp = np.array(list(disp_data.keys()))
#     water_time = np.array(list(water_data.keys()))
#     water_high = np.array(list(water_data.values()))
#     dispvalue   = pd.Series(disp)
#     time_disp12  = pd.Series(time_disp)
#     outlier,outlier_x = outLiersWithBox(dispvalue,time_disp12)
#     #
#     time_disp = np.array(time_disp)
#     dispTimeDel = np.setdiff1d(np.array(time_disp),np.array(outlier_x))    #粗差处理之后的日期
#     for i in range(dispTimeDel.shape[0]):
#         dispTimeDel[i] = np.int(time.mktime(time.strptime(dispTimeDel[i],"%Y-%m-%d")))   #日期转化为时间
#     # print("粗差处理之后的日期",dispTimeDel)
#     #
#     for i in range(time_disp.shape[0]):
#         time_disp[i] = np.int(time.mktime(time.strptime(time_disp[i],"%Y-%m-%d")))   #日期转化为时间
#     # print("转化后的时间",time_disp)
#
#     dispTimeDel =np.array(dispTimeDel)
#     time_disp = np.array(time_disp)
#     # print(dispTimeDel)
#     # print(time_disp)
#     valueDeal = []
#     for i in range(len(time_disp)):
#         index = np.where(dispTimeDel == time_disp[i])
#         # print(index)
#         if len(index[0]) ==0:
#             continue
#         else:
#             valuedata = disp[i]
#             valueDeal.append(valuedata)      #粗差测值
#     # print("粗差处理之后的值",valueDeal)
# #
#     water_time = np.array(water_time)
#     for i in range(len(water_time)):
#         water_time[i]= np.int(time.mktime(time.strptime(water_time[i],"%Y-%m-%d")))
#
#
#     data = []
#     for i in range(dispTimeDel.shape[0]):
#         index = np.where(water_time == dispTimeDel[i])
#         # print(index)
#         if len(index[0])==0:
#             continue
#         else:
#             ond_day_data = [int(dispTimeDel[i]),float((water_high[(index[0][0])])),float(valueDeal[i])]
#             data.append(ond_day_data)
#     data = np.array(data)
#     # print(data)
#     # data2 = data[~np.isnan(data).any(axis=1)]   #剔除空值
#     dam_bot = 41
#
#     return data,dam_bot


#加载数据集
# def loadDataset(load_dict):
#     data,dam_bot=readData(load_dict)
#     thet=data[:,0]
#     input_data=data[:,1:]
#     all_date=[]
#     for i in range(thet.shape[0]):      #转换成localtime
#         date=thet[i]
#         time_local = time.localtime(date)
#         dt = time.strftime("%Y-%m-%d ",time_local)
#         all_date.append(dt)
#     dispt=input_data[:,1]
#     wlevel=input_data[:,0]
#     excel_date=thet/86400
#     theta=excel_date-excel_date[0]+1
#     thetan=(theta-theta[0]+1)/100
#     whead=wlevel-dam_bot
# #    y=dispt-dispt[0]    #y测点监测值(已减去初始值)
#     y=dispt   #y测点监测值(已减去初始值)
#     H1=whead-whead[0]
#     H2=pow(whead,2)-pow(whead[0],2)
#     H3=pow(whead,3)-pow(whead[0],3)
#     T1=np.sin(np.pi*theta/365)-np.sin(np.pi*theta[0]/365)
#     T2=np.cos(np.pi*theta/365)-np.cos(np.pi*theta[0]/365)
#     T3=np.sin(2*np.pi*theta/365)-np.sin(2*np.pi*theta[0]/365)
#     T4=np.cos(2*np.pi*theta/365)-np.sin(2*np.pi*theta[0]/365)
#     D1=thetan-thetan[0]
#     D2=np.log(thetan)-np.log(thetan[0])
#     D3=(thetan-thetan[0])/(thetan-thetan[0]+1)
#     D4=1-np.exp(-thetan+thetan[0])
#     X=np.column_stack((np.double(thet),np.double(H1),np.double(H2),np.double(H3),np.double(T1),np.double(T2),np.double(T3),np.double(T4),np.double(D1),np.double(D2),np.double(D3),np.double(D4)))
#     y =np.array(y)
#     dataSet = np.insert(X, 0, values=y,axis=1)
# #    dataSet  =pd.DataFrame(dataSet)
# #    print(dataSet)
# #    df = dataSet.sample(frac=1).reset_index(drop=True)  # 打乱样本排列，索引重排，有助于降低误差值
# #    data=df.iloc[:,:].values
# #    dataSet2 = np.array(data) #第一列测值、第二列时间、第三列水位
#     return dataSet
def error_calculation(output, data_eigen, test_target):
    y_min = data_eigen[0]
    y_max = data_eigen[1]
    error = (output - test_target) * (y_max - y_min)
    return np.mean(np.abs(error))

def abnormalization(data, data_eigen):
    data = data * (max(data_eigen) - min(data_eigen)) + min(data_eigen)
    return data

def RNN_version(train_data, train_target, test_data,test_target, data_eigen,test_date,train_date):
    # dataSet=loadDataset(load_dict)
    # thet = dataSet[:,1]
    # origalY = dataSet[:,0]
    # dataSet = dataSet.astype('float32')
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # dataSets = scaler.fit_transform(dataSet)     # 归一化
    # splitline = int(len(dataSets)*0.8)    #前80%做拟合
    # predictSample = len(dataSets) -splitline  #剩余20%做预测
    # trainTime = thet[0:splitline]
    trainx= train_data
    trainy = train_target
    # testTime = thet[splitline:]
    testx = test_data
    testy = test_target
    trainx = trainx.reshape((trainx.shape[0], 1, trainx.shape[1]))    # reshape input to be 3D [samples, timesteps, features]
    trainx = trainx.astype('float64')
    trainy = trainy.astype('float64')
    testx = testx.reshape((testx.shape[0], 1, testx.shape[1]))
    testx = testx.astype('float64')
    testy = testy.astype('float64')


    points =20  # 隐含层点数
    clear_session()
    model = Sequential()
    # model.add(SimpleRNN(64))
    model.add(SimpleRNN(points, input_shape=(trainx.shape[1], trainx.shape[2])))   # 搭建神经网络
    #model.add(Dropout(0.5))  # dropout层防止过拟合
    model.add(Dense(1))  
    model.add(Activation('sigmoid'))  #激活层
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(trainx, trainy, epochs=150, batch_size=70,                 # 拟合模型的搭建
                        validation_data=(testx, testy), verbose=2, shuffle=False)

    y_fit = model.predict(trainx)
    yhat =model.predict(testx)     # 预测
    error = error_calculation(yhat, data_eigen, test_target)
    # 平均绝对误差 average_error
    average_error = error
    # 绝对误差absolute_error
    absolute_error = np.power(average_error, 2)

    # trainx = trainx.reshape((trainx.shape[0], trainx.shape[2]))
    # testx = testx.reshape((testx.shape[0], testx.shape[2]))
    # train_date=train_date.tolist()
    # test_date=test_date.tolist()
    # train_target=train_target.tolist()
    # test_target=test_target.tolist()
    date = np.hstack((train_date, test_date))
    target = np.hstack((train_target, test_target))
    target = abnormalization(target, data_eigen)
    fit_target = abnormalization(y_fit, data_eigen)
    fit_target = fit_target.tolist()
    fit_target=[i for item in fit_target for i in item]
    predict_target = abnormalization(yhat, data_eigen)
    # ymax = origalY.max()    # 反归一化
    # ymin = origalY.min()
    # predictY= yhat*(ymax - ymin) + ymin      #预测值
    # monitorY = testy*(ymax - ymin) + ymin        #监测值
    # fitY = y_fit*(ymax - ymin) + ymin   #拟合值
    # trainY = trainy*(ymax - ymin) + ymin   #训练值
    # # 均方误差MSE
    # error = 0.0
    # for i in range(len(predictY)):
    #     er = np.power((predictY[i]-monitorY[i]),2)
    #     error = error + er
    # errorMSE = error/len(predictY)
    # # 均方根误差 RMSE
    # errorRMSE = np.power(errorMSE,0.5)
    # #平均绝对误差 MAE
    # errorsum = 0
    # for i in range(len(predictY)):
    #     ers = np.abs(predictY[i]-monitorY[i])
    #     errorsum  =errorsum +ers
    # errorMAE = errorsum / len(predictY)
    # # print("RNN均方根误差",errorRMSE)
    # # print("RNN平均绝对误差",errorMAE)
    # # return predictY, monitorY,fitY,trainY,trainTime,testTime,history,errorMSE,errorRMSE,errorMAE
    # '''
    # 数据处理
    # '''
    # trainTimeList = []
    # for i in range(len(trainTime)):      #转换成localtime
    #     date1=trainTime[i]
    #     time_local1 = time.localtime(date1)
    #     dt1 = time.strftime("%Y-%m-%d",time_local1)
    #     trainTimeList.append(dt1)
    # testTimeList = []
    # for i in range(len(testTime)):      #转换成localtime
    #     date2=testTime[i]
    #     time_local2 = time.localtime(date2)
    #     dt2 = time.strftime("%Y-%m-%d",time_local2)
    #     testTimeList.append(dt2)
    # thetList = []
    # for i in range(len(thet)):      #转换成localtime
    #     date3=thet[i]
    #     time_local3 = time.localtime(date3)
    #     dt3 = time.strftime("%Y-%m-%d",time_local3)
    #     thetList.append(dt3)


    #添加拟合值
    all_train_date=[]
    all_train_target=[]
    for train in train_date:
        all_train_date.append(train)
    for target_ in fit_target:     #拟合值
        all_train_target.append(target_)
    # print(all_train_target)
    train=list(zip(all_train_date,all_train_target))
    # print(train)
    fit_value = []
    for e in range(len(train)):
        fit_value.append(list(train[e]))
    # print(fit_value)
    # 添加真实值
    all_date=[]
    all_test_target=[]
    for da in date:
        all_date.append(da)
    for test in target:
        all_test_target.append(float(test))
    test_=list(zip(all_date,all_test_target))
    # print(all_date)
    # print("+++++")
    # print(all_test_target)
    test_value=[]
    for t in range(len(test_)):
        test_value.append(list(test_[t]))
    #添加预测值
    all_test_date=[]
    all_predict_traget=[]
    for test_data in test_date:
        all_test_date.append(test_data)
    for predict in predict_target:     #预测值
        all_predict_traget.append(float(predict))
    predict_value=[]
    predict_=list(zip(all_test_date,all_predict_traget))
    for p in range(len(predict_)):
        predict_value.append(list(predict_[p]))


    # 训练、预测误差迭代图
    # loss_train = history.history['loss']  #训练误差
    # loss_test = history.history['val_loss']  #预测误差
    # loss_count = []
    # for i in range(len(loss_test)):
    #     loss_count.append(i)
    #
    # lossTrain = list(zip(loss_count,loss_train))
    # lossTrains =[]
    # for i in range(len(lossTrain)):
    #     lossTrains.append(list(lossTrain[i]))    #训练值的迭代误差
    # lossTest = list(zip(loss_count,loss_test))
    # lossTests =[]
    # for i in range(len(lossTest)):
    #     lossTests.append(list(lossTest[i]))       #测试值的迭代误差
    #
    # errorMAE =float(errorMAE)
    # errorMSE = float(errorMSE)
    # errorRMSE = float(errorRMSE)
    #
    #
    # # return predictY, monitorY,fitY,trainY,trainTime,testTime,errorMSE,errorRMSE,errorMAE
    # return {"monitorValues":test_value,"fitValues":fit_value,"predictValues":predict_value,"lossTrains":lossTrains,"lossTests":lossTests,
    # "errorMSE":errorMSE,"errorRMSE":errorRMSE,"errorMAE":errorMAE}
    data = {"average_error": average_error, "absolute_error": absolute_error, "fit_value": fit_value,
            "test_value": test_value, "predict_value": predict_value}
    maxData = gmm.getMax(data)
    minData = gmm.getMin(data)

    return {"error": round(average_error, 2), "optimal": round(absolute_error, 2), "objectValue": '无',
            "maxData": maxData, "minData": minData, "fit_value": fit_value, "test_value": test_value,
            "predict_value": predict_value}

    # return {"errorMSE":errorMSE,"errorRMSE":errorRMSE,"errorMAE":errorMAE,"fit_value":fit_value,"test_value":test_value,"predict_value":predict_value}


def plot(load_dict):
    train_data, train_target, test_data,test_target, data_eigen,test_date,train_date= load_dict
    return RNN_version(train_data, train_target, test_data,test_target, data_eigen,test_date,train_date)
