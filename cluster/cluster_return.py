import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import os

data = pd.read_csv(os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + "/database/dataDB.csv",
                   encoding="cp949")
unused_vars = ['id_scholarship', 'scholarship_name', 'link', 'other', 'major', 'region', 'date_start', 'date_end',
               'feature', 'feature_specified',
               'age_max', 'grade_max', 'last_grade_max', 'recommendation', 'year']
data_input = data.drop(unused_vars, axis=1)
scaler = MinMaxScaler()


def DBSCAN_1(Y):
    data_x = data_input[['grade_min']]
    data_x = data_x.append({'grade_min': Y}, ignore_index=True)  # 학생 데이터 추정값 구하기위해 append 해줌
    data_x = pd.DataFrame(scaler.fit_transform(data_x), columns=data_x.columns)  # 스케일링
    Y = np.array(data_x.tail(1))  # 학생데이터는 따로 떼어줌
    data_x.drop(data_x.tail(1).index, inplace=True)
    X = np.array(data_x)
    db_ = DBSCAN(eps=0.03, min_samples=20).fit(X)  # DBSCAN 해줌

    return db_.fit_predict(Y)[0]  # 학생의 클러스터 추정값 리턴해줌


def DBSCAN_2(Y):
    data_x = data_input[['last_grade_min']]
    data_x = data_x.append({'last_grade_min': Y}, ignore_index=True)  # 학생 데이터 추정값 구하기위해 append 해줌
    data_x = pd.DataFrame(scaler.fit_transform(data_x), columns=data_x.columns)  # 스케일링
    Y = np.array(data_x.tail(1))  # 학생데이터는 따로 떼어줌
    data_x.drop(data_x.tail(1).index, inplace=True)
    X = np.array(data_x)
    db_ = DBSCAN(eps=0.03, min_samples=20).fit(X)  # DBSCAN 해줌

    return db_.fit_predict(Y)[0]  # 학생의 클러스터 추정값 리턴해줌


def DBSCAN_3(Y):
    data_x = data_input[['income_max']]
    data_x = data_x.append({'income_max': Y}, ignore_index=True)  # 학생 데이터 추정값 구하기위해 append 해줌
    data_x = pd.DataFrame(scaler.fit_transform(data_x), columns=data_x.columns)  # 스케일링
    Y = np.array(data_x.tail(1))  # 학생데이터는 따로 떼어줌
    data_x.drop(data_x.tail(1).index, inplace=True)
    X = np.array(data_x)
    db_ = DBSCAN(eps=0.1, min_samples=20).fit(X)  # DBSCAN 해줌

    return db_.fit_predict(Y)[0]  # 학생의 클러스터 추정값 리턴해줌


def DBSCAN_4(Y):
    data_x = data_input[['last_grade_min', 'grade_min']]
    data_x = data_x.append({'last_grade_min': Y[0], 'grade_min': Y[1]}, ignore_index=True)  # 학생 데이터 추정값 구하기위해 append 해줌
    data_x = pd.DataFrame(scaler.fit_transform(data_x), columns=data_x.columns)  # 스케일링
    Y = np.array(data_x.tail(1))  # 학생데이터는 따로 떼어줌
    data_x.drop(data_x.tail(1).index, inplace=True)
    X = np.array(data_x)
    db_ = DBSCAN(eps=0.05, min_samples=20).fit(X)  # DBSCAN 해줌

    return db_.fit_predict(Y)[0]  # 학생의 클러스터 추정값 리턴해줌


