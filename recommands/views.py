import json
import os
import pickle
import pymysql
import numpy as np
import pymysql.cursors
import sqlalchemy.types
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
import pandas as pd
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine
from surprise import Reader, SVD, AlgoBase
from surprise.dataset import DatasetAutoFolds, Dataset
from surprise.model_selection import cross_validate, GridSearchCV
from .config.db_info import db_info


# # todo : rating 테이블에 대해서 가본 장소 + 좋아요 장소 : 0.8, 가본 장소 : 0.7, 좋아요 장소 : 0.7로 각각에 대해 가중치 값을 평점에 집어 넣고
# # todo : 행렬 분해를 계산
#

# 크롤링한 데이터를 하나의 유저로 생각하고 한 경우

# df = pd.read_csv("/Users/moongi/food_data 복사본 2/중간합계.csv")
# df.drop(columns=['Unnamed: 0'], inplace=True)
#
# # df.loc[df['vote'] == ' ', 'vote'] = '0'
# # df.loc[df['rate'] == ' ', 'rate'] = '0'
# # df['vote'] = df['vote'].str.replace('건', '')
# # df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.2'], inplace=True)
# # print(df.head(3))
# # df.to_csv("/Users/moongi/food_data 복사본 2/중간합계.csv")
#
# db_connection_str = 'mysql+pymysql://root:1234@localhost/FIT_EAT_UP'
# db_connection = create_engine(db_connection_str)
# conn = db_connection.connect()
#
#
# dtypesql = {'name': sqlalchemy.types.VARCHAR(50),
#             'vote': sqlalchemy.types.VARCHAR(50),
#             'rate': sqlalchemy.types.FLOAT,
#             'place_url': sqlalchemy.types.VARCHAR(100),
#             'id': sqlalchemy.types.VARCHAR(100),
#             '지역구': sqlalchemy.types.VARCHAR(50)
#
#
#
#             }
# df.to_sql(name='seoul_place', con=db_connection, if_exists='append', index=False, dtype=dtypesql)
#
# # 데이터 전처리
#
# df = df.rename(columns={'name': 'place_name', 'rate': 'rating', 'id': 'place_id'})  # column명 변경
# print('결측값 처리 전', df.shape)
# # print(df.isnull().sum())  # 결측값 확인
# # df.dropna(axis=0, inplace=True)  # 결측값이 있는 행 제거, 결측값이 있는 행은 음식점으로서 가치가 없을 것이라고 예상
# df = df.fillna({'vote': 0, 'rate': 0})  # 결측값이 있는 행을 vote : 0, rate : 0
# # print(df.isnull().sum())
# df['vote'] = df['vote'].str.replace('건', '')
# df['vote'] = df['vote'].str.replace(',', '')
# df['vote'] = pd.to_numeric(df['vote'], errors='coerce')
# df = df.dropna(subset=['vote'])
# df['vote'] = df['vote'].astype(int)
# df['user_id'] = 5
#
#
# engine = create_engine(db_info, convert_unicode=True)
# conn = engine.connect()
#
# rating = pd.read_sql_table('accounts_userrating', conn)
# place = pd.read_sql_table('accounts_place', conn)
#
# rating_place = pd.merge(rating, place, how='left', left_on='place_id', right_on='id')
# rating_matrix = rating_place.filter(items=['user_id', 'place_id', 'rating', 'place_name'])
#
# # print(rating_matrix)
#
# test_df_matrix = df.filter(items=['user_id', 'place_id', 'rating', 'place_name'])
#
# result1 = pd.concat([test_df_matrix, rating_matrix])
#
# print(result1.shape)

# 크롤링한 데이터를 유저에게 각각 넣어준 경우
class Train(APIView):
    def post(self, request):
        pd.set_option('display.max_columns', 100)
        pd.set_option('display.max_colwidth', 20)

        engine = create_engine(db_info, convert_unicode=True)
        conn = engine.connect()

        df = pd.read_csv("/Users/moongi/food_data 복사본 2/중간합계.csv")
        df.drop(columns=['Unnamed: 0'], inplace=True)

        db_connection_str = 'mysql+pymysql://root:1234@localhost/FIT_EAT_UP'
        db_connection = create_engine(db_connection_str)
        conn = db_connection.connect()

        dtypesql = {'name': sqlalchemy.types.VARCHAR(50),
                    'vote': sqlalchemy.types.VARCHAR(50),
                    'rate': sqlalchemy.types.FLOAT,
                    'place_url': sqlalchemy.types.VARCHAR(100),
                    'id': sqlalchemy.types.VARCHAR(100),
                    '지역구': sqlalchemy.types.VARCHAR(50)
                    }
        df.to_sql(name='seoul_place', con=db_connection, if_exists='append', index=False, dtype=dtypesql)

        # 데이터 전처리

        df = df.rename(columns={'name': 'place_name', 'rate': 'rating', 'id': 'place_id'})  # column명 변경
        print('결측값 처리 전', df.shape)
        # print(df.isnull().sum())  # 결측값 확인
        # df.dropna(axis=0, inplace=True)  # 결측값이 있는 행 제거, 결측값이 있는 행은 음식점으로서 가치가 없을 것이라고 예상
        df = df.fillna({'vote': 0, 'rate': 0})  # 결측값이 있는 행을 vote : 0, rate : 0
        # print(df.isnull().sum())
        df['vote'] = df['vote'].str.replace('건', '')
        df['vote'] = df['vote'].str.replace(',', '')
        df['vote'] = pd.to_numeric(df['vote'], errors='coerce')
        df = df.dropna(subset=['vote'])
        df['vote'] = df['vote'].astype(int)
        df = df.get(df['지역구'] == '노원구')
        df1 = df.copy()
        df['user_id'] = 1
        df1['user_id'] = 2
        test_df_matrix = df.filter(items=['user_id', 'place_id', 'rating', 'place_name'])
        test_df_matrix1 = df1.filter(items=['user_id', 'place_id', 'rating', 'place_name'])

        rating = pd.read_sql_table('accounts_userrating', conn)
        place = pd.read_sql_table('accounts_place', conn)

        rating_place = pd.merge(rating, place, how='left', left_on='place_id', right_on='id')
        rating_matrix = rating_place.filter(items=['user_id', 'place_id', 'rating', 'place_name'])

        # result1 = result1.join(test_df_matrix)
        # result1 = result1.join(test_df_matrix1)

        result1 = pd.concat([rating_matrix, test_df_matrix, test_df_matrix1])

        print(result1.head(3))

        data = request.data.copy()

        if request.data['num'] == 2:
            if not data['user_id1']:
                user1 = 0
            else:
                user1 = data['user_id1']
            if not data['user_id2']:
                user2 = 0
            else:
                user2 = data['user_id2']

            result1 = result1[(result1['user_id'] == user1) | (result1['user_id'] == user2)
                                         | (result1['user_id'] == 5)]

        if request.data['num'] == 3:
            if not data['user_id1']:
                user1 = 0
            else:
                user1 = data['user_id1']
            if not data['user_id2']:
                user2 = 0
            else:
                user2 = data['user_id2']
            if not data['user_id3']:
                user3 = 0
            else:
                user3 = data['user_id3']

            result1 = result1[(result1['user_id'] == user1) | (result1['user_id'] == user2) |
                                          (result1['user_id'] == user3)]

        if request.data['num'] == 4:
            if not data['user_id1']:
                user1 = 0
            else:
                user1 = data['user_id1']
            if not data['user_id2']:
                user2 = 0
            else:
                user2 = data['user_id2']
            if not data['user_id3']:
                user3 = 0
            else:
                user3 = data['user_id3']
            if not data['user_id4']:
                user4 = 0
            else:
                user4 = data['user_id4']

            result1 = result1[(result1['user_id'] == user1) | (result1['user_id'] == user2) |
                                          (result1['user_id'] == user3) | (result1['user_id'] == user4)]

        result1 = result1.pivot_table('rating', index='user_id', columns='place_name')
        result1 = result1.fillna(0)
        print('result1 : ', result1)

        try:
            P, Q = self.matrix_factorization(R=result1.values, K=100, steps=100, learning_rate=0.01,
                                             r_lambda=0.01)
            pred_matrix = np.dot(P, Q.T)
            pred_matrix = pred_matrix.astype(float)
            rating_pred_matrix = pd.DataFrame(data=pred_matrix, index=result1.index,
                                              columns=result1.columns)
            print(rating_pred_matrix)
        except Exception as err:
            return Response(str(err), status=status.HTTP_400_BAD_REQUEST)

        path = os.path.join(settings.MODEL_ROOT, request.data['model_name'])
        with open(path, 'wb') as file:
            pickle.dump(rating_pred_matrix, file)

        # 결과값 출력
        try:
            with open(path, 'rb') as file:
                model = pickle.load(file)

            recomm_index = model.mean().argsort().values[::-1]
            rating_place_name = model.iloc[:, recomm_index].columns

            rating_place_rate = model.mean()[recomm_index].values.round(2)
            all_recomm_df = pd.DataFrame()
            for i in range(len(recomm_index)):
                recomm_df = place[place['place_name'] == rating_place_name[i]]
                all_recomm_df = pd.concat([all_recomm_df, recomm_df])
            all_recomm_df['rating'] = rating_place_rate
            recomm_json = all_recomm_df.to_json(orient='records', indent=4, force_ascii=False)

        except Exception as err:
            return Response(str(err), status=status.HTTP_400_BAD_REQUEST)
        return HttpResponse(recomm_json, status=status.HTTP_200_OK)

    def get_rmse(self, R, P, Q, non_zeros):
        error = 0
        # 두 개의 분해된 행렬 P와 Q.T의 내적으로 예측 R 행렬 생성
        full_pred_matrix = np.dot(P, Q.T)

        # 실제 R 행렬에서 널이 아닌 깂의 위치 인덱스 추출해 실제 R 행렬과 예측 행렬의 RMSE 추출
        x_non_zero_ind = [non_zero[0] for non_zero in non_zeros]
        y_non_zero_ind = [non_zero[1] for non_zero in non_zeros]
        R_non_zeros = R[x_non_zero_ind, y_non_zero_ind]
        full_pred_matrix_non_zeros = full_pred_matrix[x_non_zero_ind, y_non_zero_ind]
        mse = mean_squared_error(R_non_zeros, full_pred_matrix_non_zeros)
        rmse = np.sqrt(mse)

        return rmse

    def matrix_factorization(self, R, K, steps=200, learning_rate=0.01, r_lambda=0.01):
        num_users, num_items = R.shape
        #  P와 Q 매트릭스의 크기를 지정하고 정규 분포를 가진 랜덤한 값으로 입력합니다.
        np.random.seed(1)
        P = np.random.normal(scale=1. / K, size=(num_users, K))
        Q = np.random.normal(scale=1. / K, size=(num_items, K))

        prev_rmse = 10000
        break_count = 0

        # R > 0 인 행 위치, 열 위치, 값을 non_zeros 리스트 객체에 저장.
        non_zeros = [(i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if R[i, j] > 0]

        # SGD기법으로 P와 Q 매트릭스를 계속 업데이트.
        for step in range(steps):
            for i, j, r in non_zeros:
                # 실제 값과 예측 값의 차이인 오류 값 구함
                eij = r - np.dot(P[i, :], Q[j, :].T)
                # Regularization을 반영한 SGD 업데이트 공식 적용
                P[i, :] = P[i, :] + learning_rate * (eij * Q[j, :] - r_lambda * P[i, :])
                Q[j, :] = Q[j, :] + learning_rate * (eij * P[i, :] - r_lambda * Q[j, :])

            rmse = self.get_rmse(R, P, Q, non_zeros)
            if (step % 10) == 0:
                print('### iteration step: ', step, "rmse : ", rmse)

        return P, Q

# 원본
# class Train(APIView):
#     def post(self, request):
#         engine = create_engine(db_info, convert_unicode=True)
#         conn = engine.connect()
#
#         df = pd.read_csv("/Users/moongi/food_data 복사본 2/중간합계.csv")
#         df.drop(columns=['Unnamed: 0'], inplace=True)
#
#         db_connection_str = 'mysql+pymysql://root:1234@localhost/FIT_EAT_UP'
#         db_connection = create_engine(db_connection_str)
#         conn = db_connection.connect()
#
#         dtypesql = {'name': sqlalchemy.types.VARCHAR(50),
#                     'vote': sqlalchemy.types.VARCHAR(50),
#                     'rate': sqlalchemy.types.FLOAT,
#                     'place_url': sqlalchemy.types.VARCHAR(100),
#                     'id': sqlalchemy.types.VARCHAR(100),
#                     '지역구': sqlalchemy.types.VARCHAR(50)
#                     }
#         df.to_sql(name='seoul_place', con=db_connection, if_exists='append', index=False, dtype=dtypesql)
#
#         # 데이터 전처리
#
#         df = df.rename(columns={'name': 'place_name', 'rate': 'rating', 'id': 'place_id'})  # column명 변경
#         print('결측값 처리 전', df.shape)
#         # print(df.isnull().sum())  # 결측값 확인
#         # df.dropna(axis=0, inplace=True)  # 결측값이 있는 행 제거, 결측값이 있는 행은 음식점으로서 가치가 없을 것이라고 예상
#         df = df.fillna({'vote': 0, 'rate': 0})  # 결측값이 있는 행을 vote : 0, rate : 0
#         # print(df.isnull().sum())
#         df['vote'] = df['vote'].str.replace('건', '')
#         df['vote'] = df['vote'].str.replace(',', '')
#         df['vote'] = pd.to_numeric(df['vote'], errors='coerce')
#         df = df.dropna(subset=['vote'])
#         df['vote'] = df['vote'].astype(int)
#         df['user_id'] = 5
#         df = df.get(df['지역구'] == '노원구')
#         print('1112', df.head(3))
#         test_df_matrix = df.filter(items=['user_id', 'place_id', 'rating', 'place_name'])
#
#         rating = pd.read_sql_table('accounts_userrating', conn)
#         place = pd.read_sql_table('accounts_place', conn)
#
#         rating_place = pd.merge(rating, place, how='left', left_on='place_id', right_on='id')
#         rating_matrix = rating_place.filter(items=['user_id', 'place_id', 'rating', 'place_name'])
#
#         result1 = pd.concat([test_df_matrix, rating_matrix])
#
#         data = request.data.copy()
#
#         if request.data['num'] == 2:
#             if not data['user_id1']:
#                 user1 = 0
#             else:
#                 user1 = data['user_id1']
#             if not data['user_id2']:
#                 user2 = 0
#             else:
#                 user2 = data['user_id2']
#
#             result1 = result1[(result1['user_id'] == user1) | (result1['user_id'] == user2)
#                               | (result1['user_id'] == 5)]
#
#         if request.data['num'] == 3:
#             if not data['user_id1']:
#                 user1 = 0
#             else:
#                 user1 = data['user_id1']
#             if not data['user_id2']:
#                 user2 = 0
#             else:
#                 user2 = data['user_id2']
#             if not data['user_id3']:
#                 user3 = 0
#             else:
#                 user3 = data['user_id3']
#
#             result1 = result1[(result1['user_id'] == user1) | (result1['user_id'] == user2) |
#                               (result1['user_id'] == user3)]
#
#         if request.data['num'] == 4:
#             if not data['user_id1']:
#                 user1 = 0
#             else:
#                 user1 = data['user_id1']
#             if not data['user_id2']:
#                 user2 = 0
#             else:
#                 user2 = data['user_id2']
#             if not data['user_id3']:
#                 user3 = 0
#             else:
#                 user3 = data['user_id3']
#             if not data['user_id4']:
#                 user4 = 0
#             else:
#                 user4 = data['user_id4']
#
#             result1 = result1[(result1['user_id'] == user1) | (result1['user_id'] == user2) |
#                               (result1['user_id'] == user3) | (result1['user_id'] == user4)]
#
#         result1 = result1.pivot_table('rating', index='user_id', columns='place_name')
#         result1 = result1.fillna(0)
#         print('result1 : ', result1)
#
#         try:
#             P, Q = self.matrix_factorization(R=result1.values, K=100, steps=100, learning_rate=0.01,
#                                              r_lambda=0.01)
#             pred_matrix = np.dot(P, Q.T)
#             pred_matrix = pred_matrix.astype(float)
#             rating_pred_matrix = pd.DataFrame(data=pred_matrix, index=result1.index,
#                                               columns=result1.columns)
#             print(rating_pred_matrix)
#         except Exception as err:
#             return Response(str(err), status=status.HTTP_400_BAD_REQUEST)
#
#         path = os.path.join(settings.MODEL_ROOT, request.data['model_name'])
#         with open(path, 'wb') as file:
#             pickle.dump(rating_pred_matrix, file)
#
#         # 결과값 출력
#         try:
#             with open(path, 'rb') as file:
#                 model = pickle.load(file)
#
#             recomm_index = model.mean().argsort().values[::-1]
#             rating_place_name = model.iloc[:, recomm_index].columns
#
#             rating_place_rate = model.mean()[recomm_index].values.round(2)
#             all_recomm_df = pd.DataFrame()
#             for i in range(len(recomm_index)):
#                 recomm_df = place[place['place_name'] == rating_place_name[i]]
#                 all_recomm_df = pd.concat([all_recomm_df, recomm_df])
#             all_recomm_df['rating'] = rating_place_rate
#             recomm_json = all_recomm_df.to_json(orient='records', indent=4, force_ascii=False)
#
#         except Exception as err:
#             return Response(str(err), status=status.HTTP_400_BAD_REQUEST)
#         return HttpResponse(recomm_json, status=status.HTTP_200_OK)
#
#     def get_rmse(self, R, P, Q, non_zeros):
#         error = 0
#         # 두 개의 분해된 행렬 P와 Q.T의 내적으로 예측 R 행렬 생성
#         full_pred_matrix = np.dot(P, Q.T)
#
#         # 실제 R 행렬에서 널이 아닌 깂의 위치 인덱스 추출해 실제 R 행렬과 예측 행렬의 RMSE 추출
#         x_non_zero_ind = [non_zero[0] for non_zero in non_zeros]
#         y_non_zero_ind = [non_zero[1] for non_zero in non_zeros]
#         R_non_zeros = R[x_non_zero_ind, y_non_zero_ind]
#         full_pred_matrix_non_zeros = full_pred_matrix[x_non_zero_ind, y_non_zero_ind]
#         mse = mean_squared_error(R_non_zeros, full_pred_matrix_non_zeros)
#         rmse = np.sqrt(mse)
#
#         return rmse
#
#     def matrix_factorization(self, R, K, steps=200, learning_rate=0.01, r_lambda=0.01):
#         num_users, num_items = R.shape
#         #  P와 Q 매트릭스의 크기를 지정하고 정규 분포를 가진 랜덤한 값으로 입력합니다.
#         np.random.seed(1)
#         P = np.random.normal(scale=1. / K, size=(num_users, K))
#         Q = np.random.normal(scale=1. / K, size=(num_items, K))
#
#         prev_rmse = 10000
#         break_count = 0
#
#         # R > 0 인 행 위치, 열 위치, 값을 non_zeros 리스트 객체에 저장.
#         non_zeros = [(i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if R[i, j] > 0]
#
#         # SGD기법으로 P와 Q 매트릭스를 계속 업데이트.
#         for step in range(steps):
#             for i, j, r in non_zeros:
#                 # 실제 값과 예측 값의 차이인 오류 값 구함
#                 eij = r - np.dot(P[i, :], Q[j, :].T)
#                 # Regularization을 반영한 SGD 업데이트 공식 적용
#                 P[i, :] = P[i, :] + learning_rate * (eij * Q[j, :] - r_lambda * P[i, :])
#                 Q[j, :] = Q[j, :] + learning_rate * (eij * P[i, :] - r_lambda * Q[j, :])
#
#             rmse = self.get_rmse(R, P, Q, non_zeros)
#             if (step % 10) == 0:
#                 print('### iteration step: ', step, "rmse : ", rmse)
#
#         return P, Q

#
# class Train(APIView):
#     def post(self, request):
#         engine = create_engine(db_info, convert_unicode=True)
#         conn = engine.connect()
#
#         rating = pd.read_sql_table('accounts_userrating', conn)
#         place = pd.read_sql_table('accounts_place', conn)
#
#         rating_place = pd.merge(rating, place, how='left', left_on='place_id', right_on='id')
#         rating_matrix = rating_place.filter(items=['user_id', 'place_id', 'rating', 'place_name'])
#
#         data = request.data.copy()
#
#         if request.data['num'] == 2:
#             if not data['user_id1']:
#                 user1 = 0
#             else:
#                 user1 = data['user_id1']
#             if not data['user_id2']:
#                 user2 = 0
#             else:
#                 user2 = data['user_id2']
#
#             rating_matrix = rating_matrix[(rating_matrix['user_id'] == user1) | (rating_matrix['user_id'] == user2)]
#
#         if request.data['num'] == 3:
#             if not data['user_id1']:
#                 user1 = 0
#             else:
#                 user1 = data['user_id1']
#             if not data['user_id2']:
#                 user2 = 0
#             else:
#                 user2 = data['user_id2']
#             if not data['user_id3']:
#                 user3 = 0
#             else:
#                 user3 = data['user_id3']
#
#             rating_matrix = rating_matrix[(rating_matrix['user_id'] == user1) | (rating_matrix['user_id'] == user2) |
#                                           (rating_matrix['user_id'] == user3)]
#
#         if request.data['num'] == 4:
#             if not data['user_id1']:
#                 user1 = 0
#             else:
#                 user1 = data['user_id1']
#             if not data['user_id2']:
#                 user2 = 0
#             else:
#                 user2 = data['user_id2']
#             if not data['user_id3']:
#                 user3 = 0
#             else:
#                 user3 = data['user_id3']
#             if not data['user_id4']:
#                 user4 = 0
#             else:
#                 user4 = data['user_id4']
#
#             rating_matrix = rating_matrix[(rating_matrix['user_id'] == user1) | (rating_matrix['user_id'] == user2) |
#                                           (rating_matrix['user_id'] == user3) | (rating_matrix['user_id'] == user4)]
#
#         rating_matrix = rating_matrix.pivot_table('rating', index='user_id', columns='place_name')
#         rating_matrix = rating_matrix.fillna(0)
#         print('rating_matrix : ', rating_matrix)
#
#         try:
#             P, Q = self.matrix_factorization(R=rating_matrix.values, K=100, steps=2800, learning_rate=0.01,
#                                              r_lambda=0.01)
#             pred_matrix = np.dot(P, Q.T)
#             pred_matrix = pred_matrix.astype(float)
#             rating_pred_matrix = pd.DataFrame(data=pred_matrix, index=rating_matrix.index,
#                                               columns=rating_matrix.columns)
#         except Exception as err:
#             return Response(str(err), status=status.HTTP_400_BAD_REQUEST)
#
#         path = os.path.join(settings.MODEL_ROOT, request.data['model_name'])
#         with open(path, 'wb') as file:
#             pickle.dump(rating_pred_matrix, file)
#
#         # 결과값 출력
#         try:
#             with open(path, 'rb') as file:
#                 model = pickle.load(file)
#
#             recomm_index = model.mean().argsort().values[::-1]
#             rating_place_name = model.iloc[:, recomm_index].columns
#
#             rating_place_rate = model.mean()[recomm_index].values.round(2)
#             all_recomm_df = pd.DataFrame()
#             for i in range(len(recomm_index)):
#                 recomm_df = place[place['place_name'] == rating_place_name[i]]
#                 all_recomm_df = pd.concat([all_recomm_df, recomm_df])
#             all_recomm_df['rating'] = rating_place_rate
#             recomm_json = all_recomm_df.to_json(orient='records', indent=4, force_ascii=False)
#
#         except Exception as err:
#             return Response(str(err), status=status.HTTP_400_BAD_REQUEST)
#         return HttpResponse(recomm_json, status=status.HTTP_200_OK)
#
#     def get_rmse(self, R, P, Q, non_zeros):
#         error = 0
#         # 두 개의 분해된 행렬 P와 Q.T의 내적으로 예측 R 행렬 생성
#         full_pred_matrix = np.dot(P, Q.T)
#
#         # 실제 R 행렬에서 널이 아닌 깂의 위치 인덱스 추출해 실제 R 행렬과 예측 행렬의 RMSE 추출
#         x_non_zero_ind = [non_zero[0] for non_zero in non_zeros]
#         y_non_zero_ind = [non_zero[1] for non_zero in non_zeros]
#         R_non_zeros = R[x_non_zero_ind, y_non_zero_ind]
#         full_pred_matrix_non_zeros = full_pred_matrix[x_non_zero_ind, y_non_zero_ind]
#         mse = mean_squared_error(R_non_zeros, full_pred_matrix_non_zeros)
#         rmse = np.sqrt(mse)
#
#         return rmse
#
#     def matrix_factorization(self, R, K, steps=200, learning_rate=0.01, r_lambda=0.01):
#         num_users, num_items = R.shape
#         #  P와 Q 매트릭스의 크기를 지정하고 정규 분포를 가진 랜덤한 값으로 입력합니다.
#         np.random.seed(1)
#         P = np.random.normal(scale=1. / K, size=(num_users, K))
#         Q = np.random.normal(scale=1. / K, size=(num_items, K))
#
#         prev_rmse = 10000
#         break_count = 0
#
#         # R > 0 인 행 위치, 열 위치, 값을 non_zeros 리스트 객체에 저장.
#         non_zeros = [(i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if R[i, j] > 0]
#
#         # SGD기법으로 P와 Q 매트릭스를 계속 업데이트.
#         for step in range(steps):
#             for i, j, r in non_zeros:
#                 # 실제 값과 예측 값의 차이인 오류 값 구함
#                 eij = r - np.dot(P[i, :], Q[j, :].T)
#                 # Regularization을 반영한 SGD 업데이트 공식 적용
#                 P[i, :] = P[i, :] + learning_rate * (eij * Q[j, :] - r_lambda * P[i, :])
#                 Q[j, :] = Q[j, :] + learning_rate * (eij * P[i, :] - r_lambda * Q[j, :])
#
#             rmse = self.get_rmse(R, P, Q, non_zeros)
#             if (step % 10) == 0:
#                 print('### iteration step: ', step, "rmse : ", rmse)
#
#         return P, Q
