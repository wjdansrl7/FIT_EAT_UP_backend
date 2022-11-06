import os
import pickle

import numpy as np
from django.conf import settings
from django.http import HttpResponse
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


def index(request):
    return HttpResponse('안녕하세요')


engine = create_engine(db_info, convert_unicode=True)
conn = engine.connect()
# # 사용자-음식점 평점 행렬로 변환
# rating_matrix = rating.pivot_table('rating', index='user_id', columns='place_id')

rating = pd.read_sql_table('accounts_userrating', conn)
place = pd.read_sql_table('accounts_place', conn)
# likePlace = pd.read_sql_table('accounts_user_like_places', conn)
# visitPlace = pd.read_sql_table('accounts_user_visit_places', conn)
#
# # todo : rating 테이블에 대해서 가본 장소 + 좋아요 장소 : 0.8, 가본 장소 : 0.7, 좋아요 장소 : 0.7로 각각에 대해 가중치 값을 평점에 집어 넣고
# # todo : 행렬 분해를 계산
#
# # for i in list(likePlace[likePlace['user_id'] == 1]['place_id']):
# #     print(list(rating[[rating['user_id'] == 1][rating['place_id'] == i]]['rating']))
# # print(list(rating[rating['user_id'] == 1]['place_id']))
#
rating_place = pd.merge(rating, place, how='left', left_on='place_id', right_on='id')
rating_place = rating_place[rating_place.user_id.isin([1, 2])]
# rating_place = rating_place.filter(items=['user_id', 'place_id', 'rating'])
rating_place = rating_place.pivot_table('rating', index='user_id', columns='place_id')
rating_place = rating_place.fillna(0)
rating_place.to_csv('recommands/models/rating_place.csv')
rating_place = pd.read_csv('recommands/models/rating_place.csv')
# rating_place = rating_place.values
rating_place = rating_place.drop('user_id', axis=1)
print(rating_place)

# num_components = 3
# U, Sigma, Vt = svd(rating_place, full_matrices=False)
# U, Sigma, Vt = svds(rating_place, k=num_components)
# print(U.shape, Sigma.shape, Vt.shape)

# print('Sigma 행렬: ', np.round(Sigma,3))
# print(U)
# print(Vt)

# Sigma_mat = np.diag(Sigma)
# print('Sigma_mat 행렬 : ', Sigma_mat)
# print(Sigma_mat.shape)
# a2 = np.dot(U, Sigma_mat)
# print('a2', a2)
# a1 = np.dot(np.dot(U, Sigma_mat), Vt)
# # print(np.round(a1, 3))
# print(a1)
#
# rating_matrix = rating_place.filter(items=['user_id', 'place_id', 'rating', 'place_name'])
# rating_matrix.to_csv('recommands/models/rating_matrix.csv', index=False, header=False)
#
# rating_matrix = rating_matrix[rating_matrix.user_id.isin([1, 2])]
#
# rating_matrix = rating_matrix.pivot_table('rating', index='user_id', columns='place_id')
# rating_matrix = rating_matrix.fillna(0)
# print(rating_matrix)


class Train(APIView):
    def post(self, request):
        engine = create_engine(db_info, convert_unicode=True)
        conn = engine.connect()

        rating = pd.read_sql_table('accounts_userrating', conn)
        place = pd.read_sql_table('accounts_place', conn)

        rating_place = pd.merge(rating, place, how='left', left_on='place_id', right_on='id')
        rating_matrix = rating_place.filter(items=['user_id', 'place_id', 'rating', 'place_name'])

        # data = request.data.copy()
        # li = [data['user_id1'], data['user_id2'], data['user_id3'], data['user_id4']]
        # print(li)
        # rating_matrix = rating_place[rating_place.user_id.isin([request.data['user_id1'], request.data['user_id2'],
        #                                                         request.data['user_id3'], request.data['user_id4']])]
        # rating_matrix = rating_place[rating_place['user_id'] == [request.data['user_id1'], request.data['user_id2'],
        #                                                          request.data['user_id3'], request.data['user_id4']]]
        # rating_matrix = rating_place[rating_place.user_id.isin(li)]

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

            rating_matrix = rating_matrix[(rating_matrix['user_id'] == user1) | (rating_matrix['user_id'] == user2)]

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

            rating_matrix = rating_matrix[(rating_matrix['user_id'] == user1) | (rating_matrix['user_id'] == user2) |
                                         (rating_matrix['user_id'] == user3)]

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

            rating_matrix = rating_matrix[(rating_matrix['user_id'] == user1) | (rating_matrix['user_id'] == user2) |
                                         (rating_matrix['user_id'] == user3) | (rating_matrix['user_id'] == user4)]

        # rating_matrix = rating_place[rating_place['user_id'] == 1]
        rating_matrix = rating_matrix.pivot_table('rating', index='user_id', columns='place_name')
        rating_matrix = rating_matrix.fillna(0)
        print(rating_matrix.values)

        # model_name = request.data.pop('model_name')
        # print(model_name)


        try:
            P, Q = self.matrix_factorization(R=rating_matrix.values, K=100, steps=2800, learning_rate=0.01, r_lambda=0.01)
            pred_matrix = np.dot(P, Q.T)
            print(pred_matrix)
        except Exception as err:
            return Response(str(err), status=status.HTTP_400_BAD_REQUEST)

        path = os.path.join(settings.MODEL_ROOT, request.data['model_name'])
        with open(path, 'wb') as file:
            pickle.dump(pred_matrix, file)

        # 결과값 출력
        try:
            with open(path, 'rb') as file:
                model = pickle.load(file)
                print(model)
        except Exception as err:
            return Response(str(err), status=status.HTTP_400_BAD_REQUEST)
        return Response(status=status.HTTP_200_OK)


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
        P = np.random.normal(scale=1./K, size=(num_users, K))
        Q = np.random.normal(scale=1./K, size=(num_items, K))

        prev_rmse = 10000
        break_count = 0

        # R > 0 인 행 위치, 열 위치, 값을 non_zeros 리스트 객체에 저장.
        non_zeros = [ (i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if R[i, j] > 0]

        # SGD기법으로 P와 Q 매트릭스를 계속 업데이트.
        for step in range(steps):
            for i, j, r in non_zeros:
                # 실제 값과 예측 값의 차이인 오류 값 구함
                eij = r - np.dot(P[i, :], Q[j, :].T)
                # Regularization을 반영한 SGD 업데이트 공식 적용
                P[i, :] = P[i, :] + learning_rate * (eij * Q[j, :] - r_lambda*P[i, :])
                Q[j, :] = Q[j, :] + learning_rate * (eij * P[i, :] - r_lambda*Q[j, :])

            rmse = self.get_rmse(R, P, Q, non_zeros)
            if (step % 10) == 0:
                print('### iteration step: ', step, "rmse : ", rmse)

        return P, Q

# ---
# rating_matrix = pd.read_csv('recommands/models/rating_matrix.csv')
# reader = Reader(line_format='user item rating', sep=',',
#                 rating_scale=(0, 5))
#
# # data = Dataset.load_from_file(rating_matrix[['user_id', 'place_id', 'rating']], reader)
# data_folds = DatasetAutoFolds(ratings_file='recommands/models/rating_matrix.csv', reader=reader)
#
# trainset = data_folds.build_full_trainset()
#
#
#
# param_grid = {'n_epochs': [20,40,60], 'n_factors': [50,100,200]}
#
# gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
# gs.fit(data_folds)
#
# print(gs.best_params['rmse'])
# algo = SVD(n_epochs=20, n_factors=50, random_state=0)
# algo1 = SVD(n_epochs=20, n_factors=200, random_state=0)
#
# # cross_validate(algo, data_folds, measures=['RMSE', 'MAE'], cv=5, verbose=True)
# algo.fit(trainset)
#
#
# uid = str(2)
# iid = str(93948546)
#
# pred = algo.predict(uid, iid, verbose=True)
#
# algo1.fit(trainset)
#
# uid = str(2)
# pred1 = algo1.predict(uid, iid, verbose=True)
#
# ---


# def get_uncheck_surprise(self, ratings, places, user_id):
#     check_place = ratings[ratings['user_id'] == user_id]['place_id'].tolist()
#
#     total_place = places['place_id'].tolist()
#
#     uncheck_place = [place for place in total_place if place not in check_place]
#
#     return uncheck_place


# class Train(APIView):
#     def post(self, request):
#         iris = datasets.load_iris()
#         print(type(iris))
#         mapping = dict(zip(np.unique(iris.target), iris.target_names))
#
#         X = pd.DataFrame(iris.data, columns=iris.feature_names)
#         y = pd.DataFrame(iris.target).replace(mapping)
#         model_name = request.data.pop('model_name')
#
#         try:
#             clf = RandomForestClassifier(**request.data)
#             clf.fit(X, y)
#         except Exception as err:
#             return Response(str(err), status=status.HTTP_400_BAD_REQUEST)
#
#         path = os.path.join(settings.MODEL_ROOT, model_name)
#         with open(path, 'wb') as file:
#             pickle.dump(clf, file)
#         return Response(status=status.HTTP_200_OK)


# class Predict(APIView):
#     def post(self, request):
#         predictions = []
#         for entry in request.data:
#             model_name = entry.pop('model_name')
#             path = os.path.join(settings.MODEL_ROOT, model_name)
#             with open(path, 'rb') as file:
#                 model = pickle.load(file)
#             try:
#                 result = model.predict(pd.DataFrame([entry]))
#                 predictions.append(result[0])
#             except Exception as err:
#                 return Response(str(err), status=status.HTTP_400_BAD_REQUEST)
#         return Response(predictions, status=status.HTTP_200_OK)


# ---------------------------------------------------------
# class MyOwnAlgorithm(AlgoBase):
#     def __init__(self):
#         AlgoBase.__init__(self)
#
#     def estimate(self, u, i):
#         pass
#
#     def fit(self, trainset):
#
#         # Here again: call base method before doing anything.
#         AlgoBase.fit(self, trainset)
#
#         # Compute baselines and similarities
#         # self.bu, self.bi = self.compute_baselines()
#         # self.sim = self.compute_similarities()
#
#         return self
#
#
# ppp = MyOwnAlgorithm()
# ppp.fit(trainset)
# ppp.predict(uid=2, iid=93948546, verbose=True)




