import json
import os
import pickle
import random

import pymysql
import numpy as np
import pymysql.cursors
import sqlalchemy.types
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
import pandas as pd
from rest_framework import status
from rest_framework.generics import GenericAPIView
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

# db_table을 불러오기 위한 코드
engine = create_engine(db_info, convert_unicode=True)
conn = engine.connect()

class surprise_train(APIView):
    def post(self, request):
        # Surprise 패키지를 통한 구현
        pd.set_option('display.max_columns', 100)
        pd.set_option('display.max_colwidth', 20)

        # df = pd.read_csv("/Users/moongi/food_data_최종본/최종합계.csv")
        # df.drop(columns=['Unnamed: 0'], inplace=True)
        #
        #
        #
        # dtypesql = {'place_name': sqlalchemy.types.VARCHAR(50),
        #             'vote': sqlalchemy.types.VARCHAR(50),
        #             'rate': sqlalchemy.types.FLOAT,
        #             'place_url': sqlalchemy.types.VARCHAR(100),
        #             'id': sqlalchemy.types.VARCHAR(100),
        #             'phone': sqlalchemy.types.VARCHAR(18),
        #             'address_name': sqlalchemy.types.VARCHAR(100),
        #             'road_address_name': sqlalchemy.types.VARCHAR(100),
        #             'category_name': sqlalchemy.types.VARCHAR(50),
        #             'x': sqlalchemy.types.VARCHAR(100),
        #             'y': sqlalchemy.types.VARCHAR(100),
        #             'image': sqlalchemy.types.VARCHAR(1000),
        #             '지역구': sqlalchemy.types.VARCHAR(50)
        #             }
        # df.to_sql(name='seoul_place', con=engine, if_exists='append', index=False, dtype=dtypesql)

        rating = pd.read_sql_table('accounts_userrating', conn)
        place = pd.read_sql_table('accounts_place', conn)
        seoul_place = pd.read_sql_table('seoul_place', conn)

        rating_place = pd.merge(rating, place, how='left', left_on='place_id', right_on='id')
        rating_matrix = rating_place.filter(items=['user_id', 'place_id', 'rating', 'place_name'])

        # 추천받고자 하는 친구 수와 해당하는 user_id의 음식점 데이터 가져오기.
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


        rating_matrix['user_id'] = rating_matrix['user_id'].astype(str)

        seoul_place['user_id'] = '999' # 크롤링한 데이터는 user_id를 임의로 지정
        seoul_place = seoul_place.rename(columns={'rate': 'rating', 'id': 'place_id'})  # column명 통일을 위한 변경

        # todo: Front-End와 인자 통일
        seoul_place = seoul_place.get(seoul_place['지역구'] == request.data['area'])  # 추천 받고자 하는 지역 설정
        seoul_place = seoul_place.drop_duplicates(['place_id'])
        seoul_place = seoul_place.fillna({'vote': 0, 'rate': 0})  # 결측값이 있는 행을 vote : 0, rate : 0
        # print(df.isnull().sum())
        seoul_place['vote'] = seoul_place['vote'].str.replace('건', '')
        seoul_place['vote'] = seoul_place['vote'].str.replace(',', '')
        seoul_place['vote'] = pd.to_numeric(seoul_place['vote'], errors='coerce')
        seoul_place = seoul_place.dropna(subset=['vote'])
        seoul_place['vote'] = seoul_place['vote'].astype(int)

        percentile = 0.6
        seoul_place_crawl = seoul_place[seoul_place['vote'] > 0]  # 평점을 남긴 횟수가 있는 음식점에 대해서만 가중치를 부여
        m = seoul_place_crawl['vote'].quantile(percentile)
        C = seoul_place_crawl['rating'].mean()

        def crawling_weighted_vote_average(record):
            v = record['vote']
            R = record['rating']

            return ((v / (v + m)) * R) + ((m / (v + m)) * C)

        seoul_place['rating'] = seoul_place_crawl.apply(crawling_weighted_vote_average, axis=1)
        print(seoul_place[['place_name', 'rating']].sort_values('rating', ascending=False)[:10])

        weighted_seoul_place = seoul_place[['user_id', 'place_id', 'rating', 'place_name']]

        rating_matrix = pd.concat([rating_matrix, weighted_seoul_place])
        rating_matrix['rating'] = rating_matrix['rating'].fillna('0')
        rating_matrix['rating'] = rating_matrix['rating'].astype(float)
        rating_matrix = rating_matrix.drop_duplicates(['place_id'])


        rating_matrix.to_csv('/Users/moongi/FIT_EAT_UP_backend/backend/recommands/models/recommands/' + str(data['user_id1']) + '번 rating_matrix' + '.csv', index=False, header=False)

        reader = Reader(line_format='user item rating', sep=',', rating_scale=(0, 5))
        # DatasetAutoFolds 클래스를 rating_matrix_noh.csv 파일을 기반으로 생성
        data_folds = DatasetAutoFolds('/Users/moongi/FIT_EAT_UP_backend/backend/recommands/models/recommands/' + str(data['user_id1']) + '번 rating_matrix' + '.csv', reader=reader)

        # 전체 데이터를 학습 데이터로 생성함.
        trainset = data_folds.build_full_trainset()

        algo = SVD(n_epochs=20, n_factors=50, random_state=0)
        algo.fit(trainset)

        unseen_places = self.get_undo_surprise(rating_place, rating_matrix, request.data['user_id1'])
        top_places_preds = self.recomm_places_by_surprise(algo, request.data['user_id1'], unseen_places, rating_matrix, rating_place, seoul_place, top_n=10)

        top_places_preds = top_places_preds.to_json(orient='records', indent=4, force_ascii=False)

        return HttpResponse(top_places_preds, status=status.HTTP_200_OK)

    def get_undo_surprise(self, ratings, places, userId):
        # 입력값으로 들어온 userId에 해당하는 사용자가 평점을 매긴 모든 음식점을 리스트로 생성
        do_places = ratings[ratings['user_id'] == userId]['place_id'].tolist()

        # 모든 음식점의 place_id를 리스트로 생성
        total_places = places['place_id'].tolist()

        # 모든 음식점의 place_id중 이미 평점을 매긴 음식점의 place_id를 제외한 후 리스트 생성
        undo_places = [place for place in total_places if place not in do_places]
        print('평점 매긴 음식점 수: ', len(do_places), '추천 대상 음식점 수 : ', len(undo_places),
              '전체 음식점 수 : ', len(total_places))

        return undo_places

    def recomm_places_by_surprise(self, algo, userId, undo_places, rating_matrix, rating_place, seoul_place, top_n=10):

        # 알고리즘 객체의 predict() 메서드를 평점이 없는 영화에 반복 수행한 후 결과를 List 객체로 저장
        predictions = [algo.predict(str(userId), str(place_id)) for place_id in undo_places]

        # predictions list 객체는 surprise의 Predictions 객체를 원소로 가지고 있음.

        # 이를 est 값으로 정렬하기 위해서 아래의 sortkey_est 함수를 정의함.
        # sortKey_est 함수는 list 객체의 sort() 함수의 키 값으로 사용되어 정렬 수행.
        def sortKey_est(pred):
            return pred.est

        # sortKey_est() 반환값의 내림 차순으로 정렬 수행하고 top_n개의 최상위값 추출
        predictions.sort(key=sortKey_est, reverse=True)

        top_predictions = predictions[:top_n]

        # top_n으로 추출된 음식점의 정보 추출. 음식점 아이디, 추천 예상 평점, 음식점 이름 추출
        top_places_ids = [pred.iid for pred in top_predictions]
        top_places_rating = [round(pred.est, 2) for pred in top_predictions]
        top_places = pd.concat([rating_place, seoul_place])
        top_places = top_places.drop_duplicates(['place_id'])
        top_places = top_places[top_places.place_id.isin(top_places_ids)]
        top_places['rating'] = top_places_rating
        top_places = top_places[['place_name', 'place_id', 'rating', 'address_name', 'category_group_name', 'phone', 'place_url', 'road_address_name', 'x', 'y', 'image', '지역구']]

        return top_places

# # todo : rating 테이블에 대해서 가본 장소 + 좋아요 장소 : 0.8, 가본 장소 : 0.7, 좋아요 장소 : 0.7로 각각에 대해 가중치 값을 평점에 집어 넣고
# # todo : 행렬 분해를 계산
#

# 랜덤 맛집 장소 추천
class random_recomm(GenericAPIView):
    def get(self, request):
        seoul_place = pd.read_sql_table('seoul_place', conn)
        seoul_place = seoul_place[seoul_place['rate'] >= 3]
        rand = random.randint(1, len(seoul_place))
        random_place = seoul_place.iloc[rand, :]
        random_place = random_place.to_json(indent=4, force_ascii=False)

        return HttpResponse(random_place, status=status.HTTP_200_OK)




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
# class Train(APIView):
#     def post(self, request):
#         pd.set_option('display.max_columns', 100)
#         pd.set_option('display.max_colwidth', 20)
#
#         engine = create_engine(db_info, convert_unicode=True)
#         conn = engine.connect()
#
#         # df = pd.read_csv("/Users/moongi/food_data 복사본 2/중간합계.csv")
#         # df.drop(columns=['Unnamed: 0'], inplace=True)
#         #

#         #
#         # dtypesql = {'name': sqlalchemy.types.VARCHAR(50),
#         #             'vote': sqlalchemy.types.VARCHAR(50),
#         #             'rate': sqlalchemy.types.FLOAT,
#         #             'place_url': sqlalchemy.types.VARCHAR(100),
#         #             'id': sqlalchemy.types.VARCHAR(100),
#         #             '지역구': sqlalchemy.types.VARCHAR(50)
#         #             }
#         # df.to_sql(name='seoul_place', con=db_connection, if_exists='append', index=False, dtype=dtypesql)
#         df = pd.read_sql_table('seoul_place', conn)
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
#         # 크롤링한 데이터에 대해서 vote 횟수에 따른 가중 평점을 부여
#         # Weighted Rating = (v/(v+m)) * R + (m/(v+m)) * C
#         # v: 개별 영화에 평점을 투표한 횟수, m: 평점을 부여하기 위한 최소 투표 횟수, R: 개별 영화에 대한 평균 평점, C: 전체 영화에 대한 평균 평점
#         # 우리에게 vote: v, rating: C, m값을 더 높이면 투표 횟수가 더 많은 음식점에 대하여 가중치를 더 부여한다.
#         df = df.get(df['지역구'] == '노원구')
#         print('크롤링한 노원역 음식점 갯수', df.shape)
#         df = df.drop_duplicates(['place_id'])
#         print('중복된 데이터 제거 후', df.shape)
#
#         # ----------------
#         df = df[df['vote'] > 0]
#         C = df['rating'].mean()
#         m = df['vote'].quantile(0.6)  # 전체 투표 횟수에서 상위 60%에 해당하는 횟수를 기준으로
#         print('노원구 평점을 매긴 음식점의 수', df[df['vote'] > 0].count())  # 2103개
#         print('노원구 vote macx', df['vote'].max())
#         print('노원구 vote describe', df['vote'].describe())
#         print('C : ', round(C, 3), 'm : ', round(m, 3))
#
#         percentile = 0.6
#         df_crawl = df[df['vote'] > 0]  # 평점을 남긴 횟수가 있는 음식점에 대해서만 가중치를 부여
#         m = df_crawl['vote'].quantile(percentile)
#         C = df_crawl['rating'].mean()
#
#         def crawling_weighted_vote_average(record):
#             v = record['vote']
#             R = record['rating']
#
#             return ((v/(v+m)) * R) + ((m/(v+m)) * C)
#
#         df['weighted_rating'] = df_crawl.apply(crawling_weighted_vote_average, axis=1)
#
#         print(df[['place_name', 'rating', 'vote', 'weighted_rating']].sort_values('weighted_rating', ascending=False)[:10])
#
#         df1 = df.copy()
#         df['user_id'] = request.data['user_id1']
#         df1['user_id'] = request.data['user_id2']
#         test_df_matrix = df.filter(items=['user_id', 'place_id', 'rating', 'place_name'])
#         test_df_matrix1 = df1.filter(items=['user_id', 'place_id', 'rating', 'place_name'])
#
#         rating = pd.read_sql_table('accounts_userrating', conn)
#         place = pd.read_sql_table('accounts_place', conn)
#
#         rating_place = pd.merge(rating, place, how='left', left_on='place_id', right_on='id')
#         rating_matrix = rating_place.filter(items=['user_id', 'place_id', 'rating', 'place_name'])
#
#         # result1 = result1.join(test_df_matrix)
#         # result1 = result1.join(test_df_matrix1)
#
#         result1 = pd.concat([rating_matrix, test_df_matrix, test_df_matrix1])
#
#         print(result1.head(3))
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
#             result1 = result1[(result1['user_id'] == user1) | (result1['user_id'] == user2)]
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
#                                           (result1['user_id'] == user3)]
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
#                                           (result1['user_id'] == user3) | (result1['user_id'] == user4)]
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

# 원본
# class Train(APIView):
#     def post(self, request):
#         engine = create_engine(db_info, convert_unicode=True)
#         conn = engine.connect()
#
#         df = pd.read_csv("/Users/moongi/food_data 복사본 2/합계.csv")
#         df.drop(columns=['Unnamed: 0'], inplace=True)
#

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
