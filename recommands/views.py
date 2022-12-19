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
        # predictions = [algo.predict(str(userId), str(id)) for id in undo_places]

        # predictions list 객체는 surprise의 Predictions 객체를 원소로 가지고 있음.

        # 이를 est 값으로 정렬하기 위해서 아래의 sortkey_est 함수를 정의함.
        # sortKey_est 함수는 list 객체의 sort() 함수의 키 값으로 사용되어 정렬 수행.
        def sortKey_est(pred):
            return pred.est

        # sortKey_est() 반환값의 내림 차순으로 정렬 수행하고 top_n개의 최상위값 추출
        predictions.sort(key=sortKey_est, reverse=True)

        top_predictions = predictions[:top_n]

        # top_n으로 추출된 음식점의 정보 추출. 음식점 아이디, 추천 예상 평점, 음식점 이름 추출
        top_places_uids = [pred.uid for pred in top_predictions]
        top_places_ids = [pred.iid for pred in top_predictions]
        top_places_rating = [round(pred.est, 2) for pred in top_predictions]
        top_places = pd.concat([rating_place, seoul_place])
        top_places = top_places.drop_duplicates(['place_id'])
        # top_places = top_places.drop_duplicates(['id'])
        top_places = top_places[top_places.place_id.isin(top_places_ids)]
        top_places['rating'] = top_places_rating
        top_places['pk'] = top_places_uids
        top_places = top_places.rename(columns={'place_id': 'id'})

        top_places = top_places.fillna({'image': '/media/recommands/reastaurant_image.png'})
        print(top_places['image'])

        top_places = top_places[['pk', 'place_name', 'id', 'rating', 'address_name', 'category_group_name', 'phone', 'place_url', 'road_address_name', 'x', 'y', 'image', '지역구']]

        return top_places

# 랜덤 맛집 장소 추천
class random_recomm(GenericAPIView):
    def get(self, request):
        seoul_place = pd.read_sql_table('seoul_place', conn)
        seoul_place = seoul_place[seoul_place['rate'] >= 3]
        rand = random.randint(1, len(seoul_place))
        random_place = seoul_place.iloc[rand, :]
        random_place = random_place.to_json(indent=4, force_ascii=False)

        return HttpResponse(random_place, status=status.HTTP_200_OK)