import json
import urllib.request

from django.shortcuts import render


file_path = "foods/config_secret/settings_debug.json"

with open(file_path, "r") as f:
    config_secret_debug = json.load(f)


def search(request):
    if request.method == 'GET':
        client_id = config_secret_debug['NAVER']['CLIENT_ID']
        client_secret = config_secret_debug['NAVER']['CLIENT_SECRET']

        q = request.GET.get('q')
        encText = urllib.parse.quote("{}".format(q))
        url = "https://openapi.naver.com/v1/search/local.json?query=" + encText + "&display=5"  # json 결과

        food_api_request = urllib.request.Request(url)
        food_api_request.add_header("X-Naver-Client-Id", client_id)
        food_api_request.add_header("X-Naver-Client-Secret", client_secret)
        response = urllib.request.urlopen(food_api_request)
        rescode = response.getcode()

        if (rescode == 200):
            response_body = response.read()
            result = json.loads(response_body.decode('utf-8'))
            items = result.get('items')
            print(result)  # request를 예쁘게 출력해볼 수 있다.

            context = {
                'items': items
            }
            return render(request, 'search.html', context=context)


def image_search(request):
    if request.method == 'GET':
        client_id = config_secret_debug['NAVER']['CLIENT_ID']
        client_secret = config_secret_debug['NAVER']['CLIENT_SECRET']

        q_img = request.GET.get('q_img')
        encText = urllib.parse.quote("{}".format(q_img))
        image_url = 'https://openapi.naver.com/v1/search/image?query=' + encText  # json image 결과

        food_image_api_request = urllib.request.Request(image_url)
        food_image_api_request.add_header("X-Naver-Client-Id", client_id)
        food_image_api_request.add_header("X-Naver-Client-Secret", client_secret)

        response = urllib.request.urlopen(food_image_api_request)
        rescode = response.getcode()

        if (rescode == 200):
            response_body = response.read()
            result = json.loads(response_body.decode('utf-8'))
            items = result.get('items')
            print(result)  # request를 예쁘게 출력해볼 수 있다.

            context = {
                'items': items
            }
            return render(request, 'search.html', context=context)

