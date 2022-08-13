# import json
# import urllib.request
#
# from django.shortcuts import render
#
#
# def search(request):
#     if request.method == 'GET':
#         config_secret_debug = json.loads(open(CONFIG_SECRET_DEBUG_FILE).read())
#         client_id = config_secret_debug['NAVER']['CLIENT_ID']
#         client_secret = config_secret_debug['NAVER']['CLIENT_SECRET']
#
#         q = request.GET.get('q')
#         encText = urllib.parse.quote("{}".format(q))
#         url = "https://openapi.naver.com/v1/search/movie?query=" + encText  # json 결과
#         movie_api_request = urllib.request.Request(url)
#         movie_api_request.add_header("X-Naver-Client-Id", client_id)
#         movie_api_request.add_header("X-Naver-Client-Secret", client_secret)
#         response = urllib.request.urlopen(movie_api_request)
#         rescode = response.getcode()
#         if (rescode == 200):
#             response_body = response.read()
#             result = json.loads(response_body.decode('utf-8'))
#             items = result.get('items')
#             print(result)  # request를 예쁘게 출력해볼 수 있다.
#
#             context = {
#                 'items': items
#             }
#             return render(request, 'search/search.html', context=context)
