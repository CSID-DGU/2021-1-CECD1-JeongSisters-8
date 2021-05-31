'''
test_dataset아래의 사진들을 한 번에 테스트
'''
import os,requests #requests는 웹요청 모듈
os.chdir('./test_dataset')
directory = os.listdir('./')

for file in directory:
    if file.endswith('.jpeg') or file.endswith('.jpg') or file.endswith('.png'):
        params = {
            'image': (file, open(file, 'rb')),
            'output':'result_{}'.format(file)
        }

    response = requests.post('http://localhost:5000/test_files', files=params)
# print(response)