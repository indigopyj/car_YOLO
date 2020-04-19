
#사용방법


###1. 폴더내에 darkflow 설치
[darkflow] [https://github.com/thtrieu/darkflow] github에서 git clone 후 readme에 나온 설명대로 설치를 한다.


###2. ckpt폴더에 weight 파일 다운받기
google drive ckpt라는 폴더에 있는 파일들 모두 다운받아서 'ckpt' 폴더에 모두 넣어준다


###3. cfg 파일 넣어주기
cfg 폴더에 'yolo-voc-3c-aug.cfg' 넣어주기


###4. yolo 결과 돌리기
```
python3 yolo.py imagename.jpg 
```
이미지의 yolo 결과가 photo/yolo/before과 photo/yolo/after 폴더에 저장된다.


###5. 전후결과 비교하기
```
python3 compare.py imagename.jpg 
```
이미지해싱에 의한 비교결과가 photo/template/before과 photo/template/after 폴더에 저장된다.
