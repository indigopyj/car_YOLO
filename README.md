
# 사용방법


### 1. 폴더내에 darkflow 설치


[darkflow](https://github.com/thtrieu/darkflow) github에서 git clone 후 readme에 나온 설명대로 설치를 한다.


추가) car_YOLO내의 모든 파일을 darkflow 폴더 내로 넣어준다. 이때 중복된 파일은 덮어쓰기를 한다.


### 2. ckpt폴더에 weight 파일 다운받기


google drive ckpt라는 폴더에 있는 파일들 모두 다운받아서 'ckpt' 폴더에 모두 넣어준다


### 3. cfg 파일 넣어주기


cfg 폴더에 'yolo-voc-3c-aug.cfg' 넣어주기


### 4. yolo 결과 돌리기


```
python3 yolo.py before imagename.jpg 

python3 yolo.py after imagename.jpg

```

이미지의 yolo 결과 json 파일이 photo/yolo/before과 photo/yolo/after 폴더에 저장된다.

photo/yolo/before/imagename.json 	: before image의 yolo result에 대한 json file

photo/yolo/after/imagename.json 	: after image의 yolo result에 대한 json file


#### ex) json file 예시

```
{
	"predictions": [
		{
			"label": "scratch",
			"topx": 289,
			"topy": 181,
			"btmx": 366,
			"btmy": 236
		},
		{
			"label": "scratch",
			"topx": 520,
			"topy": 237,
			"btmx": 538,
			"btmy": 247
		},
		{
			"label": "scratch",
			"topx": 352,
			"topy": 266,
			"btmx": 430,
			"btmy": 325
		}
	]
}
```


### 5. 전후결과 비교하기


```
python3 compare.py imagename.jpg 
```

이미지해싱에 의한 비교결과가 photo/template 폴더에 저장된다.

photo/template/imagename.json 	: 이미지 해싱 결과 새로 발생한 결함에 대한 json file


#### ex) json file 예시
```
{
	"new_defects": [
		{
			"label": "dent",
			"topx": 170,
			"topy": 110,
			"btmx": 234,
			"btmy": 155
		},
		{
			"label": "scratch",
			"topx": 251,
			"topy": 185,
			"btmx": 276,
			"btmy": 198
		}
	]
}
```
