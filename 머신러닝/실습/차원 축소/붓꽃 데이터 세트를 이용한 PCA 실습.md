## 붓꽃 데이터 세트를 이용한 PCA 실습

붓꽃 데이터 세트는 sepal length, sepal width,petal length,petal width의 4개의 속성으로 되어있는데, 이 4개의 속성을 2개의 PCA차원으로 압축해 원래 데이터 세트와 압축된 데이터 세트가 어떻게 달라졌는지 확인해 보자.

먼저 사이킷런의 붓꽃 데이터 세트를 로딩해 보자.

```python
**from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

iris=load_iris()
#넘파이 데이터 세트를 판다스 DataFrame으로 변환
columns=['sepal_length','sepal_width','petal_length','petal_width']
irisDF=pd.DataFrame(iris.data,columns=columns)
irisDF['target']=iris.target
irisDF.head(3)**
```

[output]

![결과1](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/85726d0e-2c51-4f86-bb22-7fe766edf170/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210628%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210628T144422Z&X-Amz-Expires=86400&X-Amz-Signature=e2ab3394b2525404fe06cd515c5a78e4921e79303aad400d45d3f1b0848ab2d6&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

각 품종에 따라 원본 붓꽃 데이터 세트가 어떻게 분포돼 있는지 2차원으로 시각화해 보자. 2차원으로 표현하므로 두 개의 속성인 sepal length와 sepal width를 X축, Y축으로 해 품종 데이터 분포를 나타낸다.

```python
makers=['^','s','o']

#setosa의 target값은 0, versicolor는 1, virginica는 2.각 target별로 다른 모양으로 산점도 표시
for i, maker in enumerate(makers):
  x_axis_data=irisDF[irisDF['target']==i]['sepal_length']
  y_axis_data=irisDF[irisDF['target']==i]['sepal_width']
  plt.scatter(x_axis_data,y_axis_data,marker=maker,label=iris.target_names[i])

plt.legend()
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.show()
```

[output]

![결과2](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/3cbdd01a-50ec-4132-b1b3-988d1ce533f2/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210628%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210628T144450Z&X-Amz-Expires=86400&X-Amz-Signature=c76ca271de1721a13c75b09d311f0104676f67f246e84680f4430eafcef5a0a2&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

setosa 품종의 경우 sepal width가 3.0보다 크고, sepal length가 6.0이하인 곳에 일정하게 분포돼있다. Versicolor와 virginica의 경우는 sepal length와 sepal width 조건만으로는 분류가 어려운 복잡한 조건임을 알 수 있다. 

이제 PCA로 4개의 속성을 2개로 압축한 뒤 앞의 예제와 비슷하게 2개의 PCA속성으로 붓꽃 데이터의 품종 분포를 2차원으로 시각화해 보자.

PCA는 여러 속성의 값을 연산해야 하므로 속성의 스케일에 영향을 받는다. 따라서 여러 속성을 PCA로 압축하기 전에 각 속성값을 동일한 스테일로 변환하는 것이 필요하다. StandarrdScaler를 이용해 평균이 0, 분산이 1인 표준 정규분포로 iris 데이터 세트의 속성값들을 변환하자.

```python
from sklearn.preprocessing import StandardScaler

#Target 값을 제외한 모든 속성값을 StandardScaler를 이용해 표준 정규 분포를 가지는 값들로 변환
iris_scaled=StandardScaler().fit_transform(irisDF.iloc[:,:-1])
```

이제 스케일링이 적용된 데이터 세트에 PCA를 적용해 4차원의 붓꽃 데이터를 2차원 PCA 데이터로 변환해 보자.

```python
from sklearn.decomposition import PCA

pca=PCA(n_components=2)

#fit()과 transform()을 호출해 PCA 변환 데이터 반환
pca.fit(iris_scaled)
iris_pca=pca.transform(iris_scaled)
print(iris_pca.shape)
```

[output]

(150, 2)

PCA 객체의 transform() 메서드를 호출해 원본 데이터 세트를 (150,2)의 데이터 세트로 iris_pca객체 변수로 반환하였다. iris_pca는 변환된 PCA데이터 세트를 150X2 넘파이 행렬로 가지고 있다. 이를 DataFrame으로 변환한 뒤 데이터값을 확인해 보자.

```python
#PCA 변환된 데이터의 칼럼 명을 각각 pca_component_1, pca_component_2로 명명
pca_columns=['pca_component_1','pca_component_2']
irisDF_pca=pd.DataFrame(iris_pca,columns=pca_columns)
irisDF_pca['target']=iris.target
irisDF_pca.head(3)
```

[output]

![결과3](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/f774d2d6-7a5a-4d50-b822-6b8aaa96be5a/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210628%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210628T144532Z&X-Amz-Expires=86400&X-Amz-Signature=d531777d8837aaf5222dc75c47c78a01078f06d7f00d45e115ccbe8968d69da7&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

이제 2개의 속성으로 PCA변환된 데이터 세트를 2차원상에서 시각화해 보자.

```python
makers=['^','s','o']

#pca_component_1을 X축, pca_component_2를 y축으로 scatter plot수행
for i, maker in enumerate(makers):
  x_axis_data=irisDF_pca[irisDF['target']==i]['pca_component_1']
  y_axis_data=irisDF_pca[irisDF['target']==i]['pca_component_2']
  plt.scatter(x_axis_data,y_axis_data,marker=maker,label=iris.target_names[i])

plt.legend()
plt.xlabel('pca_component_1')
plt.ylabel('pca_component_2')
plt.show()
```

[output]

![결과4](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/135a3c41-e4b3-4520-a44d-f35ceffb7404/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210628%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210628T144553Z&X-Amz-Expires=86400&X-Amz-Signature=d4543b23c8181caedcbffd9f60aa2b2782074bf2411499b106c410e790e47e23&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

PCA로 변환한 후에도 pca_component_1 축을 기반으로 setosa의 품종은 명확히 구분 가능하다. Versicolor와 Virginica는 pca_component_1축을 기반으로 서로 겹치는 부분이 일부 존재하지만, 비교적 잘 구분되었다.

PCA Component별로 원본 데이터의 변동성을 얼마나 반영하고 있는지 알아보자. PCA변환을 수행한 PCA객체의 explained_variance_ratio_속성은 변동성에서 개별 PCA컴포넌트별로 차지하는 변동성 비율을 제공하고 있다.

```python
print(pca.explained_variance_ratio_)
```

[output]

[0.72962445 0.22850762]

첫 번째 PCA 변환 요소인 pca_component_1이 전체 변동성이 약 72.9%를 차지하며, 두 번째인 pca_component_2가 약 22.8%를 차지한다. 따라서 두개의 요소로만 변환해도 원본 데이터의 변동성을 95% 설명할 수 있다.

이번에는 원본 붓꽃 데이터 세트와 PCA로 변환된 데이터 세트에 각각 분류를 적용한 후 결과를 비교하자.

먼저 원본 붓꽃 데이터에 앤덤 포레스트를 적용한 결과는 다음과 같다.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

rcf=RandomForestClassifier(random_state=0)
scores=cross_val_score(rcf,iris.data,iris.target,scoring='accuracy',cv=3)
print('원본 데이터 교차 검증 개별 정확도:',scores)
print('원본 데이터 평균 정확도:',np.mean(scores))
```

[output]

원본 데이터 교차 검증 개별 정확도: [0.98 0.94 0.98]
원본 데이터 평균 정확도: 0.9666666666666667

이번에는 기존 4차원 데이터를 2차원으로 PCA변환한 데이터 세트에 랜덤 포레스트를 적용해 보자.

```python
pca_X=irisDF_pca[['pca_component_1','pca_component_2']]
scores_pca=cross_val_score(rcf,pca_X,iris.target,scoring='accuracy',cv=3)
print('PCA 변환 데이터 교차 검증 개별 정확도:',scores_pca)
print('PCA 변환 데이터 평균 정확도:',np.mean(scores_pca))
```

[output]

PCA 변환 데이터 교차 검증 개별 정확도: [0.88 0.88 0.9 ]
PCA 변환 데이터 평균 정확도: 0.8866666666666667

원본 데이터 세트 대비 예측 정확도는 PCA 변환 차원 개수에 따라 예측 성능이 떨어질 수밖에 없다. 위 붓꽃 데이터의 경우 4개의 속성이 2개의 변환 속성으로 감소하면서 예측 성능의 정확도가 원본 데이터 대비 10% 하락했다. 10%의 정확도 하락은 비교적 큰 성능 수치의 감소이지만 4개의 속성이 2개로, 속성 개수가 50% 감소한 것을 고려한다면 PCA변환 후에도 원본 데이터의 특성을 상당 부분 유지하고 있음을 알 수 있다.
