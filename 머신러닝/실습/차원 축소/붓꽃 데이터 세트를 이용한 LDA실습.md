## 붓꽃 데이터 세트를 이용한 LDA실습

붓꽃 데이터 세트를 사이킷런의 LDA를 이용해 변환하고, 그 결과를 품종별로 시각해 보자.

붓꽃 데이터 세트를 로드하고 표준 정규 분포로 스케일링 하자.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

iris=load_iris()
iris_scaled=StandardScaler().fit_transform(iris.data)
```

2개의 컨포넌트로 붓꽃 데이터를 LDA변환하자. PCA와 다르게 LDA에서 한 가지 유의해야 할 점은 LDA는 실제로는 PCA와 다르게 비지도학습이 아닌 지도학습 이라는 것이다. 즉, 클래스 결정값이 변환시에 필요하다.

```python
lad=LinearDiscriminantAnalysis(n_components=2)
lad.fit(iris_scaled,iris.target)
iris_lda=lad.transform(iris_scaled)
print(iris_lda.shape)
```

[output]

(150, 2)

이제 LDA변환된 입력 데이터 값을 2차원 평명네 품종별로 표현해 보자.

```python
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

lda_columns=['lda_component_1','lda_component_2']
irisDF_lda=pd.DataFrame(iris_lda,columns=lda_columns)
irisDF_lda['target']=iris.target

makers=['^','s','o']

#setosa의 target값은 0, versicolor는 1, virginica는 2.각 target별로 다른 모양으로 산점도 표시
for i, maker in enumerate(makers):
  x_axis_data=irisDF_lda[irisDF_lda['target']==i]['lda_component_1']
  y_axis_data=irisDF_lda[irisDF_lda['target']==i]['lda_component_2']
  plt.scatter(x_axis_data,y_axis_data,marker=maker,label=iris.target_names[i])

plt.legend(loc='upper right')
plt.xlabel('lda_component_1')
plt.ylabel('lda_component_2')
plt.show()
```

[output]

![결과1](https://user-images.githubusercontent.com/77263283/126158711-9069a5c4-a21b-4632-9c52-de15343f1843.png)
