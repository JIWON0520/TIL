## PolynomialFeature 클래스를 이용한 다항 회귀 구현

PolynomialFeature를 이용해 단항값[x1,x2]를 2차 다항값 [1,x1,x2,x1^2,x1x2,x2^2]값으로 변환하는 예제이다.

```python
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

#다항식으로 변환한 단항식 생성, [[0,1],[2,3]]의 2X2 행렬 생성
X=np.arange(4).reshape(2,2)
print('일차 단항식 계수 피처:\n',X)

#degree=2인 2차 다항식으로 변환하기 위해 PolynomialFeatures를 이용해 변환
poly=PolynomialFeatures(degree=2)
poly.fit(X)
poly_ftr=poly.transform(X)
print('변환된 2차 다항식 계수 피처:\n',poly_ftr)
```

[output]

![결과1](https://user-images.githubusercontent.com/77263283/125933864-e4244ff8-f8a6-4122-b083-d7ff037c0bce.png)

단항 계수 피처[x1,x2]를 2차 다항계수 [1,x1,x2,x1^2,x1x2,x2^2]로 변경하므로 첫 번째 입력 단항 계수 피처 [x1=0, x2=1]은 [1,x1=0,x2=1,x1^2=0,x1x2=0,x2^2=1] 형태인 [1,0,1,0,0,1]로 변환된다. 이렇게 변환된 Polynomial 피처에 선형 회귀를 적용해 다항 회귀를 구현한다. 

PolynomialFeatures 클래스가 어떻게 단항식 값을 다항식 값으로 변환하는지 설명했으니, 이번에는 3차 다항 계수를 이용해 3차 다항 회귀 함수식을 PolynomialFeatures와 LinearRegression 클래스를 이용해 유도해 보자.

```python
def polynomial_func(X):
  y=1+2*X[:,0]+3*X[:,0]**2+4*X[:,1]**3
  return y

X=np.arange(4).reshape(2,2)
print('일차 단항식 계수 feature:\n',X)
y=polynomial_func(X)
print('삼차 다항식 계수 결정값:\n',y)
```

[output]

![결과2](https://user-images.githubusercontent.com/77263283/125933880-e1e0de7b-a679-4f36-afb7-99cf515c2a86.png)

이제 일차 단항식 계수를 삼차 다항식 계수로 변환하고, 이를 선형 회귀에 적용하면 다항 회귀로 구현된다.

```python
from sklearn.linear_model import LinearRegression
#3차 다항식 변환
poly_ftr=PolynomialFeatures(degree=3).fit_transform(X)
print('3차 다항식 계수 features:\n',poly_ftr)

#LinearRegression에 3차 다항식 계수 feature와 3차 다항식 결정값으로 학습 후 회귀 계수 확인
model=LinearRegression()
model.fit(poly_ftr,y)
print('Polynomial 회귀 계수 \n',np.round(model.coef_,2))
print('Polynomial 회귀 Shape \n',model.coef_.shape)
```

[output]

![결과3](https://user-images.githubusercontent.com/77263283/125933889-1101010d-b8e0-4940-8502-2f0380d33ea8.png)

일차 단항식 계수는 2개 였지만, 3차 다항식 Polynomial변환 이후에는 다항식 계수 피처가 10개로 늘어난다. 이 피처 데이터 세트에 LinearRegression을 통해 3차 다항 회귀 형태의 다항 회귀를 적용하면 회귀 계수가 10개로 늘어난다. 회귀 계수 [0. 0.18 0.18 0.36 0.54 0.72 0.72 1.08 1.62 2.34]가 도출 되었으며, 실제 회귀 계수 값인 [1,2,0,3,0,0,0,0,0,4]와는 차이가 있지만 다항 회귀로 근사하고 있음을 알 수 있다.
