## LinearRegression을 이용한 보스턴 주택 가격 회귀 구현

LinearRegression 클래스를 이용해 선형 회귀 모델을 만들어보자. 사이킷런에 내장된 데이터 세트인 보스턴 주택 가격 데이터를 이용한다.

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_boston
%matplotlib inline

#boston 데이터 세트 로드
boston=load_boston()

#boston 데이터 세트 DataFrame 반환
bostonDF=pd.DataFrame(boston.data,columns=boston.feature_names)

#boston 데이터 세트의 target 배열은 주택 가격임. 이를 PRICE칼럼으로 DataFrame에 추가함
bostonDF['PRICE']=boston.target
print("Boston 데이터 세트 크키:",bostonDF.shape)
bostonDF.head()
```

[output]

![결과1](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/798a18a6-dc3c-4550-ac85-2b0a6830190f/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210617%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210617T102613Z&X-Amz-Expires=86400&X-Amz-Signature=1ef9fd3c590c43f216e2da24dbb12e49d62afec61ddf2f589906a5464e737a26&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

```python
bostonDF.info()
```

[output]

![결과2](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/91fb5e38-ddbd-45be-b8c1-6ed70d3aab10/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210617%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210617T102641Z&X-Amz-Expires=86400&X-Amz-Signature=74bffc115e4588110528d91e6a3a176a36b77bc28cf84e2a3c50708077400a33&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

데이터 세트 피처의 NULL값은 없으며 모두 실수형이다.

다음으로 각 칼럼이 회귀 결과에 미치는 영향이 어느 정도인지 시각화해서 알아보자.

```python
#2개의 행과 4개의 열을 가진 subplots를 이용. axs는 4x2개의 ax를 가짐
fig,axs=plt.subplots(figsize=(16,8),ncols=4, nrows=2)
lm_features=['RM','ZN','INDUS','NOX','AGE','PTRATIO','LSTAT','RAD']
for i,feature in enumerate(lm_features):
  row=int(i/4)
  col=i%4
  #시본의 regplot을 이용해 산점도와 선형 회귀 직선을 함께 표현
  sns.regplot(x=feature,y='PRICE',data=bostonDF,ax=axs[row][col])
```

[output]

![결과3](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/23688ac7-02e4-419a-85a5-9431be1181c2/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210617%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210617T102713Z&X-Amz-Expires=86400&X-Amz-Signature=d381349a9b53832277a0647e731ef8a075479048f99a2712ed8f54e267f06039&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

다른 칼럼보다 RM과 LSTAT의 PRICCE영향도가 가장 두드러지게 나타난다. RM(방 개수)은 양 방향의 선형성이 가장 크다. 즉 방의 크기가 클수록 가격이 증가하는 모습을 확연히 보여준다. LSTAT(하위 계층의 비율)은 음 방향의 선형성이 가장 크다.

이제 LinearRegression 클래스를 이용해 보스턴 주택 가격의 회귀 모델을 만들어보자.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

y_target=bostonDF['PRICE']
X_data=bostonDF.drop(['PRICE'], axis=1, inplace=False)

X_train,X_test,y_train,y_test=train_test_split(X_data,y_target,test_size=0.3,random_state=156)

#선형 회귀 OLS로 학습/평가 수행
lr=LinearRegression()
lr.fit(X_train,y_train)
y_preds=lr.predict(X_test)
mse=mean_squared_error(y_test,y_preds)
rmse=np.sqrt(mse)

print('MSE : {0:.3f}, RMSE : {1:.3f}'.format(mse,rmse))
print("Variance score : {0: .3f}".format(r2_score(y_test,y_preds)))
```

[output]

![결과4](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/fc4c6e00-c5f5-4383-8f84-eb8a895cab55/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210617%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210617T102754Z&X-Amz-Expires=86400&X-Amz-Signature=79443828394d4e62cfcb6b54110af76776779cb238a7b0cda9d9e62e87dbc54e&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

LinearRegression으로 생성한 주택가격 모델의 intercept(절편)과 coefficient(회귀 계수) 값을 보자.

```python
#회귀 계수를 큰 값 순으로 정렬하기 위해 Series로 생성. 인덱스 칼럼명에 유의
coeff=pd.Series(data=np.round(lr.coef_,1),index=X_data.columns)
coeff.sort_values(ascending=False)
```

[output]

![결과5](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/dc038eba-2eab-483c-acfc-5d1b41117483/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210617%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210617T102826Z&X-Amz-Expires=86400&X-Amz-Signature=17e341c71b1985408156d9e736af0275dfec6ca530cfb122c68d23c1496c4c88&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

RM이 양의 값으로 회귀 계수가 가장 크며, NOX 피처의 회귀 계수 - 값이 너무 커 보인다. 차츰 최적화를 수행하면서 피처 coefficients의 변화도 같이 살펴보자.

이번에는 5개의 폴드 세트에서 cross_val_score()를 이용해 교차 검증으로 MSE와 RMSE를 측정해 보자.

```python
from sklearn.model_selection import cross_val_score

#cross_val_score()로 5 폴드 세트로 MSE를 구현 한 뒤 이를 기반으로 다시 RMSE를 구함
neg_mse_scores=cross_val_score(lr,X_data,y_target,scoring='neg_mean_squared_error',cv=5)
rmse_scores=np.sqrt(-1*neg_mse_scores)
avg_rmse=np.mean(rmse_scores)

#cross_val_score(scoring="neg_mean,squared_Wrror")로 반환된 값은 모두 음수
print('5 folds의 개별 Negatve MSE scores:',np.round(neg_mse_scores,2))
print('5 folds의 개별 RMSE scores:',np.round(rmse_scores,2))
print('5 folds의 평균 RMSE: {0:.3f}'.format(avg_rmse))
```

[output]

![결과6](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/1440adaf-dcd5-4a12-a53e-858d03ed93e4/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210617%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210617T102849Z&X-Amz-Expires=86400&X-Amz-Signature=a80258d58684fadbd6b91373abe88273bbc1f2415430f3e33d4f0a439f6d187f&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

5개의 폴드 세트에 대해서 교차 검증을 수행한 결과, 평균 rmse는 약 5.829가 나왔다.
