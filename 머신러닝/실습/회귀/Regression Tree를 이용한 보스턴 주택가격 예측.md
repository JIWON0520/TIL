## 회귀 트리를 이용한 보스턴 주택가격 예측

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

#보스턴 데이터 세트 로드
boston=load_boston()
bostonDF=pd.DataFrame(boston.data,columns=boston.feature_names)

bostonDF['PRICE']=boston.target
y_target=bostonDF['PRICE']
X_data=bostonDF.drop(['PRICE'],axis=1,inplace=False)

rf=RandomForestRegressor(random_state=0,n_estimators=1000)
neg_mse_scores=cross_val_score(rf,X_data,y_target,scoring='neg_mean_squared_error',cv=5)
rmse_scores=np.sqrt(-1*neg_mse_scores)
avg_rmse=np.mean(rmse_scores)

print("5 교차 검증의 개별 Negative MSE scores:",np.round(neg_mse_scores,2))
print('5 교차 검증의 개별 RMSE scores:',np.round(rmse_scores,2))
print('5 교차 검증의 평균 RMSE: {0:.3f}'.format(avg_rmse))
```

[output]

5 교차 검증의 개별 Negative MSE scores: [ -7.93 -13.06 -20.53 -46.31 -18.8 ]
5 교차 검증의 개별 RMSE scores: [2.82 3.61 4.53 6.8  4.34]
5 교차 검증의 평균 RMSE: 4.420

이번에는 랜덤 포레스트뿐만 아니라 결정 트리, GBM, XGBoost, LightGBM의 Regressor를 모두 이용해 보스턴 주택 가격 예측을 수행하자. 이를 위해 입력 모델과 데이터 세트를 입력 받아 교차 검증으로 평균 RMSE를 계산해주는 함수를 만들자.

```python
def get_model_cv_prediction(model,X_data,y_target):
  neg_mse_scores=cross_val_score(model,X_data,y_target,scoring='neg_mean_squared_error',cv=5)
  rmse_scores=np.sqrt(-1*neg_mse_scores)
  avg_rmse=np.mean(rmse_scores)
  print('#### ',model.__class__.__name__,' ####')
  print('5 교차 검증의 평균 RMSE: {0:.3f}'.format(avg_rmse))
```

이제 다양한 유형의 트리를 생성하고, 이를 이용해 보스턴 주택 가격을 예측해 보자.

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

dt_reg=DecisionTreeRegressor(random_state=0,max_depth=4)
rf_reg=RandomForestRegressor(random_state=0,n_estimators=1000)
gb_reg=GradientBoostingRegressor(random_state=0,n_estimators=1000)
xgb_reg=XGBRegressor(n_estimators=1000)
lgb_reg=LGBMRegressor(n_estimators=1000)

#트리 기반의 회귀 모델을 반복하면서 평가 수행
models=[dt_reg,rf_reg,gb_reg,xgb_reg,lgb_reg]
for model in models:
  get_model_cv_prediction(model,X_data,y_target)
```

[outpupt]

####  DecisionTreeRegressor  ####
5 교차 검증의 평균 RMSE: 5.978
####  RandomForestRegressor  ####
5 교차 검증의 평균 RMSE: 4.420
####  GradientBoostingRegressor  ####
5 교차 검증의 평균 RMSE: 4.269

####  XGBRegressor  ####
5 교차 검증의 평균 RMSE: 4.089
####  LGBMRegressor  ####
5 교차 검증의 평균 RMSE: 4.646

회귀 트리 Regereesor 클래스는 선형 회귀와 다른 처리 방식이므로 회귀 계수를  제공하는  coef_ 속성이 없다. 대신 feature_importances_를 이용해 피처별 중요도를 시각화해 보자

```python
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

rf_reg=RandomForestRegressor(n_estimators=1000)
rf_reg.fit(X_data,y_target)

feature_series=pd.Series(data=rf_reg.feature_importances_, index=X_data.columns)
feature_series=feature_series.sort_values(ascending=False)
sns.barplot(x=feature_series,y=feature_series.index)
```

[output]

![결과1](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/61f1e582-e31f-4612-bd5d-4cde31208847/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210624%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210624T134244Z&X-Amz-Expires=86400&X-Amz-Signature=f00c492906f378ace230381d7fe52e1af9ea0f92877c689cc315b0182eff6619&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

이번에는 회귀 트리가 어쩧게 예측값을 판단하는지 선형 회귀와 비교해 시각화해 보자.

결정 트리의 하이퍼 파라미터인 max_depth의 크기를 변화시키면서 어떻게 회귀 트리 예측선이 변화하는지 살펴보자.

2차원 평면상에서 회귀 예측선을 쉽게 표현하지 위해서 Price와 가장 밀접한 양의 상관간계를 가지는 RM 칼럼만 추출하겠다. 

보스턴 데이터 세트에서 100개의 데이터만 추출하여 X축에 RM칼럼을, Y축에 Price을 두고 산점도형태로 시각화 하겠다.

```python
bostonDF_sample=bostonDF[['RM','PRICE']]
bostonDF_sample=bostonDF_sample.sample(n=100,random_state=0)
plt.figure()
plt.scatter(bostonDF_sample.RM,bostonDF_sample.PRICE)
```

[otuput]

![결과2](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/fa5721c4-8a3d-4651-893f-ac5673c6fe93/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210624%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210624T134318Z&X-Amz-Expires=86400&X-Amz-Signature=f8ea87061cfe102c7782610b8f13e8032178fcd1dce0466be2fc8109dc05283d&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

다음으로 보스턴 데이터 세트에 대해 LinearRegression과 DecisionTreeRegression를 max_depth를 각각 2,7로 해서 학습해 보자.

```python
from sklearn.linear_model import LinearRegression
#선형 회귀와 결정 트리 기반의 Regression 생성. DecisionTreeRegression의 max_depth는 각각 2,7,
lr_reg=LinearRegression()
dt_reg2=DecisionTreeRegressor(max_depth=2)
dt_reg7=DecisionTreeRegressor(max_depth=7)

#실제 예측을 적용헐 테스트용 데이터 세트를 4.5~8.5까지의 100개 데이터 세트로 생성
X_test=np.arange(4.5,8.5,0.04).reshape(-1,1)

#보스턴 주택 가격 데이터에서 시각화를 위해 피처는 RM만, 그리고 결정 데이터인 PRICE추출
X_feature=bostonDF_sample['RM'].values.reshape(-1,1)
y_target=bostonDF_sample['PRICE'].values.reshape(-1,1)

#학습과 예측 수행
lr_reg.fit(X_feature,y_target)
dt_reg2.fit(X_feature,y_target)
dt_reg7.fit(X_feature,y_target)

pred_lr=lr_reg.predict(X_test)
pred_dt2=dt_reg2.predict(X_test)
pred_dt7=dt_reg7.predict(X_test)

fig,(ax1,ax2,ax3)=plt.subplots(figsize=(14,4),ncols=3)

#X축 값을 4.5~8.5로 변환하며 입력했을 때 선형 회귀와 결정트리 예측선 시각화
#선형 회귀로 학습된 모델 회귀 예측선
ax1.set_title('Linear Regression')
ax1.scatter(bostonDF_sample.RM,bostonDF_sample.PRICE,c='y')
ax1.plot(X_test,pred_lr,label='linear',linewidth=2)

#DecisionTreeGression의 max_depth를 2로 했을 때 회귀 예측선
ax2.set_title('DecisionTreeRegression: \n max_depth=2')
ax2.scatter(bostonDF_sample.RM,bostonDF_sample.PRICE,c='y')
ax2.plot(X_test,pred_dt2,label='max_depth:2',linewidth=2)

#DecisionTreeGression의 max_depth를 7로 했을 때 회귀 예측선
ax3.set_title('DecisionTreeRegression: \n max_depth=7')
ax3.scatter(bostonDF_sample.RM,bostonDF_sample.PRICE,c='y')
ax3.plot(X_test,pred_dt7,label='max_depth:7',linewidth=2)
```

[output]

![결과3](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/a791f200-cfb6-4cfa-ad2b-c17914b1ba5b/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210624%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210624T134410Z&X-Amz-Expires=86400&X-Amz-Signature=97bc51332b96eb4cd07686899f6fd0860820016bcb1682a9c701c38b13877237&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

선형 회귀는 직선으로 예측 회귀선을 표현하는 데 반해, 회귀 트리의 경우 분할되는 데이터 지점에 따라 브랜치를 만들면서 계단 형채로 회귀선을 만든다. 

DecisionTreeRegression의 max_depth=7인 경우에는 학습 데이터 세트의 이상치 데이터도 학습하면서 복잡한 계단 형태의 회귀선을 만들어 과적합이 되기 쉬운 모델이 되었다.
