## XGBoost를 적용한 위스콘신 유방암 예측 실습

위스콘신 유방암 데이터 세트를 활용하여 XGBoost실습을 진행해보자.

위스콘신 유방암 데이터 세트는 종양의 크기, 모양 등의 다양한 속성값을 기반으로 악성 종양인지 양성 종양 인지를 분류한 데이터 세트이다.

위스콘신 데이터 세트를 로드하고 데이터들을 살펴보자.

```python
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

dataset=load_breast_cancer()
X_features=dataset.data
y_label=dataset.target

cancer_df=pd.DataFrame(data=X_features, columns=dataset.feature_names)
cancer_df['target']=y_label
cancer_df.head(3)
```

[output]

![결과1](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/257fb3af-87b5-4eea-b578-dd196229a90a/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210608%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210608T150859Z&X-Amz-Expires=86400&X-Amz-Signature=f8b2a5626eb9f1be7da0023a0afaf25e6d00e7c6c9341a2107ca3f19a922414e&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

종양과 관련된 많은 속성이 숫자형으로 되어있다. 타깃 레이블 값의 종류는 악성이면 0, 양성이면 1의 값을 갖는다.

레이블의 분포를 확인해보자.

```python
print(dataset.target_names)
print(cancer_df['target'].value_counts())
```

[output]

![결과2](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/24db7236-90c1-4ee5-a83e-981315a50d18/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210608%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210608T151043Z&X-Amz-Expires=86400&X-Amz-Signature=643efec5d2909783b155cb4e05bd4524336cf404bc9e3a5b86b2d86fce89a69c&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

양성인 1값이 357건, 악성인 0값이 212개로 구성되어있다.

전체 데이터중 80%를 학습용으로, 20%를 테스트용으로 분할해보자.

```python
X_train,X_test,y_train,y_test=train_test_split(X_features,y_label,test_size=0.2,random_state=0)
print(X_train.shape,X_test.shape)
```

[output]

![결과3](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/830e3d3b-d001-4319-a945-c39b1651fb46/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210608%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210608T151110Z&X-Amz-Expires=86400&X-Amz-Signature=5817a498e2f6f4849007d0cfd600bd9cdedc48f6cd5952058ad55a6ec1a6fd31&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

이제 XGBClassifier 클래스를 이용해 학습과 예측을 수행하자.

```python
xgb_wrapper=XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
xgb_wrapper.fit(X_train,y_train)
w_pred=xgb_wrapper.predict(X_test)
w_pred_proba=xgb_wrapper.predict_proba(X_test)[:,1]
get_clf_eval(y_test,w_pred,w_pred_proba)
```

[output]

![결과4](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/78fad0cf-31a4-4d6e-8b0d-a594d7d7fc39/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210608%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210608T151201Z&X-Amz-Expires=86400&X-Amz-Signature=b4d04195cb2164ebc4a53c2c11501fe0af60240144af80ce3b04776ad82bb592&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

조기 중단은 XGBoost의 train 함수에 early_stopping_rounds 파라미터를 입력하여 설정한다. early_stopping_rounds 파라미터를 설정해 조기 중단을 수행하기 위해서는 반드시 eval_set과 eval_metric이 함께 설정돼야한다. XGBoost는 반복마다 eval_set으로 지정된 데이터 세트에서 eval_metric의 지정된 평가 지표로 예측 오류를 측정한다.

XGBoost는 수행 속도를 개선하기 위해 조기 중단 기능을 제공한다. 조기중단은 XGBoost가 수행 성능을 개선하기 위해서 더 이상 지표 개선이 없을 경우에 num_boost_round 횟수를 모두 채우지 않고 중간에 반복을 빠져 나올 수 있도록 하는 것이다.

```python
xgb_wrapper=XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=3)
evals=[(X_test,y_test)]
xgb_wrapper.fit(X_train,y_train,early_stopping_rounds=60, eval_metric='logloss',eval_set=evals, verbose=True)
ws100_preds=xgb_wrapper.predict(X_test)
ws100_pred_proba=xgb_wrapper.predict_proba(X_test)[:,1]
get_clf_eval(y_test,ws100_preds,ws100_pred_proba)
```

[output]

![결과5](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/2f1511b2-1a5e-4e7b-a90e-7fd6156b607e/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210608%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210608T151412Z&X-Amz-Expires=86400&X-Amz-Signature=a83fe6fe72c15f992acfae30a3bac94a1f2df6b4fe92c624513fb4c57cb57bf2&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

......

![결과6](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/6683c65e-a74a-45b4-8b89-63074e09eadb/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210608%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210608T151432Z&X-Amz-Expires=86400&X-Amz-Signature=d1e7b83be0b68a0ab87c7d84239b167422240dd76eebb319041fb1fd0fac0cef&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

n_estimators를 500으로 설정해도 500번 반복을 수행하지 않고 231번 반복학습 한 후 학습을 완료했음을 알 수 있다.  결과를 보면 조기중단 되지 않은 결과보다 수치가 떨어졌지만, 큰 차이는 아니다. 하지만 조기중단 값을 너무 적게 설정하면 예측 성능이 저하될 우려가 크다. 

피쳐의 중요도를 시각화 하는 모듈인 plot_importance()으로 피쳐의 중요도를 살펴보자.

```python
from xgboost import plot_importance
import matplotlib.pyplot as plt
%matplotlib inline

fig,ax=plt.subplots(figsize=(10,12))
plot_importance(xgb_wrapper,ax=ax)
```

[output]

![결과7](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/00ec87bb-f44c-4e8d-a707-19fe61a2d00c/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210608%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210608T151526Z&X-Amz-Expires=86400&X-Amz-Signature=7b8e6dcb901c842c0c1c8f97a8ac8bcd45bbc2a9b036863e0287fea738582253&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)
