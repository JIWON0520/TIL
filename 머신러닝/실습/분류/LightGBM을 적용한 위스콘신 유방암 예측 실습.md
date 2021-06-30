## LightGBM을 적용한 위스콘신 유방암 예측 실습

앞의 여러 실습에서 사용한 위스콘신 유방암 데이터 세트를 이용해 LightGBM을 실습해보자.

```python
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

dataset=load_breast_cancer()
ftr=dataset.data
target=dataset.target

#전에 데이터 중 80%는 학습용 데이터, 20%는 테스트용 데이터 추출
X_train,X_test,y_train,y_test=train_test_split(ftr,target,test_size=0.2,random_state=0)

lgbm_wrapper=LGBMClassifier(n_estimators=400)

evals=[(X_test,y_test)]
lgbm_wrapper.fit(X_train,y_train,early_stopping_rounds=100,eval_metric='logloss',eval_set=evals, verbose=True)
preds=lgbm_wrapper.predict(X_test)
pred_proba=lgbm_wrapper.predict_proba(X_test)[:,1]

get_clf_eval(y_test,preds,pred_proba)
```

[output]

![결과1](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/8a1c973d-41ea-4365-8f8b-d69b63a46ca3/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210630%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210630T121759Z&X-Amz-Expires=86400&X-Amz-Signature=01669c3fe4b9b8cda9a24ad32812a74d16c13f243c4c82f6f3db70a0459fea35&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

.....

![결과2](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/fadf7c30-bf96-4ba3-bade-f02c8aa2b16f/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210630%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210630T121813Z&X-Amz-Expires=86400&X-Amz-Signature=f5b0cad60cfbdd86d955e2367b5d489418ca93afcfde7cda32b1a0fb5d6be47f&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

조기 중단으로 186번 반복까지만 수행하고 학습을 종료했다. 정확도는 97.37%가 나왔다.

마지막으로 plot_importance클래스를 사용해 피처별 중요도를 살펴보자.

```python
#plot_importance()를 이용해 피처 중요도 시각화
from lightgbm import plot_importance
import matplotlib.pyplot as plt
%matplotlib inline

fig, ax= plt.subplots(figsize=(10,12))
plot_importance(lgbm_wrapper,ax=ax)
```

[output]

![결과3](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/7dde6f9f-8b36-40b3-879d-b07a44c7e4ed/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210630%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210630T121851Z&X-Amz-Expires=86400&X-Amz-Signature=e9f08ebfdb05e31c74986269118dfd28d562f9cbbfef0c6411b62c65312b732a&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)
