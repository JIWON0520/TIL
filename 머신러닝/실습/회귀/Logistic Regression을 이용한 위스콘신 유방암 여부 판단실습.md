## 로지스틱 회귀를 이용한 위스콘신 유방암 여부 판단실습

```python
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
cancer=load_breast_cancer()

#StandardScaler()로 평균이 0, 분산이 1로 데이터 분포도 변환
scaler=StandardScaler()
data_scaled=scaler.fit_transform(cancer.data)

X_train,X_test,y_train,y_test=train_test_split(data_scaled,cancer.target,test_size=0.3,random_state=0)

#로지스틱 회귀를 이용해 학습 및 예측 수행
lr_clf=LogisticRegression()
lr_clf.fit(X_train,y_train)
lr_pred=lr_clf.predict(X_test)

#정확도와 roc_auc 측정
print('accuracy:{0:.3f}'.format(accuracy_score(y_test,lr_pred)))
print('roc_auc:{0:.3f}'.format(roc_auc_score(y_test,lr_pred)))
```

[output]

accuracy:0.977
roc_auc:0.972

정확도가 약 97.7%, roc_auc가 약 97.2%이다.

사이킷런 LogisticRegression 클래스의 주요 하이퍼 파라미터로 penalty와 C가 있다. penalty는 규제의 유형을 성정하며 'l2'로 설정시 L2규제를, 'l1'로 설정 시 L1 규제를 뜻한다. 기본은 'l2'이다. C는 규제 강도를 조절하는 alpha값의 역수 이다. C 값이 작을 수록 규제 강도가 크다. 

GridSearchCV를 이용해 위스콘신 데이터 세트에서 이 하이퍼 파라미터를 최적화해 보자.

```python
from sklearn.model_selection import GridSearchCV

params={'penalty':['l2','l1'],
        'C':[0.01,0.1,1,5,10]}
grid_clf=GridSearchCV(lr_clf,param_grid=params,scoring='accuracy',cv=3)
grid_clf.fit(data_scaled,cancer.target)
print('최적 하이퍼 파라미터:{0}, 최적 평균 정확도:{1:.3f}'.format(grid_clf.best_params_,grid_clf.best_score_))
```

[output]

최적 하이퍼 파라미터:{'C': 1, 'penalty': 'l2'}, 최적 평균 정확도:0.975

로지스틱 회귀는 가볍고 빠르지만, 이진 분류 예측 성능도 뛰어난다. 로지스틱 회귀는 희소한 데이터 세트 분류에도 뛰어난 성능을 보여서 텍스트 분류에서도 자주 사용된다.
