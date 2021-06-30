## GBM을 이용한 사용자 행동인식 분류 실습

이전의 '결정 트리를 사용한 사용자 행동 인식 분류 실습'에서 사용한 데이터 세트 사용.

[랜덤 포레스트 방식을 이용한 사용자 행동인식 분류 실습](https://github.com/JIWON0520/TIL/blob/main/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/%EC%8B%A4%EC%8A%B5/%EB%B6%84%EB%A5%98/%EB%9E%9C%EB%8D%A4%20%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8%20%EB%B0%A9%EC%8B%9D%EC%9D%84%20%EC%9D%B4%EC%9A%A9%ED%95%9C%20%EC%82%AC%EC%9A%A9%EC%9E%90%20%ED%96%89%EB%8F%99%EC%9D%B8%EC%8B%9D%20%EB%B6%84%EB%A5%98%20%EC%8B%A4%EC%8A%B5.md)

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import time
import warnings
warnings.filterwarnings('ignore')

#결정 트리에서 사용한 get_human_dataset()을 이용해 학습/테스트용 DataFrame반환
Xtrain,X_test,y_train,y_test=get_human_dataset()

#GBM 수행 시간 측정을 위함. 시작 시간 설정
start_time=time.time()

gb_clf=GradientBoostingClassifier(random_state=0)
gb_clf.fit(X_train,y_train)
gb_pred=gb_clf.predict(X_test)
accuracy=accuracy_score(y_test,gb_pred)

print('GBM 정확도: {0:.4f}'.format(accuracy))
print("GBM 수행시간:{0:.1f}초".format(time.time()-start_time))
```

[output]

![결과1](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/73449749-aec1-4011-80b8-7ba3f1194fd3/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210630%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210630T121635Z&X-Amz-Expires=86400&X-Amz-Signature=b33be4f3f7002f0c0a4956cc2b028f4dc9eba4beba2c503942a67d57aa4c18b0&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

기본 하이퍼 파라미터만으로 93.89%의 예측 정확도로 앞의 랜덤 포레스트보다 나은 예측 성능을 나타냈다. 그렇지 않은 경우도 있겠지만, 일반적으로 GBM이 랜덤 포레스트보다는 예측 성능이 조금 뛰어난 경우가 많다. 그러나 수행시간이 오래 걸리고, 하이퍼 파라미터 튜닝 노력도 필요하다. 위의 코드를 실행하고 결과도출까지 10분이 걸렸다...

이번에는 하이퍼 파리미터를 튜닝해서 실행해보자.

```python
from sklearn.model_selection import GridSearchCV

params={
    'n_estimators':[100,500],
    'learning_rate':[0.05,0.1]
}

grid_cv=GridSearchCV(gb_clf,param_grid=params,cv=2,verbose=1)
grid_cv.fit(X_train,y_train)
print('최적 하이퍼 파라미터:\n',grid_cv.best_params_)
print('최고 예측 정확도:{0:.4f}'.format(grid_cv.best_score_))
```

[output]

![결과2](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/6f8da5c8-9d4b-49f5-be52-42f8dae30f11/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210630%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210630T121659Z&X-Amz-Expires=86400&X-Amz-Signature=59425242b6345555be790d833c933a4185c0d1faca1f0df90ef3fc67033f4767&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

learning_rate가 0.1, n_estimators가 500일때 2개의 교차 검증 세트에서 90.11%의 정확도가 최고로 도출되었다. 이 설정을 그대로 테스트 데이터 세트에 적용해 예측 정확도를 확인해 보자

```python
#GRidSearchCV를 이용해 최적으로 학습된 estimator로 예측 수행
gb_pred=grid_cv.best_estimator_.predict(X_test)
gb_accuracy=accuracy_score(y_test,gb_pred)
print('GBM 정확도:{0:.4f}'.format(gb_accuracy))
```

![결과3](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/ee7ab7a1-9522-495b-9a5d-0b45423f74dc/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210630%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210630T121725Z&X-Amz-Expires=86400&X-Amz-Signature=557a3492d0885fcbc7179e4fb9bcee0177b8546f10085e23e8dee43dc4cbff73&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

테스트 데이터 세트에서 약 94.20%의 정확도가 나왔다.
