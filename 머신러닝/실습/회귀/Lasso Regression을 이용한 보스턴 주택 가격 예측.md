## 라쏘 회귀를 이용해 보스턴 주택 가격 예측하기

사이킷런의 Lasso 클래스를 이용해 보스턴 주캑 가격 예측 실습을해 보자.

이전 실습 참고

[LinearRegression을 이용한 보스턴 주택 가격 회귀 구현](https://github.com/JIWON0520/TIL/blob/main/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/%EC%8B%A4%EC%8A%B5/%ED%9A%8C%EA%B7%80/LinearRegression%EC%9D%84%20%EC%9D%B4%EC%9A%A9%ED%95%9C%20%EB%B3%B4%EC%8A%A4%ED%84%B4%20%EC%A3%BC%ED%83%9D%20%EA%B0%80%EA%B2%A9%20%ED%9A%8C%EA%B7%80%20%EA%B5%AC%ED%98%84.md)

[Ridge Regression으로 보스턴 주택가격 예측하기](https://github.com/JIWON0520/TIL/blob/main/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/%EC%8B%A4%EC%8A%B5/%ED%9A%8C%EA%B7%80/Ridge%20Regression%EC%9C%BC%EB%A1%9C%20%EB%B3%B4%EC%8A%A4%ED%84%B4%20%EC%A3%BC%ED%83%9D%EA%B0%80%EA%B2%A9%20%EC%98%88%EC%B8%A1%ED%95%98%EA%B8%B0.md)

```python
from sklearn.linear_model import Lasso

coeff_df=pd.DataFrame()
lasso_alphas=[0.07,0.1,0.5,1,3]
for alpha in lasso_alphas:
  lasso=Lasso(alpha=alpha)
  neg_mse_scores=cross_val_score(lasso,X_data,y_target,scoring='neg_mean_squared_error',cv=5)
  avg_rmse=np.mean(np.sqrt(-1*neg_mse_scores))
  print('alpha: {0}일 때 평균 RMSE: {1:.3f}'.format(alpha,avg_rmse))
  lasso.fit(X_data,y_target)
  coeff=pd.Series(data=lasso.coef_,index=X_data.columns)
  colname='alpha:'+str(alpha)
  coeff_df[colname]=coeff

sort_column='alpha:'+str(lasso_alphas[0])
coeff_df.sort_values(by=sort_column,ascending=False)
```

[output]

![결과1](https://user-images.githubusercontent.com/77263283/125778215-564a3e90-b57e-41cc-b4c4-7c0a32e5216e.png)

alpha의 값이 0.07일 때 평균 RMSE 값이 5.612로 가장 낮게 나타났다.

alpha의 크기가 증가함에 따하 일부 피처의 회귀계수는 아예 0으로 바뀌고 있다. 회귀 계수가 0인 피처는 회귀 식에서 제외되면서 피처 선택의 효과를 얻을 수 있다.
