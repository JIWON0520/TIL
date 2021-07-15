## 엘라스틱넷 회귀를 이용한 보스턴 주택가격 예측 실습

사이킷런의 ElasticNet 클래스를 이용하여 보스턴 주택가격 예측 실습을 해보자.

```python
from sklearn.linear_model import ElasticNet

#l1_ratio는 0.7로 고정
coeff_df=pd.DataFrame()
elastic_alphas=[0.07,0.1,0.5,1,3]
for alpha in lasso_alphas:
  elastic=ElasticNet(alpha=alpha,l1_ratio=0.7)
  neg_mse_scores=cross_val_score(elastic,X_data,y_target,scoring='neg_mean_squared_error',cv=5)
  avg_rmse=np.mean(np.sqrt(-1*neg_mse_scores))
  print('alpha: {0}일 때 평균 RMSE: {1:.3f}'.format(alpha,avg_rmse))
  elastic.fit(X_data,y_target)
  coeff=pd.Series(data=elastic.coef_,index=X_data.columns)
  colname='alpha:'+str(alpha)
  coeff_df[colname]=coeff

sort_column='alpha:'+str(elastic_alphas[0])
coeff_df.sort_values(by=sort_column,ascending=False)
```

[output]

![결과1](https://user-images.githubusercontent.com/77263283/125778045-a2da36fa-0763-4c35-b0f0-5ccff3104031.png)

alpha가 0.5일 때 평균 RMSE가 5.467로 가장 좋은 예측 성능을 보이고 있다.

alpha 값에 따른 피처들의 회귀 계수들 값이 라쏘보다는 상대적으로 0이 되는 값이 적음을 알 수 있다.
