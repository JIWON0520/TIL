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

![결과](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/80be9dc3-9437-4ae6-8cc9-5ba79aaf2964/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210622%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210622T123843Z&X-Amz-Expires=86400&X-Amz-Signature=4b0f4a77e74e9bb648706fea5353a7aee2873aed0c842b9641dc5cfb033630c4&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

alpha가 0.5일 때 평균 RMSE가 5.467로 가장 좋은 예측 성능을 보이고 있다.

alpha 값에 따른 피처들의 회귀 계수들 값이 라쏘보다는 상대적으로 0이 되는 값이 적음을 알 수 있다.
