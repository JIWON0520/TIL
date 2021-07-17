## 릿지 회귀를 이용한 보스턴 주택가격 예측

앞에서 한 보스턴 주택 가격 예측 실습을 릿지 회귀를 이용해 실습해 보자.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

#alpha=10으로 설정해 릿지 회귀 수행
ridge=Ridge(alpha=10)
neg_mse_scores=cross_val_score(ridge,X_data,y_target,scoring='neg_mean_squared_error',cv=5)
rmse_scores=np.sqrt(neg_mse_scores*-1)
avg_rmse=np.mean(rmse_scores)
print('5 folds의 개별 Negatve MSE scores:',np.round(neg_mse_scores,2))
print('5 folds의 개별 RMSE scores:',np.round(rmse_scores,2))
print('5 folds의 평균 RMSE: {0:.3f}'.format(avg_rmse))
```

[output]

![결과1](https://user-images.githubusercontent.com/77263283/126025367-0c521f8e-a93d-4ffe-86c2-d7839987af3a.png)
릿지의 5개 폴드세트 검증에서 평균RMSE가 5.518입니다. 앞 예제의 규제가 없는 LinearRegression의 RMSE 평균인 5.82보다 더 뛰어난 예측 성능을 보여준다.

이번에는 릿지의 alpha값을 변화시키면서  RMSE와 회귀계수를 시각화하고 DataFrame에 저장해 보자.

```python
#릿지에 사용될 alpha 파라미터의 값을 정의
alphas=[0,0.1,1,10,100]

#alphas list 값을 반복하면서 alpha에 따른 평균 rmse를 구함
for alpha in alphas:
  ridge=Ridge(alpha=alpha)

  #cross_val_score를 이용해 5 폴드의 평균 RMSE를 계산
  neg_mse_scores=cross_val_score(ridge,X_data,y_target,scoring='neg_mean_squared_error',cv=5)
  avg_rmse=np.mean(np.sqrt(-1*neg_mse_scores))
  print("alpha {0}일 때 5 fold의 평균 RMSE: {1:.3f}".format(alpha,avg_rmse))
```

[output]

![결과2](https://user-images.githubusercontent.com/77263283/126025371-5e84779a-4df8-43ab-80ed-9ebe6f8ea695.png)

alpha가 100일 때 RMSE가 5.330으로 가장 좋다.

이번에는 alpha 값의 변화에 따른 피처의 회귀 계수 값을 가로 막대 그래프로 시각화해 보자.

```python
#각 alpha에 따른 회귀 계수 값을 시각화하기 위해 5개의 열로 된 matplitilb 축 생성
fig,axs=plt.subplots(figsize=(18,6),nrows=1,ncols=5)
#각 alpha에 따른 회귀 계수 값을 데이터로 저장하기 위한 DataFrame 생성
coeff_df=pd.DataFrame()

#alpha 리스트 값을 차례로 입력해 회귀 계수 값 시각화 및 데이터 저징, pos는 axis의 위치 지정
for pos,alpha in enumerate(alphas):
  ridge=Ridge(alpha=alpha)
  ridge.fit(X_data,y_target)
  #alpha에 따른 피처별로 회귀 계수를 Series로 변환하고 이를 DataFrame의 칼럼으로 추가.
  coeff=pd.Series(data=ridge.coef_,index=X_data.columns)
  colname='alpha:'+str(alpha)
  coeff_df[colname]=coeff
  #막대 그래프로 각 alpha 값에서의 회귀 계수를 시각화, 회귀 계수값이 높은 순으로 표현
  coeff=coeff.sort_values(ascending=False)
  axs[pos].set_title(colname)
  axs[pos].set_xlim(-3,6)
  sns.barplot(x=coeff.values,y=coeff.index,ax=axs[pos])
plt.show()
```

[output]

![결과3](https://user-images.githubusercontent.com/77263283/126025373-e67d2ebe-beea-4000-819d-28a466c433e5.png)
alpha 값을 계속 증가 시킬수록 회귀 계수 값은 지속적으로 작아짐을 알 수 있다. 특히 NOX의 경우 alpha값을 계속 증가시킴에 따라 회귀 계수가 크게 작아지고 있다.

DataFrame에 저장된 alpha값의 변화에 따른 릿지 회귀 계수 값을 구해 보자.

```python
ridge_alphas=[0,0.1,1,10,100]
sort_column='alpha:'+str(ridge_alphas[0])
coeff_df.sort_values(by=sort_column,ascending=False)
```

[output]

![결과4](https://user-images.githubusercontent.com/77263283/126025376-59fd25ca-7485-4324-8697-cd44fc197f61.png)
alpha 값이 증가하면서 회귀 계수가 지속적으로 작아지고 있음을 알 수 있다. 하지만 릿지 회귀의 경우에는 회귀 계수를 0으로 만들지는 않는다.
