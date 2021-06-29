## 신용카드 데이터 세트를 이용한 PCA실습

더 많은 데이터 피처를 가진 데이터 세트를 적은 PCA컴포넌트 기반으로 변환한 뒤, 예측 영향도가 어떻게 되는지 변환된 PCA데이터 세트에 기반해서 비교해 보자.

```python
#header로 의미 없는 첫 행 제거, iloc로 기존 id제거
import pandas as pd

df=pd.read_excel('/content/drive/MyDrive/Colab Notebooks/credit_card.xls',header=1,sheet_name='Data').iloc[0:,1:]
print(df.shape)
df.head(3)
```

[output]

![결과1](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/d8104237-c5f2-48b0-98ae-7578b71d2b64/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210629%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210629T121901Z&X-Amz-Expires=86400&X-Amz-Signature=b2f78b9e802008eb33ab7a5cc94928d65d502e66fbfde2293417de63afd0b765&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

신용카드 데이터 세트는 30,000개의 레코드와 24개의 속성을 가지고 있다. 이 중에서 'defualt payment next month'속성이 Target값으로 '다음달 연체 여부'를 의미하며 '연체'일 경우 1, '정상납부'가0이다.

PAY_0 다음에 PAY_2이므로 PAY_1칼럼을 PAY_1로 칼럼명을 변경하는 등 기본적인 데이터 수정을 하자.

```python
df.rename(columns={'PAY_0':'PAY_1','default payment next month':'default'},inplace=True)
y_target=df['default']
X_features=df.drop('default',axis=1)
```

해당 데이터 세트는 23개의 속성 데이터가 있느나 각 속성끼리 상관도가 매우 높다.

상관도를 시각화해 보자.

```python
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

corr=X_features.corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr,annot=True,fmt='.1g')
```

[output]

![결과2](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/5130bfb7-82b7-4a3d-a0e2-bbdbce7a474e/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210629%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210629T121935Z&X-Amz-Expires=86400&X-Amz-Signature=65f3a4f3e80f75201a024d320762a70450086149f80ec619564e69751d5a29fc&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

BILL_AMT1 ~ BILL_AMT6 6개 속성끼리의 상관도가 대부분 0.9이상으로 매우 높음을 알 수 있다. 이보다는 낮지만 PAY_1~PAY_6까지의 속성 역시 상관도가 높다. 이렇게 높은 상관도를 가진 속성들은 소수의 PCA만으로도 자연스럽게 이 속성들의 변동성을 수용할 수 있다. 

이 BILL_AMT1 ~ BILL_AMT6까지 6개 속성을 2개의 커포넌트로 PCA변환한 뒤 개별 컴포넌트의 변동성을 알아보자.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#BILL_ATM1 ~ BILL_ATM6까지 6개의 속성명 생성
cols_bill=['BILL_AMT'+str(i) for i in range(1,7)]
print('대상 속성명:',cols_bill)

#2개의 PCA 속성을 가진 PCA객체 생성하고, explained_variance_ratio_ 계산을 위해 fit호출
scaler=StandardScaler()
df_cols_scaled=scaler.fit_transform(X_features[cols_bill])
pca=PCA(n_components=2)
pca.fit(df_cols_scaled)
pca.fit_transform(df_cols_scaled)
print('PCA Component별 변동성:',pca.explained_variance_ratio_)
```

[output]

대상 속성명: ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
PCA Component별 변동성: [0.90555253 0.0509867 ]

단 2개의 PCA컴포넌트만으로도 6개 속성의 변동성을 약 95%이상 설명할 수 있으며 특히 첫 번째 PCA축으로 90%의 변동성을 수용할 정도로 6개 속성의 상관도가 매우 높다.

이번에는 원본 데이터 세트와 6개의 컴포넌트로 PCA변환한 데이터 세트의 분류 예측 결과를 상호 비교해 보자.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

rcf=RandomForestClassifier(n_estimators=300,random_state=0)
scores=cross_val_score(rcf,X_features,y_target,scoring='accuracy',cv=3)

print('CV=3인 경우의 개별 Fold세트별 정확도:',scores)
print('평균 정확도:{0:.4f}'.format(np.mean(scores)))
```

[output]

CV=3인 경우의 개별 Fold세트별 정확도: [0.8087 0.8206 0.8208]
평균 정확도:0.8167

3개의 교차 검증 세트에서 평균 예측 정확도는 약 81.67%를 나타내었다.

이번에는 6개의 컴포넌트로 PCA변환한 데이터 세트에 대해서 동일하게 분류 예측을 적용해 보자.

```python
#원본 데이터 세트에 먼저 StandardScaler적용
scaler=StandardScaler()
df_scaled=scaler.fit_transform(X_features)

#6개의 컴포넌트를 가진 PCA 변환을 수행하고 cross_val_score()로 분류 예측 수행
pca=PCA(n_components=6)
df_pca=pca.fit_transform(df_scaled)
scores_pca=cross_val_score(rcf,df_pca,y_target,scoring='accuracy',cv=3)

print('CV=3인 경우의 PCA변환된 개별 Fold세트별 정확도:',scores_pca)
print('PCA 변환 데이터 세트 평균 정확도:{0:.4f}'.format(np.mean(scores_pca)))
```

[output]

CV=3인 경우의 PCA변환된 개별 Fold세트별 정확도: [0.7919 0.7974 0.8028]
PCA 변환 데이터 세트 평균 정확도:0.7974

전체 23개 속성의 약 1/4 수준인 6개의 PCA 컴포넌트만으로도 원본 데이터를 기반으로 한 분류 예측 결과보다 약 1~2% 정도의 예측 성능 저하만 발생했습니다. 전체 속성의 1/4 정도만으로도 이정도 수치의 예측 성능을 유지할 수 있다는 것은 PCA의 뛰어난 압축 능력을 잘 보여주는 것이다.
