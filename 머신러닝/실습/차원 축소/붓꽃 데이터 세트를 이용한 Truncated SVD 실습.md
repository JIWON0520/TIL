## 붓꽃 데이터 세트를 이용한 Truncated SVD 실습

붓꽃 데이터 세트를 TruncatedSVD를 이용해 변환해 보자.(사이킷런의 TruncatedSVD 클래스는 사이파이의 svds와 같이 Truncated SVD 연산을 수행해 U, Sigma, Vt를 반환하지 않는다.)

```python
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
%matplotlib inline

iris=load_iris()
iris_ftrs=iris.data
#2개의 주요 컴포넌트로 TruncatedSVD 변환
tsvd=TruncatedSVD(n_components=2)
tsvd.fit(iris_ftrs)
iris_tsvd=tsvd.transform(iris_ftrs)

#산점도 2차언으로 TruncatedSVD 변환된 데이터 표현. 푼동은 색깔로 구분
plt.scatter(x=iris_tsvd[:,0],y=iris_tsvd[:,1],c=iris.target)
plt.xlabel('TruncatedSVD component 1')
plt.ylabel('TruncatedSVD component 2')
```

[output]

![결과1](https://user-images.githubusercontent.com/77263283/126586644-044dba25-1337-4789-9a4b-b533b95187bd.png)

TruncatedSVD 변환 후 품종별러 어느 정도 클러스터링이 가능할 정도로 각 변환 속성으로 뛰어난 고유성을 가지고 있음을 알 수 있다.

사이킷런의 TruncatedSVD와 PCA클래스 모두 SVD를 이용해 행렬을 분해한다.

붓꽃 데이터를 스케일링으로 변환한 뒤에 TrncatedSVD와 PCA클래스 변환을 해보면 두 개가 거의 동일함을 알 수 있다.

```python
from sklearn.preprocessing import StandardScaler

#붓꽃 데이터를 StandardScaler로 변환
scaler=StandardScaler()
iris_scaled=scaler.fit_transform(iris_ftrs)

#스케일링된 데이터를 기반으로 TruncatedSVD 변환 수행
tsvd=TruncatedSVD(n_components=2)
tsvd.fit(iris_scaled)
iris_tsvd=tsvd.transform(iris_scaled)

#스케일링된 데이터를 기반으로 PCA 변환 수행
pca=PCA(n_components=2)
pca.fit(iris_scaled)
iris_pca=pca.transform(iris_scaled)

#Truncated SVD 변환 데이터를 왼쪽에, PCA 변환 데이터를 오른쪽에 표현
fig,(ax1,ax2)=plt.subplots(figsize=(9,4),ncols=2)
ax1.scatter(x=iris_tsvd[:,0],y=iris_tsvd[:,1],c=iris.target)
ax2.scatter(x=iris_pca[:,0],y=iris_pca[:,1],c=iris.target)
ax1.set_title('Truncated SVD Transformed')
ax2.set_title('PCA Transformed')
```

[output]

![결과2](https://user-images.githubusercontent.com/77263283/126586648-608c2f85-f77d-4322-a5b0-ecce4e96915d.png)

두 개의 변환 행렬 값과 원본 속성변 컴포넌트 비율값을 실제로 서로 비교해 보면 거의 같음을 알 수 있다.

```python
print((iris_pca-iris_tsvd).mean())
print((pca.components_-tsvd.components_).mean())
```

[outuput]

2.3607620488104906e-15
2.7755575615628914e-17

모두 0에 가까운 값이므로 2개의 변환이 서로 동일함을 알 수 있다. 즉, 데이터 세트가 스케일링으로 데이터 중신이 동일해지면 사이킷런의 SVD와 PCA는 동일한 변환을 수행한다. 이는 PCA가 SVD알고리즘으로 구현됐음을 의미한다.
