## K-평균을 이용한 붓꽃 데이터 세트 군집화

붓꽃 데이터를 이용해 K-평균 군집화를 수행해 보자.

### 붓꽃 세트 군집화

꽃받침, 꽃잎의 길이에 따라 각 데이터의 군집화가 어떻게 결정되는지 확인해 보고, 이를 분류 값과 비교해 보자.

```python
from sklearn.preprocessing import scale
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
%matplotlib inline

iris=load_iris()
#더 편리한 핸들링을 위해 DataFrame으로 변환
irisDF=pd.DataFrame(iris.data,columns=[['sepal_length','sepal_width','petal_length','petal_width']])
irisDF.head(3)
```

[output]

![결과1](https://user-images.githubusercontent.com/77263283/130068665-d2b4cedc-b4db-4e48-8313-18c4d49a89ba.png)

붓꽃 데이터 세트를 3개 그룹으로 군집화해 보자.

```python
kmeans=KMeans(n_clusters=3,init='k-means++',max_iter=300,random_state=0)
kmeans.fit(irisDF)
```

fit()을 수행해 irisDF 데이터에 대한 군집화 수행 결과가 kmeans 객체 변수로 반환되었다.

kmeans의 labels_속성값을 확인해 보면 irisDF의 각 데이터가 어떤 중심에 속하는지를 알 수 있다,

```python
print(kmeans.labels_)
```

[output]

[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 2 0 0 0 0 2 0 0 0 0 0 0 2 2 0 0 0 0 2 0 2 0 2 0 0 2 2 0 0 0 0 0 2 0 0 0 0 2 0 0 0 2 0 0 0 2 0 0 2]

labels_의 값이 0,1,2로 되어있으며, 이는 각 레코드가 첫 번째 군집, 두 번째 군집, 세 번째 군집에 속함을 의미한다.

실제 붓꽃 품종 분류 값과 얼마나 차이가 나는지로 군집화가 효과적으로 됐는지 확인해 보자.

```python
irisDF['target']=iris.target
irisDF['cluster']=kmeans.labels_
iris_result=irisDF.groupby(['target','cluster'])['sepal_length'].count()
print(iris_result)
```

[output]

target  cluster
0       1          50
1       0           2
        2          48
2       0          36
        2          14

분류 타깃이 0값인 데이터는 1번 군집으로 모두 잘 그루핑되었다. Target 1 값 데이터는 2개만 2번 군집으로 그루핑되었고, 나머지 48개는 모두 0번 군집으로 그루핑되었다. 하지만 Target 2값 데이터는 0번 군집에 36개, 2번 군집에 14개로 분산되어 그루핑되었다.

이번에는 붓꽃 데이터 세트의 군집화를 시각화해 보자. 붓꽃 데이터 세트의 속성이 4개이므로 2차원 평면에 적합치 않아 PCA를 이용해 4개의 속성을 2개로 차원 축소한 뒤에 X좌표, Y좌표로 개별 데이터를 표현하도록 하자.

```python
from sklearn.decomposition import PCA

pca=PCA(n_components=2)
pca_transformed=pca.fit_transform(iris.data)

irisDF['pca_x']=pca_transformed[:,0]
irisDF['pca_y']=pca_transformed[:,1]

markers=['^','s','o']

for i,marker in enumerate(markers):
  x_axis_data=irisDF[irisDF['cluster']==i]['pca_x']
  y_axis_data=irisDF[irisDF['cluster']==i]['pca_y']
  plt.scatter(x_axis_data,y_axis_data,marker=marker)

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('3 Cluster Visualization by 2 PCA Components')
plt.show()
```

[output]

![결과2](https://user-images.githubusercontent.com/77263283/130068685-61e222eb-4118-4359-bba2-c0b625631123.png)

cluster 1을 나타내는 네모는 명확히 다른 군집과 잘 분리되어있다. cluster0을 나타내는 세모와 cluster2를 나타내는 동그라미는 상당 수준 준리되어있지만 네모만큼 명확하게 분리되어 있지 않다.

### 군집화 평가

이제 군집화 결과를 실루엣 분석으로 평가해 보자.

```python
from sklearn.metrics import silhouette_samples, silhouette_score

#iris의 모든 개별 데이터에 실루엣 계수 값을 구함
socre_samples=silhouette_samples(iris.data,irisDF['cluster'])
print('silhouette_samples() return 값의 shape',socre_samples.shape)

#irisDF에 실루엣 계수 칼럼 추가
irisDF['silhouette_coef']=socre_samples

#모든 데이터의 평균 실루엣 계수 값을 구함.
average_score=silhouette_score(iris.data,irisDF['cluster'])
print('붓꽃 데이터 세트 Silhouette Analysis Score:{0:.3f}'.format(average_score))
irisDF.head(3)
```

[output]

![결과3](https://user-images.githubusercontent.com/77263283/130068697-7a96be70-2af6-4d14-86ba-ba1a2a4e433a.png)

평균 실루엣 계수 값이 약 0.553이다. 1번 군집의 경우 0.8이상의 높은 실루엣 계수 값을 나타내고 있다. 평균 실루엣 계수 값이 낮은 이유는 1번 군집이 아닌 다른 군집의 경우 실루엣 계수 값이 평균 보다 낮기 때문일 것이다.

군집별 평균 실루엣 계수 값을 알아보자.

```python
irisDF.groupby('cluster')['silhouette_coef'].mean()
```

[output]

cluster
0    0.451105
1    0.798140
2    0.417320

1번 군집은 실루엣 계수 평균값이 약 0.79인데 반해, 0번은 약 0.45, 1전은 약 0.41로 상대적으로 평균값이 1번에 비해 낮다.

### 군집 개수 최적화

군집 개수를 젼화시키면서 K-평균 군집을 수행했을때 개별 군집별 평균 실루엣계수 값을 시각화하는 함수를 정의하자.

```python
### 여러개의 클러스터링 갯수를 List로 입력 받아 각각의 실루엣 계수를 면적으로 시각화한 함수 작성
def visualize_silhouette(cluster_lists, X_features): 
    
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import math
    
    # 입력값으로 클러스터링 갯수들을 리스트로 받아서, 각 갯수별로 클러스터링을 적용하고 실루엣 개수를 구함
    n_cols = len(cluster_lists)
    
    # plt.subplots()으로 리스트에 기재된 클러스터링 수만큼의 sub figures를 가지는 axs 생성 
    fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols)
    
    # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 실루엣 개수 시각화
    for ind, n_cluster in enumerate(cluster_lists):
        
        # KMeans 클러스터링 수행하고, 실루엣 스코어와 개별 데이터의 실루엣 값 계산. 
        clusterer = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0)
        cluster_labels = clusterer.fit_predict(X_features)
        
        sil_avg = silhouette_score(X_features, cluster_labels)
        sil_values = silhouette_samples(X_features, cluster_labels)
        
        y_lower = 10
        axs[ind].set_title('Number of Cluster : '+ str(n_cluster)+'\n' \
                          'Silhouette Score :' + str(round(sil_avg,3)) )
        axs[ind].set_xlabel("The silhouette coefficient values")
        axs[ind].set_ylabel("Cluster label")
        axs[ind].set_xlim([-0.1, 1])
        axs[ind].set_ylim([0, len(X_features) + (n_cluster + 1) * 10])
        axs[ind].set_yticks([])  # Clear the yaxis labels / ticks
        axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        
        # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현. 
        for i in range(n_cluster):
            ith_cluster_sil_values = sil_values[cluster_labels==i]
            ith_cluster_sil_values.sort()
            
            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = cm.nipy_spectral(float(i) / n_cluster)
            axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, \
                                facecolor=color, edgecolor=color, alpha=0.7)
            axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
            
        axs[ind].axvline(x=sil_avg, color="red", linestyle="--")
```

붓꽃 데이터 세트를 이용해 K-평균 수행 시 최적의 군집 개수를 알아보자.

```python
visualize_silhouette([2,3,4,5],iris.data)
```

[output]

![결과4](https://user-images.githubusercontent.com/77263283/130068710-451d915e-f600-47d8-a51f-5ad3d2478d89.png)

붓꽃 데이터를 K-평균으로 군집화 할 경우 군집 개수를 2개로 하는 것이 가장 좋아보인다. 3개의 경우 평균 실루엣 계수 값고 2개 보다 작을 뿐더러 1번 군집과 다른 0번, 2번 군집과의 실루엣 계수의 편차가 크다.
