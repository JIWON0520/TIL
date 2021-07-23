## 붓꽃 데이터 세트를 이용한 NMF실습

붓꽃 데이터를 NMF를 이용해 2개의 컴포넌트로 변환하고 이를 시각화해 보자.

```python
from sklearn.decomposition import NMF
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
%matplotlib inline

iris=load_iris()
iris_ftrs=iris.data
nmf=NMF(n_components=2)
nmf.fit(iris_ftrs)
iris_nmf=nmf.transform(iris_ftrs)
plt.scatter(x=iris_nmf[:,0],y=iris_nmf[:,1],c=iris.target)
plt.xlabel('NMF Component 1')
plt.ylabel('NMF Component 2')
```

[output]

![결과1](https://user-images.githubusercontent.com/77263283/126768734-5a465496-44cc-418a-934c-0ba48cd2730d.png)
