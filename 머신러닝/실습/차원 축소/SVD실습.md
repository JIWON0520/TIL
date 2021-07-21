## SVD실습

넘파이의 SVD를 이용해 SVD 연산을 수행하고, SVD로 분해가 어떤 식으로 되는지 간단한 예제를 통해 살펴보자.

```python
#넘파이의 svd 모듈 import
import numpy as np
from numpy.linalg import svd

# 4x4 랜덤 행렬 a 생성
np.random.seed(121)
a=np.random.randn(4,4)
print(np.round(a,3))
```

[output]

[[-0.212 -0.285 -0.574 -0.44 ]
 [-0.33   1.184  1.615  0.367]
 [-0.014  0.63   1.71  -1.327]
 [ 0.402 -0.191  1.404 -1.969]]

이렇게 생성된 a 행렬에 SVD를 적용해 U,sigma,Vt를 도출해보자.

```python
U,Sigma,Vt=svd(a)
print(U.shape,Sigma.shape,Vt.shape)
print('U matrix:\n',np.round(U,3))
print('Sigma Value:\n',np.round(Sigma,3))
print('V transpose matrix:\n',np.round(Vt,3))
```

[output]

(4, 4) (4,) (4, 4)
U matrix:
 [[-0.079 -0.318  0.867  0.376]
 [ 0.383  0.787  0.12   0.469]
 [ 0.656  0.022  0.357 -0.664]
 [ 0.645 -0.529 -0.328  0.444]]
Sigma Value:
 [3.423 2.023 0.463 0.079]
V transpose matrix:
 [[ 0.041  0.224  0.786 -0.574]
 [-0.2    0.562  0.37   0.712]
 [-0.778  0.395 -0.333 -0.357]
 [-0.593 -0.692  0.366  0.189]]

시그마 행렬의 경우 대각에 위치한 값만 0이 아니므로 0이 아닌 값만 1차원 행렬로 표현한다.

U, Vt행렬이 4 x 4, Sigma행렬의 경우는 1차원 행렬로 반환되었다.

분해된 U,Sigma,Vt를 이용해 다시 원본 행렬로 정확히 복원되는지 확인해 보자. 이 세개의 행렬을 내적하면 된다.

```python
#Sigma를 다시 0을 포함한 대칭행렬로 변환
Sigma_mat=np.diag(Sigma)
a_=np.dot(np.dot(U,Sigma_mat),Vt)
print(np.round(a_,3))
```

[output]

[[-0.212 -0.285 -0.574 -0.44 ]
 [-0.33   1.184  1.615  0.367]
 [-0.014  0.63   1.71  -1.327]
 [ 0.402 -0.191  1.404 -1.969]]

U,Sigma,Vt를 이용해 a_는 원본 행렬 a와 동일하게 복원됨을 알 수 있다.

이번에는 데이터 세트가 로우 간 의존성이 있을 경우 어떻게 Sigma 값이 변하고, 이에 따른 차원 축소가 진행될 수 있는지 알아보자. 일부러 의존성을 부여하기 위해 a 행렬의 3번째 로우를 '첫 번째 로우+ 두 번째 로우'로 업데이트하고, 4번째 로우는 첫 번째 로우와 같다고 엡데이트하자.

```python
a[2]=a[0]+a[1]
a[3]=a[0]
print(np.round(a,3))
```

[output]

[[-0.212 -0.285 -0.574 -0.44 ]
 [-0.33   1.184  1.615  0.367]
 [-0.542  0.899  1.041 -0.073]
 [-0.212 -0.285 -0.574 -0.44 ]]

이제 a 행렬은 이전과 다르게 로우 간 관계가 매우 높아졌다.

이 데이터를 SVD로 다시 분해해 보자.

```python
#다시 SVD를 수행해 Sigma값 확인
U,Sigma,Vt=svd(a)
print(U.shape,Sigma.shape,Vt.shape)
print('Sigma Value:\n',np.round(Sigma,3))
```

[output]

(4, 4) (4,) (4, 4)
Sigma Value:
 [2.663 0.807 0.    0.   ]

이전과 차원은 같지만 Sigma값 중 2개가 0으로 변했다. 즉, 선형 독립인 로우 벡터의 개수가 2개라는 의미이다.

이렇게 분해된 U,Sigma,Vt를 이용해 다시 원본 행렬로 복원해 보자. 이번에는 U,Sigma.Vt의 전체 데이터를 이용하지 않고 Sigma의 0에 대응되는 U,Sigma,Vt의 데이터를 제외하고 복원해 보자.

```python
#U행렬의 경우는 Sigma와 내적을 수행하므로 Sigma의 앞 2행에 대응되는 앞 2열만 추출
U_=U[:,:2]
Sigma_=np.diag(Sigma[:2])
#Vt행렬의 경우는 앞 2행만 추출
Vt_=Vt[:2]
print(U_.shape,Sigma_.shape,Vt_.shape)
#U,Sigma,Vt의 내적을 수행하며, 다시 원본 행렬 복원
a_=np.dot(np.dot(U_,Sigma_),Vt_)
print(np.round(a_,3))
```

[output]

(4, 2) (2, 2) (2, 4)
[[-0.212 -0.285 -0.574 -0.44 ]
 [-0.33   1.184  1.615  0.367]
 [-0.542  0.899  1.041 -0.073]
 [-0.212 -0.285 -0.574 -0.44 ]]

a_행렬은 원본 행렬인 a와 동일하게 복원 되었다.

이번에는 Truncated SVD를 이용해 행렬을 분해해 보자.

Truncated SVD는 시그마 행렬에 있는 대각원소, 즉 특이값 중 상위 일부 데이터만 추출해 분해하는 방식이다. 이렇게 분해하면 인위적으로 더 적은 차원의 행렬으로 분해하기 때문에 원본 행렬을 정확하게 다시 복원할 수는 없다. 하지만 데이터 정보가 압축되어 분해됨에도 불구하고 상당한 수준으로 원본 행렬을 근사할 수 있다.

임의의 원본 행렬 6 x 6을 Normal SVD로 분해해 분해된 행렬의 차원과 Sigma 행렬 내의 특이값을 확인한 뒤 다시 Truncated SVD로 분해해 분해된 행렬의 차원, Sigma 행렬 내의 특의값, 그리고 Truncated ACD로 분해된 행렬의 내적을 계산하여 다시 복원된 데이터와 원본 데이터를 비교해 보자.

```python
import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import svd

#원본 행렬을 출력하고 SVD를 적용할 경우 U,Sigma,Vt의 차원 확인
np.random.seed(121)
matrix=np.random.random((6,6))
print('원본 행렬:\n',matrix)
U,Sigma,Vt=svd(matrix,full_matrices=False)
print('\n분해 행렬 차원:',U.shape,Sigma.shape,Vt.shape)
print('\nSigma값 행렬:',Sigma)

#Truncated SVD로 Sigma 행렬의 특의값을 4개로 하여 Truncated SVD수행
num_components=4
U_tr,Sigma_tr,Vt_tr=svds(matrix,k=num_components)
print('\nTruncated SVD 분해 행렬 차원:',U_tr.shape,Sigma_tr.shape,Vt_tr.shape)
print('\nTruncated SVD Sigma값 행렬:',Sigma_tr)
matrix_tr=np.dot(np.dot(U_tr,np.diag(Sigma_tr)),Vt_tr)

print('\nTruncated SVD로 분해 후 복원 행렬:\n',matrix_tr)
```

[output]

원본 행렬:
 [[0.11133083 0.21076757 0.23296249 0.15194456 0.83017814 0.40791941]
 [0.5557906  0.74552394 0.24849976 0.9686594  0.95268418 0.48984885]
 [0.01829731 0.85760612 0.40493829 0.62247394 0.29537149 0.92958852]
 [0.4056155  0.56730065 0.24575605 0.22573721 0.03827786 0.58098021]
 [0.82925331 0.77326256 0.94693849 0.73632338 0.67328275 0.74517176]
 [0.51161442 0.46920965 0.6439515  0.82081228 0.14548493 0.01806415]]

분해 행렬 차원: (6, 6) (6,) (6, 6)

Sigma값 행렬: [3.2535007  0.88116505 0.83865238 0.55463089 0.35834824 0.0349925 ]

Truncated SVD 분해 행렬 차원: (6, 4) (4,) (4, 6)

Truncated SVD Sigma값 행렬: [0.55463089 0.83865238 0.88116505 3.2535007 ]

Truncated SVD로 분해 후 복원 행렬:
 [[0.19222941 0.21792946 0.15951023 0.14084013 0.81641405 0.42533093]
 [0.44874275 0.72204422 0.34594106 0.99148577 0.96866325 0.4754868 ]
 [0.12656662 0.88860729 0.30625735 0.59517439 0.28036734 0.93961948]
 [0.23989012 0.51026588 0.39697353 0.27308905 0.05971563 0.57156395]
 [0.83806144 0.78847467 0.93868685 0.72673231 0.6740867  0.73812389]
 [0.59726589 0.47953891 0.56613544 0.80746028 0.13135039 0.03479656]]

6 x 6 행렬을 SVD분해하면 U,Sigma,Vt가 각각 (6,6)(6,)(6,6)차원 이지만, Truncated SVD의 n_components를 4로 설정해 U, Sigma, Vt를 (6,4),(4,),(4,6)로 각각 분해했다.

Truncated SVD로 분해된 행렬로 다시 복원할 경우 완벽하게 복원되지 않고 근사적으로 복원됨을 알 수 있다.
