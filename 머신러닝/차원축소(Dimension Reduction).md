## 차원 축소의 개요

- 차원 축소: 매우 많은 피처로 구성된 다차원 데이터 세트의 차원울 축소 → 새로운 차원의 데이터 세트를 생성하는 것
- 수백 개 이상의 피처로 구성된 데이터 세트의 경우 상대적으로 작은 차원에서 학습된 모델보다 예측 신뢰도가 떨어진다.

    또한 피처가 많을 경우 개별 피처 간에 상관관계가 높을 가능성이 크다.

- 다차원의 피처를 차원 축소하면 더 직관적으로 데이터를 해석할 수 있다.
- 차원 축소는 피처 선택과 피처 추출로 나눌 수 있다.

    > 피처 선택: 특정 피처에 종속성이 강한 불필요한 피처는 아예 제거하고, 데이터의 특징을 잘 나타내는 주요 피처만 선택하는 것

    > 피처 추출: 기존 피처를 저차원의 중요 피처로 압축해서 추출하는 것(기존의 피처와는 완전히 다른것)

- 차원 축소의 의미는 단순한 데이터의 압축이 아닌 데이터를 더 잘 설명할 수 있는 잠재적인 요소를 추출하는데 있다.

## PCA(Principal Component Analysis)

- PCA는 가장 대표적인 차원축소 기법이다.
- 여러 변수간에 존재하는 상관관계를 이용해 이를 대펴하는 주성분을 추출해 차원을 축소하는 기법이다.
- PCA는 가장 높은 분산을 가지는 데이터의 축을 찾아 이 축으로 차원을 축소한다. (분산이 데이터의 특성을 가장 잘 나타내는 것으로 간주한다.)
- PCA는 제일 먼저 가당 튼 데이터 변동성을 기반으로 첫 번째 벡터 축을 생성하고, 두 번째 축은 이 벡터 축에 직각이 되는 벡터를 축으로 한다. 세 번째 축은 다시 두 번째 축과 직각이 되는 벡터를 설정하는 방식으로 축을 생성한다. ⇒이렇게 생성된 벡터 축에 원본 데이터를 투영하면 벡터 축의 개수만큼의 차원으로 데이터가 차원축소 된다.
- 보통 PCA는 다음과 같은 스텝으로 수행된다
    1. 입력 데이터 세트의 공분산 행렬을 생성한다.
    2. 공분산 행렬의 고유벡터와 고유값을 계산한다.
    3. 고유값이 가장 큰 순으로  K개(PCA 변환 차수만큼)만큼 고유벡터를 추출한다.
    4. 고유값이 가장 큰 순으로 추출된 고유벡터를 이용해 새롭게 입력 데이터를 변환한다.

    [붓꽃 데이터 세트를 이용한 PCA 실습](https://github.com/JIWON0520/TIL/blob/main/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/%EC%8B%A4%EC%8A%B5/%EC%B0%A8%EC%9B%90%20%EC%B6%95%EC%86%8C/%EB%B6%93%EA%BD%83%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%84%B8%ED%8A%B8%EB%A5%BC%20%EC%9D%B4%EC%9A%A9%ED%95%9C%20PCA%20%EC%8B%A4%EC%8A%B5.md)

    [신용카드 데이터 세트를 이용한 PCA실습](https://github.com/JIWON0520/TIL/blob/main/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/%EC%8B%A4%EC%8A%B5/%EC%B0%A8%EC%9B%90%20%EC%B6%95%EC%86%8C/%EC%8B%A0%EC%9A%A9%EC%B9%B4%EB%93%9C%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%84%B8%ED%8A%B8%EB%A5%BC%20%EC%9D%B4%EC%9A%A9%ED%95%9C%20PCA%EC%8B%A4%EC%8A%B5.md)

![PCA이미지](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/8acebf62-aa31-4c81-9c97-4c31fac7c9bf/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210717%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210717T042954Z&X-Amz-Expires=86400&X-Amz-Signature=01107a14181f4f7b71b62a4c13e68370acc20e38c8d7187f695b7ac56b834086&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

## LDA(Linear Discriminant Analysis)

- LDA는 선형 판별 분석법이다.
- LDA는 입력 데이터의 결정값 클래스를 최대한 분리할 수 있는 축을 찾는다.
- 클래스 간 분산과 클래스 내부 분산의 비율을 최대화하는 방식으로 차원을 축소한다.
- LDA는 다음과 같은 스텝으로 수행된다.
    1. 클래스 내부와 클래스 간 분산 행렬을 구한다.
    2. 두 행렬을 고유벡터로 분해한다.
    3. 고유값이 가장 큰 순으로 K개(LDA변환 차수만큼) 추출한다.
    4. 고유값이 가장 큰 순으로 추출된 고유벡터를 이용해 새롭게 입력 데이터를 변환한다.

[붓꽃 데이터 세트를 이용한 LDA실습](https://github.com/JIWON0520/TIL/blob/main/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/%EC%8B%A4%EC%8A%B5/%EC%B0%A8%EC%9B%90%20%EC%B6%95%EC%86%8C/%EB%B6%93%EA%BD%83%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%84%B8%ED%8A%B8%EB%A5%BC%20%EC%9D%B4%EC%9A%A9%ED%95%9C%20LDA%EC%8B%A4%EC%8A%B5.md)

![LDA이미지](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/e7364c1f-6dbc-4b91-96fb-db810abb2015/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210717%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210717T043038Z&X-Amz-Expires=86400&X-Amz-Signature=b0483895d34e9d49a0ab13722dd43ef84f30986418ca42a63fd474765f43c60a&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

## SVD(Singular Value Decomposition)

- PCV와 유사한 행렬 분해 기법을 이용
- PCA의 경우 정방행렬만을 고유벡터로 분해할 수 있지만 SVD는 행과 열의 크기가 다른 행렬에도 적용가능
- 일반적으로 SVD는 m X n 크기의 행렬 A를 다음과 같이 분해하는 것을 의미한다.

    ![SVD이미지](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/17d6fa05-c44c-41c6-8fb0-140d05857e79/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210717%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210717T043123Z&X-Amz-Expires=86400&X-Amz-Signature=71742d0fcb5114fd56e3f491a60d5898f69c6a1d426ca1871943195cc3032b4e&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

- 행렬 U와 V에 속한 벡터는 특이벡터이며, 모든 특이 벡터는 서로 직교하는 성질을 가진다.
- Σ는 대각 행렬이며, 행렬의 대각에 위치한 값(특이값)만 0이 아니고 나머지 위치의 값은 모두 0이다.
- SVD는 A의 차원이 m X n일 때, U의 차원이 m X m, Σ 차원이 m X n, Vt의 자원이 n X n으로 분해한다.
- Truncated SVD는 Σ의 대각원소 중에 상위 몇개만 추출해서 여기에 대응하는 U와 V의 원소도 함께 제거해 더욱 차원을 줄인 형태로 분해하는 것이다.

    [SVD실습](https://github.com/JIWON0520/TIL/blob/main/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/%EC%8B%A4%EC%8A%B5/%EC%B0%A8%EC%9B%90%20%EC%B6%95%EC%86%8C/SVD%EC%8B%A4%EC%8A%B5.md)

    [붓꽃 데이터 세트를 이용한 Truncated SVD 실습](https://www.notion.so/Truncated-SVD-3e2df29b0cda4c1a87ace5ba4fbafc4a)

## NMF(Non-Negative Matrix Factorization)

- NMF는 Truncated SVD와 같이 낮은 랭트를 통한 행렬 근사 방식의 변형이다.
- NMF는 원본 행렬 내의 모든 원소 값이 모두 양수라는게 보장되면 다음과 같이 좀 더 간단하게 두 개의 기반 양수 행렬로 분해될 수 있는 기법이다.

![NMF이미지](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/d344abb3-7d41-4aa4-911a-38e75fdf92f7/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210717%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210717T043150Z&X-Amz-Expires=86400&X-Amz-Signature=a0d5f65460f0367b50def4a65ae3e1b0d5e436418a49c04b0ac357b93a846b1f&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

- 이처럼 행렬 분해를 하게 되면 분해된 행렬은 잠재 요소를 특성으로 가지게 된다.
- 분해 행렬 W는 원본 행에 대해서 이 잠재 요소의 값이 얼마나 되는지에 대응하며, 분해 행렬 H는 이 잠재 요소가 원본 열로 어떻게 구성됐는지를 나타내는 행렬이다.
- NMF는 SVD와 유사하게 차원 축소를 통한 잠재 요소 도출로 이미지 변환 및 압축, 텍스트의 토픽 도출 등의 영역에서 사용되고 있다.

[NMF 차원축소 실습](https://www.notion.so/0674f08d2b66448e97068fd5d008a7f7)
