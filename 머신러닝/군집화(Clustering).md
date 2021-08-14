## K-평균 알고리즘

- K-평균은 군집화에서 가장 일반적으로 사용되는 알고리즘이다.

    K-평균 알고리즘: 임의의 지점을 군집 중심점으로 선택 → 군집 중심점은 데이터 포인트의 평균 지점으로 이동 → 다시 가까운 데이터 포인터의 평균 지점으로 이동 → 중심점의 이동이 없을때까지 반복

- K-평균의 장점:

    일반적인 군집화에서 가장 많이 활용되는 알고리즘이다.

    알고리즘이 쉽고 간결하다.

- K-평균의 단점:

     거리 기반 알고리즘으로 속성의 개수가 매우 많을 경우 군집화 정확도가 떨어진다.(이를 위해 PCA로 차원 감소를 적용해야 할 수도 있음)

    반복을 수행하는데, 반복횟수가 많을 경우 수행시간이 매우 느려진다.

    몇 개의 군집을 선택해야 할지 가이드하기어렵다.

![k-평균 이미지](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/8356cc0b-8978-4c4f-bac4-748315d767f9/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210814%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210814T111119Z&X-Amz-Expires=86400&X-Amz-Signature=80161e97a1ee07b35c1b63f9d7a535b2b33fab9f5d4a8f384f26bacddac4faa6&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

## 군집 평가(Cluster Evaluation)

- 대부분의 군집화 데이터 세트는 비교할만한 타깃 레이블을 가지고 있지 않아 군집화가 얼마나 효율적으로 되었는지 짐작하기 어렵다.
- 군집화 평가 방법으로는 실루엣 분석이 있다.

    실루엣 분석: 실루엣 계수(개별 데이터가 가지는 군집화 지표)를 기반으로 하는 군집 평가 방법

- 개별 데이터가 가지는 실루엣 계수는 해당 데이터가 같은 군집 내의 데이터와 얼마나 가깝게 군집화되어 있고, 다른 군집에 있는 데이터와는 얼마나 멀리 분리돼 있는지를 나타내는 지표

![군집평가](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/6aa6743d-f95e-4120-a13d-82e1c14b01e0/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210814%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210814T111148Z&X-Amz-Expires=86400&X-Amz-Signature=d63ab50bd1472be66da6c4bf39350abb6a20ee22b4edc5444a023f69e19e581c&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

a(i):해당 데이터 포인트와 같은 군집 내에 있는 다른 데이터 포인트와의 거리를 평균한 값

b(i):해당 데이터 포인트가 속하지 않은 군집 중 가장 가까운 군집과의 평균 거리

MAX( a(i),b(i) ): 정규화 하기 위한 값

- 실루엣 계수는 -1에서 1 사이의 값을 가지며, 1로 가까워질수록 근처의 군집과 더 멀리 떨어져 있다는 것이고 0에 가까울수록 근처의 군집과 가까워진다는 것이다, - 값을 아예 다른 군짐에 데이터 포인트가 할당됐음을 뜻한다.
- 좋은 군집화가 되려면 다음 기준 조건을 만족해야 한다.
    1. 전체 실루엣 계수의 평균값이 1에 가까울수록 좋다
    2. 전체 실루엣 계수의 평균값과 더불어 개별 군집의 평균값의 편차가 크지 않아야 한다.

[K-평균을 이용한 붓꽃 데이터 세트 군집화 및 평가](https://www.notion.so/K-56d21059effc47e4bf907396ff2d3dce)

## 평균 이동(Mean Shift)

- 평균 이동은 K-평균과 유사하게 중심을 군집의 중심으로 지속적으로 움직이면서 군집화를 수행한다.

    but) K-평균이 중심에 소속된 데이터의 평균 거리 중심으로 이동하는데 반해, 평균 이동은 중심을 데이터가 모여 있는 밀도가 가장 높은 곳으로 이동시킨다.

- 확률 밀도 함수를 이용한다. (주어딘 모델의 확률 밀도 함수를 찾기 위해서 KDE를 이용한다.)
- 특정 데이터를 반경 내의 데이터 분포 확률 밀도가 가장 높은 곳으로 이동하기 위해 주변 데이터와의 거리 값을 KDE 함수 값으로 입력한 뒤 그 반환 겂을 현재 위치레서 업데이트 하면서 이동하는 방식을 취한다.

![평균이동 이미지](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/b3259d24-7974-4bc3-929d-8a639accf810/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210814%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210814T111231Z&X-Amz-Expires=86400&X-Amz-Signature=fde6874f5c5ba9f37dfc65f4d6f2ea99e2a1b1f1fcb724144acd93583294cb68&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

- KDE는 개별 관측 데이터에 커널 함수를 적용한 뒤, 이 적용 값을 모두 더한 후 개별 관측 데이터의 건수로 나눠 밀도 확률 밀도 함수를 추정한다.
- KDE에서 중요한 점은 대역폭 h인데, 대역폭이 작으면 좁고 뾰족한 KDE를 가지게 되고 이는 과적합되기 쉽다.

    반면, 대역폭이 큰 값이면 KDE는 과도하게 평활화되어 과소적합되기 쉽다.

- 일반적으로 평균 이동 군집화는 대역폭이 클수록 평활화된  KDE로 인해 적은 수의 군집 중심점을 가지게 되며 대역폭이 적을수록 많은 수의 군집 중심점을 가진다.

## GMM(Gaussian Mixture Model)

- GMM 군집화는 군집화를 적용하고자 하는 데이터가 여러 개의 가우시안 분포를 가진 데이터 집합들이 섞여서 생성된 것이라는 가정하에 군집화를 수행하는 방식이다.

![GMM이미지](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/b538da06-8cc1-47f5-94d0-deeb741033e3/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210814%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210814T111312Z&X-Amz-Expires=86400&X-Amz-Signature=af3fd3477af3ca2a8a7005b91517c45bd7f834f3e3064df916b2444aaa99d081&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

- GMM운 데이터를 여러 개의 가우시안 분포가 섞인 것으로 간주한다. →섞인 데이터 분포에서 개별 유형의 가우시안 분포를 추출한다. →개별 데이터가 이중 어떤 정규 분포에 속하는지 결정하는 방식이다.

    ### GMM과 평균이동의 차이

    ![GMM이미지2](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c8599389-c213-42d3-9764-2f1c0864c42f/Untitled.pnghttps://s3.us-west-2.amazonaws.com/secure.notion-static.com/c8599389-c213-42d3-9764-2f1c0864c42f/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210814%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210814T111343Z&X-Amz-Expires=86400&X-Amz-Signature=b202cc03779b24aa5c065792117d190bea1d408904f01b34ef4be8585b145268&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

## DBSCAN

- DBSCAN은 밀도 기반 군집화이다.
- 특정 공간 내에 데이터 밀도 차리를 기반 알고리즘으로 하고 있어서 복잡한 기하학적 분포도를 가진 데이터 세트에 대해서 군집화를 잘 수행한다.
- DBSCAN에서 중요한 두 가지 파라미터는 입실론 주변 영역과 이 영역에 포함되는 최소 데이터 개수이다.
    1. 입실론 주변 영역(epsilon): 개별 데이터를 중심으로 입실론 반경을 가지는 원형의 영역
    2. 최소 데이터 개수(min points): 개별 데이터의 입실론 주변 영역에 포함되는 타 데이터의 개수

![DBSCAN](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/b1f12343-8448-4d60-8780-259ca99ea4f0/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210814%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210814T111413Z&X-Amz-Expires=86400&X-Amz-Signature=c76b49556973662b2780e7788fb7c57d3c7b2b493eef5a3a31b13146af565570&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

- 핵심 포인트(Core point): 주변 영역 내에 최소 데이터 개수 이상의 타 데이터를 가지고 있을 경우 해당 데이터 포인트를 다음과 같이 정의한다.
- 이웃 포인트(Neighbor pint): 주변 영역 내에 위치한 타 데이터를 이웃 포인트라고 한다.
- 경계 포인트(Border point): 주변 영역 내에 최소 개수 이상의 이웃 포인트를 가지고 있지 않지만 핵심 포인트를 이웃 포인트로 가지고 있는 데이터를 경계 포인트라고 한다.
- 잡음 포인트(Noise pint): 최소 데이터 개수 이상의 이웃 포인터를 가지고 있지 않으며, 핵심 포인트도 이웃 포인트로 가지고 있지 않은 데이터를 잡음 포인트라고 한다.
- DBSCAN은 입실론 주변 영역의 최소 데이터 개수를 포함하는 밀도 기준을 충족하는 데이터인 핵심 포인트를 연결하면서 군집화응 구성하는 방식이다.
- 일반적으로 eps의 값을 크게 하면 반경이 커져 포함하는 데이터가 많아지므로 노이즈 데이터 개수가 작아진다.
- min_samples를 크게하면 주어진 반경 내에서 더 많은 데이터를 포함 시텨야 하므로 노이즈 데이터 개수가 커지게 된다.

[군집화 실습](https://www.notion.so/ee4d8ab9d3bb433896b3fe9adbb5d4ee)
