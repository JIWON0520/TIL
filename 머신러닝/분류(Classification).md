# 분류(Classification)란?

- 분류는 대표적인 지도학습의 유형이다.

- 학습 데이터로 주어진 데이터로 주어진 데이터의 피쳐와 레이블값을 머신러닝 알고리즘으로 학습해 모델을 생성한 후, 생성된 모델에 새로운 데이터 값이 주어졌을 때 어떤 레이블 값을 갖는지 예측하는 것이다.
# 결정트리

- 결정트리는 ML알고리즘 중 직관적으로 이해하기 쉬운 알고리즘이다.

- 데이터에 있는 규칙을 학습을 통해 자동으로 찾아내 트리 기반의 분류 규칙을 만드는 것이다.

- 사이킷런은 DecisionTreeClassifier와 DecisionTreeRegressor 클래스를 제공한다.

- 정보의 균일도만 신경쓰므로 피쳐의 스케일링과 정규화 같은 전처리 작업이 필요없다.

but)결정 트리의 가장 큰 단점은 과적합으로 정확도가 떨어진다는 점이다. 따라서 파라미터를 사전에 튜닝해야 한다. (트리의 깊이를 제한 한다던지, 말단 노드가 되기위한 최소한의 샘플데이터를 늘인다던지 등)

[결정트리 과적합 예제](https://github.com/JIWON0520/TIL/blob/main/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/%EC%8B%A4%EC%8A%B5/%EA%B2%B0%EC%A0%95%ED%8A%B8%EB%A6%AC%20%EA%B3%BC%EC%A0%81%ED%95%A9%20%EC%8B%A4%EC%8A%B5.md)

![결정트리 이미지](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/892a8bc5-5ade-46c6-b7f9-8fefca5628d5/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210607%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210607T143026Z&X-Amz-Expires=86400&X-Amz-Signature=3e07cc88dc313dbe52dfd67573b25a60ed7f6125ae6f3ca3971e8267610ea2aa&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)
- 결정트리를 시각화 한 모습 (자세한 실습과정은 밑의 링크 참조)

[결정트리를 이용한 붓꽃분류 실습](https://www.notion.so/d0021f64d50142f0bdc2f49c9052bcab)
# 앙상블 학습

- 앙상블 학습을 통한 분류는 여러 개의 분류기(Classifier)를 생성하고 그 예측을 결합함으로써 보다 정확한 최종 예측을 도출하는 기법을 말한다.

- 어려운 문제의 결론을 내기 위해 여러 명의 전문가로 위원회를 구성해 다양한 위견을 수렴하고 결정하듯이 앙상블 학습의 목표는 다양한 분류기의 예측 결과를 결합함으로써 다양한 분류기의 예측 결과를 결합함으로서 단일 분류기보다 신뢰성이 높은 예측값을 얻는 것이다.

- 이미지나 영상 등의 비정형 데이터의 분류는 딥러닝이 뛰어난 성능을 보이고 있지만, 대부분의 정형 데이터 분류시에는 앙상블이 뛰어난 성능을 나타내고있다.

- 앙상블의 유형에는 보팅(Voting), 배깅(Bagging), 부스팅(Boostiong)의 세가지로 나눌 수 있으며, 이외에도 스태킹을 포함한 다양한 앙상블 방법이 있다.

### 보팅(Voting)

- 보팅 방법에는 하드 보팅(Hard Voting)과 소프트 보팅(Soft Voting)의 두 가지가 있다. 

> 하드보팅:  다양항 분류기가 예측한 결괏값들중 다수의 분류기가 결정한 예측값을 최종 보팅 결괏값으로 선정하는 것. (다수결 원칙)

> 소프트 보팅: 분류기들의 레이블 값 결정 확률을 모두 더하고 이를 평균해서 이들 중 확률이 가장 높은 레이블 값을 최종 보팅 결괏값으로 선정. 일반적인 보팅 방법.

- 사이킷런은 VotingClassifier 클래스를 제공함으로써 보팅 방식의 앙상블을 구현할 수 있다. (밑의 실습 참고)

[보팅을 이용한 위스콘신 유방함 예측 분석](https://www.notion.so/6497957dbd2e492c8fb057f58129c8a5)

- 보팅으로 여러개의 분류기를 결합한다고 해서 무조건 개별 분류기보다 예측 성능이 향상되지 않는다. 데이터의 특성과 분포 등 다양한 요건에 따라 오히려 기반 분류기 중 가장 좋은 분류기의 성능이 보팅했을 때보다 나을 수 있다.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6090b115-f5f7-45ab-8216-b03c8f7923ad/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6090b115-f5f7-45ab-8216-b03c8f7923ad/Untitled.png)

### 배깅(Bagging)

⇒배깅은 앞에서 소개한 보팅과는 다르게 같은 알고리즘(결정 트리)으로 여러개의 분류기를 만들어서 보팅으로 최종 결정하는 알고리즘이다. 

⇒또한 각각의 분류기가 같은 학습데이터를 이용해 학습하는 보팅과 달리 배깅은 데이터 샘플링을 거쳐 학습을 수행한다.

⇒대표적인 배깅 알고리즘은 랜덤 포레스트이다.

-랜덤 포레스트는 여러 개의 결정 트리 분류기가 전체 데이터에서 배깅 방식으로 각자의 데이터를 샘플링해 개별적으로 학습을 수행한 뒤 최종적으로 모든 분류기가 보팅을 통해 예측 결정을 한다.

-랜덤 포레스트의 개별 분류기가 학습하는 데이터 세트는 전체 데이터에서 일부가 중첩되게 샘플링 된 데이터 세트이다. (부트스트래핑(bootstrappping)분할 방식)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/12c472df-8c09-49a7-9a6b-0c5dbfbb35ca/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/12c472df-8c09-49a7-9a6b-0c5dbfbb35ca/Untitled.png)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/01975723-afb8-439e-89b6-8f281ea21a1a/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/01975723-afb8-439e-89b6-8f281ea21a1a/Untitled.png)

⇒사이킷런은 RandomForestClassifier 클래스를 통해 랜덤 포레스트 분류를 지원한다.(밑의 예제참고)

[랜덤 포레스트 방식을 이용한 사용자 행동인식 분류 실습](https://www.notion.so/b028526fd58e4b1981f20043dcd3f4d7)

### 부스팅(Booting)

⇒부스팅 알고리즘은 여러 개의 약한 학습기를 순차적으로 학습-예즉하면서 잘못 예측한 데이터에 가중치 부여를 통해 오류를 개선해 나가면서 학습하는 방식이다.

⇒대표적인 부스팅 알고리즘에는 AdaBoost(Adaptive boosting)과 GBM, XGBoost, LightGBM이 있다.

### AdaBoost(Adaptive Boosting)

⇒에이다 부스트는 오류 데이터에 가중치를 부여하면서 부스팅을 수행하는 대표적인 알고리즘이다.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7491e9d2-86da-42fe-b40e-ced3a2cf86d6/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7491e9d2-86da-42fe-b40e-ced3a2cf86d6/Untitled.png)

→옆의 그림에서 Model1은 +와 -를 기준1로 분류했지만 동그라미로 표시된 오류 데이터가 존재한다

→다음 단계에서 이런 오류 데이터에 가중치를 부여하고 다음 분류기가 더 잘 분류할 수 있도록 한다.

→두번째 학습기가 기준2로 +와 -를 구분한다.  하지만 여전히 오류 데이터가 존재한다.

→다시한번 오류 데이터에 가중치를 부여하고 Model3가 기준3로 다시 분류한다.

⇒에이다 부스트는 이렇게 약한 학습기가 순차적으로 오류 값에 대해 가중치를 부여한 예측 결정 기준을 모두 결합해 예측을 수행한다.

### GBM(Gradient Boostiong Machine)

⇒GBM도 에이다 부스트와 유사하지만, 가중치 업데이트를 경사 하강법을 이용하는 것이 큰 차이이다.

⇒GBM은 CART 기반의 다른 알고리즘과 마찬가지로 분류는 물론이고, 회귀도 가능하다.

⇒사이킷런은 GBM기반 분류를 위해서 GradientBoostingClassifier 클래스를 제공한다. (예제는 밑의 링크 참고)

[GBM을 이용한 사용자 행동인식 분류 실습](https://www.notion.so/GBM-7608d42f12234860a4260b471f7725db)

### XGBoost(eXtra Gradient Boost)

⇒XGBoost는 GBM에 기반하고 있지만 GBM의 단점인 느린 수행 시간 및 과적합 규제 부재 등의 문제를 해결했다.

(일반적인 GBM에 비해 빠르다는 것이지, 다른 머신러닝 알고리즘에 비해서 빠르다는 의미는 아니다.)

⇒XGBoost는 tree pruning(가지치기)로 더 이상 긍정 이득이 없는 분할을 가지치기 해서 분할 수를 더 줄이는 추가적인 장점을 가지고있다.

⇒XGBoost는 반복 수행 시마다 내부적으로 학습 데이터 세트와 평가 데이터 세트에 대한 교차 검증을 수행해 최적화된 수행 횟수를 가질 수 있다.

⇒지정된 반복 횟수가 아니라 교차 검증을 통해 평기 데이터세트의 평가 값이 최적화 되면 반복을 중간에 멈출 수 있는 조기 중단 기능이 있다.

[XGBoost를 적용한 위스콘신 유방암 예측 실습](https://www.notion.so/XGBoost-79b38c0164474480827d024a43dab8d9)

### LightGBM

⇒ LightGBM은 XGBoost와 같은 부스팅 계열 알고리즘이다.

⇒LightGBM은 XGBoost보다 학습에 걸리는 시간이 훨씬 적고 메모리 사용량도 상대적으로 적다.

but) LightGBM은 적은 데이터 세트에 적용할 결루 과적합이 발생하기 쉽다. (이반적으로 10,000건 이하의 데이터 세트)

⇒LightGBM은 일반 GBM 계열의 트리 분할 방법과 다르게 리프 중심 트리분할 방식을 사용한다. 기존의 대부분 트리 기반 알고리즘은 트리의 깊이를 효과적으로 줄이기 위해 균형 트리분할 방식을 사용한다. 이렇게 균형잡힌 트리는 오버피팅에 보다 강한 구조를 가진다. 하지만 균형을 맞추기 위한 시간이 걸린다는 단점이 있다.

[LightGBM을 적용한 위스콘신 유방암 예측 실습](https://www.notion.so/LightGBM-c63f54748a394f9d8f78b094dc992267)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fe67b4aa-d347-45a9-8b7e-8a7f2c15abcf/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fe67b4aa-d347-45a9-8b7e-8a7f2c15abcf/Untitled.png)

### 스태킹(Stacking)

⇒스태킹은 개별적인 여러 알고리즘을 서로 결합해 예측 결과를 도출한다는 점에서 앞에 소개한 배깅 및 무스팅과 공통점을 가지고 있다.

But)개별 알고리즘으로 예측한 데이터를 기반으로 다시 예측을 수행한다는 차이점이 있다.

즉, 개별 알고리즘의 예측 결과 데이터세트를 최종적인 메타 데이터 세트로 만들어 별도의 ML알고리즘으로 최종 학습을 수행하고 테스트 데이터를 기반으로 다시 최종 예측을 수행하는 방식이다.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/00e5491a-dc0f-495a-a1fc-8939dc0f68c7/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/00e5491a-dc0f-495a-a1fc-8939dc0f68c7/Untitled.png)

→기본 스태킹 모델

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/51ff51b9-f9b5-4fa0-ad41-901c757d8bdc/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/51ff51b9-f9b5-4fa0-ad41-901c757d8bdc/Untitled.png)

→ CV기반 스태킹 모델
