## 회귀란?

* 통계학적 관점에서 회귀는 여러 개의 독립변수와 한 개의 종속변수 간의 상관관계를 모델링하는 기법이다.

* 머신 러닝 관점에서 보면 독립변수는 피처에 해당되며 종속변수는 결정 값이다.

* 머신러닝 회귀 예측의 핵심을 주어진 피처와 결정 값 데이터 기반에서 학습을 통해 최적의 회귀 계수를 찾아내는 것이다.

* 지도학습은 두 가지 유형으로 나뉘는데, 바로 분류와 회귀이다.

* 분류는 예측 값이 카테고리와 같은 이산형 클래스 값이다.

* 회귀는 예측 값이 연속형 숫자 값이다.

## 단순 선형 회귀를 통한 회귀 이해

* 단순 선형 회귀는 독립변수도 하나, 종속 변수도 하나인 선형 회귀이다.

ex)주택 가격이 주택의 크기로만 결정된다고 하자. 일반적으로 주택의 크기가 크면 가격이 높아지는 경향이 있기 때문에 다음과 같이 가격은 주택 크기에 대해 선형(직선 형태)의 관계로 표현 할 수 있다.

독립변수가 1개인 단순 선형 회귀에서 에측값 Y는 W0+W1*X로 계산 할 수 있다. 독립변수가 1개인 단순 선형 회귀에서는 이 기울기 W1과 절편 W0을 회귀계수로 지칭한다. Y=W0+W1*X와 같은 1차 함수로 모델링 한 회귀 모델은 실제 Y값에서 예측한 Y값을 빼거나 더한값이 오류 값이 되는데 이 오류 값을 최소로 만드는 것이 회귀 모델의 목표이다. 다시말해 이 오류 값 합이 최소가 되는 최적의 회귀 계수를 찾는다는 것이다. 

오류 값은 +나 -값이 될 수 있다. 그래서 전체 데이터의 오류 합을 구하기 위해 단순히 더했다가는 뜻하지 않게 오류의 합이 크게 줄어들 수 있다. 따라서 보통 오류 합을 계산할때는 절댓값을 취해서 더하거나, 오류 값의 제곱을 구해서 더하는 방식(RSS)을 취한다. 즉 Error^2=RSS이다. 회귀에서 이 RSS는 비용(cost)이며, w변수(회귀 계수)로 구성되는 RSS를 비용함수라고 한다.

![선형회귀 이미지](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/5ceb2da1-048e-4fed-9683-859ce009964a/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210613%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210613T135752Z&X-Amz-Expires=86400&X-Amz-Signature=af8308fefcd94f5572af7b29202b4f1d72f745b17f7fd03937baee3664e56b3f&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

그렇다면 어떻게 비용함수가 최소가 되는 W파라미터를 구할 수 있을까? 정답은 경사하강법이다. 

![경사하강법이미지](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/242cf8fc-995c-46b3-95cb-2c0793a2865e/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210613%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210613T135812Z&X-Amz-Expires=86400&X-Amz-Signature=d1ad1e95369af9fb1e4c57b914b87598a647a0770298c8a0637224a68b477257&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)


예를 들어 비용함수가 다음 그림과 같은 포물선 형태의 2차 함수하면 경사 하강법은 최초 w에서부터 미분을 적용한 뒤 이 미분 값이 계속 감소하는 방향으로 순차적으로 w를 업데이트한다. 마침내 더 이상 미분된 1차 함수의 기울기가 감소하지 않는 지점을 비용 함수가 최소인 지점으로 간주하고 그때의 w를 반환한다.

[파이썬으로 구현한 경사 하강법](https://github.com/JIWON0520/TIL/blob/main/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/%EC%8B%A4%EC%8A%B5/%ED%9A%8C%EA%B7%80/%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9C%BC%EB%A1%9C%20%EA%B5%AC%ED%98%84%ED%95%9C%20%EA%B2%BD%EC%82%AC%20%ED%95%98%EA%B0%95%EB%B2%95.md)

## LinearRrgression 클래스

LinearRegression 클래스는 예측값과 실제 값의 RSS를 최소화해 OLS추정방식으로 구현한 클래스이다. 이 클래스는 fit() 메서드로 X,y 배열을 입력 받으면 회귀 계수 (Coefficients)인 W를  coef_ 속성에 저장한다.

회귀희 평가를 위한 지표는 실제 값과 회귀 예측값의 차이 값을 기반으로 한 지표가 중심이다. 

실제값과 예측값의 차이를 그냥 더하면 +와 -가 섞여서 오류가 상쇄된다. 

이때문에 오류의 절댓값의 평균이나 제곱, 또는 제곱한 뒤 다시 루트를 씌운 평균값을 구한다.

***회귀 평가 지표***

![MAE](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/b96a9d8a-2bed-4094-8cee-ac0167a05d5b/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210621%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210621T135140Z&X-Amz-Expires=86400&X-Amz-Signature=dde4c7152e0d47be09f58b18a52f1400864ba65a70944d95c7d0fe07f8856939&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

![MSE](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/d9eaf933-1083-4f3c-b79f-3f175aeb6dfe/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210621%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210621T135205Z&X-Amz-Expires=86400&X-Amz-Signature=157b40bfe4c1a583fddae3357364148406d72eb183c27dbc8c8e7b89fb1b95c3&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

![RMSE](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/9c21f277-3898-4a24-a940-dba8c151c53d/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210621%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210621T135223Z&X-Amz-Expires=86400&X-Amz-Signature=1931dd4284ff509b8b14a56b0af0d3aa4a2fc24dc57f601d46bca0ed0973f52f&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

## 다항 회귀

지금까지 설명한 회귀는 독립변수와 종속변수의 관계가 일차 방정식 형태로 표현된 회귀였다. 하지만 세상의 모든 관계를 직선으로만 표현할 수는 없다. 회귀가 독립변수의 단항식이아닌 2차, 3차 방정식과 같은 다항식으로 표현되는 것을 **다항 회귀**라고 한다. 

한 가지 주의할 점은 다항회귀는 선형 회귀라는 것이다. 회귀에서 선형/비선형 회귀를 나누는 기준은 회귀 계수가 선형/비선형인지에 따른 것이지 독립변수의 선형/비선형 여부와는 무관하다.

![다항회귀](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/c87be989-b69b-40be-a99e-8b536861164d/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210621%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210621T135243Z&X-Amz-Expires=86400&X-Amz-Signature=a0bdbfc52fea1efa567672ca54324310ea13bdb57d28b5135e8288d59359cf64&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

사진에 주어진 데이터는 단항 회귀보다는 다항 회귀가 더 효과적이다.

아쉽게도 사이킷런은 다항 회귀를 위한 클래스를 명시적으로 제공하지 않는다. 대신 다항 회귀 역시 선형 회귀이기 때문에 비선형 함수를 선형 모델에 적용시키는 방법을 사용해 구현한다. 이를 위해 사이킷런은 PolynomialFeatures 클래스를 통해 피처를 다항식 피처로 변환한다. 

### 다항 회귀를 이용한 과소적합 및 과적합 이해

다항 회귀는 피처의 직선적 관계가 아닌 복잡한 다항 관계를 모델링할 수 있다. 다항식의 차수가 높아질수록 매우 복잡한 피처간의 관계까지 모델링 가능하다. 하지만 다항 회귀의 차수를 높일 수록 학습 데이터에만 너무 맞춘 학습이 이루어져서 정작 테스트 데이터 환경에서는 오히려 예측 정확도가 떨어진다. **즉, 차수가 높아질수록 과적합의 문제가 크게 발생한다.**

![과소적합](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/0e99387b-f159-448a-9fd4-c4d923627877/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210621%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210621T135506Z&X-Amz-Expires=86400&X-Amz-Signature=30678290b4fdfcc2ac45145fcf8097e82aef3061a3a502b5ed2aaed29b72979b&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

맨 왼쪽의 Degree 1예측 곡선은 단순항 직선으로서 선형 회귀와 똑같다. 실제 데이터 세트인 코사인 데이터 세트를 직선으로 예측하기에는 너무 단순해보인다. 예측 곡선이 학습 데이터의 패턴을 제대로 반영하지 못하고 있는 과소 적합 모델이 되었다.

가운데 Degree 4 예측 곡선은 실제 데이터 세트와 유사한 모습니다. 변동하는 잡음까지 예측하지는 못했지만, 학습 데이터 세트를 비교적 잘 반영해 코사인 곡선 기반으로 테스트 데이터를 잘 예측한 곡선을 가진 모델이 되었다.

맨 오른쪽 Degree 15 예측 곡선을 보면 데이터 세트의 변동 잡음 값까지 지나치게 반영한 결과, 예측 곡선이 학습 데이터 세트만 정확히 예측하고, 테스트 값의 실제 곡선과는 완전히 다른 형태의 예측 곡선이 만들어졌다. 결과 적으로 학습 데이터에 너무 충실하게 맞춘 과적합이 심한 모델이 되었다.

결국 좋은 예측 모델은 Degree1 과 같이 학습 데이터의 패턴을 지나치게 단순화한 과소적합 모델도 아니고 Degree 15와 같이 모든 학습 데이터의 패던을 하나하나 감안한 지나치게 복잡한 과적합 모델도 아닌, 학습 데이터의 패턴을 잘 반영하면서도 복잡하지 않는 균형 잡힌 모델을 의미한다.

[PolynomialFeature 클래스를 이용한 다항 회귀 구현](https://github.com/JIWON0520/TIL/blob/main/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/%EC%8B%A4%EC%8A%B5/%ED%9A%8C%EA%B7%80/PolynomialFeature%20%ED%81%B4%EB%9E%98%EC%8A%A4%EB%A5%BC%20%EC%9D%B4%EC%9A%A9%ED%95%9C%20%EB%8B%A4%ED%95%AD%20%ED%9A%8C%EA%B7%80%20%EA%B5%AC%ED%98%84.md)

### 편향-분산 트레이오프

편향-분산 트레이오프는 머신러닝이 극복해야 할 가장 중요한 이슈 중의 하나이다. 앞의 Degree1과 같은 모델은 매우 단순화된 모델로서 지나치게 한 방향성으로 치우친 경향이 있다. 이런 모델은 고편향성을 가졌다고 표현한다. 반대로 Degree15와 같은 모델은 학습 데이터 하나하나의 특성을 반영하면서 매우 복잡한 모델이 되었고 지나치게 높은 변동성을 가지게 되었다. 이런 모델을 고분산성을 가졌다고 표현한다.

![편향-분산](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/b1d8d1ee-9537-4cd1-93ab-2e20dc7645e9/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210621%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210621T135530Z&X-Amz-Expires=86400&X-Amz-Signature=3ad992153c397ba051f046633a1a56def9d6985a377a129bc6edc339d0778534&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

그림 상단 왼쪽의 저편향/저분산은 예측 결과가 실제 결과에 매우 잘 근접하면서도 예측 변동이 크지 않고 큭정 부분에 집중돼 있는 아주 뛰어난 성능을 보여준다. 

상단 오른쪽의 저편향/고분산은 예측 결과가 실제 결과에 비교적 근접하지만 예측 결과가 실제 결과를 중심으로 꽤 넓은 부분에 분포돼있다. 

하단 왼쪽의 고편향/저분산은 정확한 결과에서 벗어나면서도 예측이 특정 부분에 집중되어있다. 

마지막으로 하단 오른똑의 고편향/고분산은 정확한 예측 결과를 벗어나면서도 넓은 부분에 분포돼있다.

일반적으로 편향과 분산은 한 쪽이 높으면 한 쪽이 낮아지는 경향이 있다. 

즉 편향이 높으면 분산은 낮아지고 반대로 분산이 높으면 편향이 낮아진다.

높은 편향/낮은 분산은 과소 적합에서 나타기 쉬우며, 낮은 편향/높은 분산에서 과적합이되기 쉽다. 

편향과 분산이 서로 트레이드오프를 이루면서 오류 cost값이 최대로 낮아지는 모델을 구축하는 것이 가장 효율적인 머신러닝 예측 모델을 만드는 방법이다.

## 규제 선형 모델 - 릿지, 라쏘, 엘라스틱넷

이전까지 선형 모델의 비용함수는 RSS를 최소화하는 , 즉 실제 값과 예측값의 차이를 최소화하는 것만 고려했다. 그러다 보니 학습 데이터에 지나치게 맞추게 되고, 회귀 계수가 쉽게 커졌다. 이럴 경우 변동성이 오히려 심해져서 테스트 데이터 세트에서는 예측 성능이 저하되기 쉽다.  이를 반영해 비용 함수는 학습 데이터의 진차 오류 값을 최소로하는 RSS최소화 방법과 과적합을 방지하기 위해 회귀 계수 값이 커지지 않도록 하는 방법이 서로 균형을 이뤄야 한다.

이렇게 회귀 게수의 크기를 제어해 과적합을 개선하려면 비용함수의 목표가 다음과 같아야 한다.

![비용함수](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/77ac2af5-d55e-4743-8fb6-db8dbcaee896/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210621%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210621T135553Z&X-Amz-Expires=86400&X-Amz-Signature=8a129b8c567a616699903f8d4098a6aff3ea82a73b60afa13871dbfc4c4c1490&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

여기서 alpha는 학습 데이터 적합 정도와 회귀 계수 값의 크기 제어를 수행하는 튜닝 파라미터이다. 비용 함수의 목표가 ***Min*(*RSS*(*W*)+*alpha*∗∣∣*W*∣∣)**을 최소화 하는 W벡터를 찾는 것일 때 alpha가 어떤 역할을 하는지 살펴보자.

alpha가 0또는 매우 작은 값이라면 비용 함수 식은 기존과 동일한 ***Min*(*RSS*(*W*)+0)** 이 될 것이다. 반면에 alpha거 무한대라면 비용함수 식은 **RSS(W)**에 비해 alpha*||W||값이 너무 커지게 되므로 W값을 0(또는 매우 작게)으로 만들어야 Cost가 최소화되는 비용 함수 목표를 달성할 수 있다.

즉, alpha 값을 크게하면 비용 함수는 회귀 계수 **W**의 값을 작게 해 과적합을 개선할 수 있으며, alpha값을 작게 하면 회귀 계수 W의 값이 커져도 어느 정도 상쇄가 가능하므로 학습 데이터의 적합을 더 개선 할 수 있다.

이처럼 비용 함수에 alpha값으로 패널티를 부여해 회귀 계수 값의 크기를 감소시켜 과적합을 개선하는 방식을 규제라고한다. 규제는 크게 L2방식과 L1방식으로 구분된다. L2규제는 위에서 설명한 바와 같이 **alpha*||W||^2**와 같이 W의 제곱에 대해 패널티를 부여하는 방식을 말한다. 이 규제를 적용한 회귀를 릿지 회귀라고 한다. 라쏘 회귀는 L1규제를 적용한 회귀이다. L1규제는 **alpha*||W||**와 같이 W의 절댓값에 대해 패널티를 부여한다. L1규제를 적용하면 영향력이 크지 않은 회귀 계수 값을 0으로 변환한다.

[릿지 회귀를 이용한 보스턴 주택가격 예측](https://github.com/JIWON0520/TIL/blob/main/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/%EC%8B%A4%EC%8A%B5/%ED%9A%8C%EA%B7%80/Ridge%20Regression%EC%9C%BC%EB%A1%9C%20%EB%B3%B4%EC%8A%A4%ED%84%B4%20%EC%A3%BC%ED%83%9D%EA%B0%80%EA%B2%A9%20%EC%98%88%EC%B8%A1%ED%95%98%EA%B8%B0.md)

[라쏘 회귀를 이용해 보스턴 주택 가격 예측하기](https://github.com/JIWON0520/TIL/blob/main/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/%EC%8B%A4%EC%8A%B5/%ED%9A%8C%EA%B7%80/Lasso%20Regression%EC%9D%84%20%EC%9D%B4%EC%9A%A9%ED%95%9C%20%EB%B3%B4%EC%8A%A4%ED%84%B4%20%EC%A3%BC%ED%83%9D%20%EA%B0%80%EA%B2%A9%20%EC%98%88%EC%B8%A1.md)

### 엘라스틱넷 회귀

엘라스틱넷 회귀는  L2규제와 L1규제를 결합한 회귀이다. 라쏘 회귀는 서로 상관관계가 높은 피처들의 경우에 이들 중에서 중요 피처만 셀렉션하고 다른 피처들은 모두 회귀 계수를 0으로 만드는 성향이 강하다. 엘라스틱넷 회귀는 이를 완화하기 위해 L2 규제를 라쏘 회귀에 추가한 것이다. 반대로 엘라스틱넷 회귀의 단점은 L1과 L2규제가 결합된 규제로 인해 수행시간이 상대적으로 오래 걸린다는 것이다.

[엘라스틱넷 회귀를 이용한 보스턴 주택가격 예측 실습](https://github.com/JIWON0520/TIL/blob/main/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/%EC%8B%A4%EC%8A%B5/%ED%9A%8C%EA%B7%80/ElasticNet%20Regression%EC%9D%84%20%EC%9D%B4%EC%9A%A9%ED%95%9C%20%EB%B3%B4%EC%8A%A4%ED%84%B4%20%EC%A3%BC%ED%83%9D%EA%B0%80%EA%B2%A9%20%EC%98%88%EC%B8%A1%20%EC%8B%A4%EC%8A%B5.md)

## 선형 회귀 모델을 위한 데이터 변환

선형 회귀 모델과 같은 선형 모델은 일반적으로 피처와 타깃값 간에 선형의 관계가 있다고 가정한다. 또한 선형 회귀 모델은 피처값과 타깃값의 분포가 정규 분포 형태를 매우 선호한다. 따라서 선형 회귀 모델을 적용하기 전에 먼저 데이터에 대한 스케일링/정규화 작업을 수행하는 것이 일반적이다.

피처 데이처 세트/타깃 데이터세트에 대한 스케일링/정규화 방법

1.StandardScaler 클래스를 이용해 평균이 0, 분산이 1 인 표준 정규 분포를 가진 데이터 세트로 변환하거나 MinMaxScaler 클래스를 이용해 최솟값이 0이고 최댓값이 1인 값으로 정규화를 수행한다.

2.스케일링/정규화를 수행한 데이터 세트에 다시 다항 특성을 적용하요 변환하는 방법이다.

3.원래 값에 log함수를 적용하면 보다 정규 분포에 가까운 형태로 값이 분포된다. (실제로 1,2번보다 많이 사용되는 변환 방법)

타깃값의 경우는 일반적으로 로그 변환을 적용한다. 결정 값을 정규 분포나 다른 정규값으로 변환하면 변환된 값을 다시 원본 타깃값으로 복원하기 어려울 수 있기 때문이다.

## 로지스틱 회귀

로지스틱 회귀는 선형 회귀 방식을 분류에 적용한 알고리즘이다. 로지스틱 회귀가 선형 회귀와 다른 점은 학습을 통해 선형 함수의 회귀 최적선을 찾는 것이 아니라 시그모이드 함수 최적선을 찾고 이 시그모이드 함수의 반환 값을 확률로 간주해 확률에 따라 분류를 결정한다는 것이다.

시그모이드 함수의 정의는 다음과 같다.

![시그모이드 함수](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/4e12eccb-18bc-43b9-b37f-82decc1fba44/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210622%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210622T124313Z&X-Amz-Expires=86400&X-Amz-Signature=1d6d65812b5b89cc8c5c4f3f13f31eec13c9d68d4c9f23874a742660954538e1&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

시그모이드 함수는 x값이 +,-로 아무리 커지거나 작아져고 y값은 항상 0과 1사이 값을 반환한다.

다음 사진과 같이 왼쪽의 선형 회귀는 0과 1일 제대로 분류하지 못하고 있지만 오른쪽의 시그모이드 함수를 이용하면 좀 더 정확하게 0과 1에 대해 분류를 할 수 있음을 알 수 있다. 로지스틱 회귀는 이처럼 선형 회귀 방식을 기반으로 하되 시그모이드 함수를 이용해 분류를 수행하는 회귀이다. 

[로지스틱 회귀를 이용한 위스콘신 유방암 여부 판단실습](https://github.com/JIWON0520/TIL/blob/main/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/%EC%8B%A4%EC%8A%B5/%ED%9A%8C%EA%B7%80/Logistic%20Regression%EC%9D%84%20%EC%9D%B4%EC%9A%A9%ED%95%9C%20%EC%9C%84%EC%8A%A4%EC%BD%98%EC%8B%A0%20%EC%9C%A0%EB%B0%A9%EC%95%94%20%EC%97%AC%EB%B6%80%20%ED%8C%90%EB%8B%A8%EC%8B%A4%EC%8A%B5.md)

![시그모이드 그래프](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/bc509c31-0c98-4dba-95f2-524cc660c50e/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210622%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210622T124343Z&X-Amz-Expires=86400&X-Amz-Signature=58a3eb9c2dba206202839997d4c193cf4b0db62c913b1c9ca711be58df2ced55&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

## 회귀 트리

회귀 트리는 회귀 함수를 기반으로 하지 않고 결정 트리와 같이 트리를 기반으로 하는 회귀 방식이다. 즉, 회귀를 위한 트리를 생성하고 이를 기반으로 회귀 예측을 하는 것이다.

옆의 그림처럼 X 피처 데이터 세트와 결정값 Y가 2차원 평면상이 있다고 가정하자.

이 데이터 세트의 X 피처를 결정 트리 기반으로 분할 하면 X값의 균일도를 반영한 지니 계수에 따라 그림과 같이 분할할 수 있다.

루트 노드를 Split 0 기준으로 분할하고 이렇게 분할된 규칙 노드에서 다시 Split1 과 Split2로 분할할 수 있다.

리프 노드 생성 기준에 부합하는 트리 분할이 완료 됐다면, 리프 노드에 소속된 데이터 값의 평균값을 구해서 최종적으로 리프 노드에 결정 값으로 할당한다.

모든 트리 기반의 알고리즘(결정 트리, 랜덤 포레스트, GBM 등)은 분류뿐만 아니라 회귀에도 적용가능하다.

![회귀트리 이미지](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/3f1e4119-00df-4ef8-a158-b6fa4b17abb5/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210622%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210622T124443Z&X-Amz-Expires=86400&X-Amz-Signature=8cfcdd68eecfdd01e46eabd844271c9b0a4f35595b5c5ac5cf1915022d60eee9&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)

[회귀 트리를 이용한 보스턴 주택가격 예측](https://www.notion.so/967ceca2832e4ac5a4f30692386e3b84)

