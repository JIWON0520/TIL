## lecture 2 - Image Classification pipeline

![사진1](https://user-images.githubusercontent.com/77263283/131355084-9fbd7643-09e4-4758-9500-9e59a5797880.png)


컴퓨터가 이미지를 보고 사물을 인식하는 것은 매우 어려운 일이다.

우리는 의미론적으로 저 사진을 보면 '고양이'라고 인식하지만 컴퓨터는 단지  큰 그리드의 숫자로 인식하기 때문이다. 그리고 카메라의 방향이 바뀌거나 빛의 위치, 고양이의 자세 등이 바뀌면 우리는 고양이를 잘 인식하지만 컴퓨터에겐 힘든 일이다. 사진의 사소한 것만 바뀌어도 개개의 픽셀은 바뀔것이기 때문이다. 

![사진2](https://user-images.githubusercontent.com/77263283/131355266-fc89407a-2919-4101-bd0b-64d005235c8c.png)

우리는 사물의 경계선이 사물을 판별하는데 중요한것을 알수 있듯이, 컴퓨터도 이런 경계선을 계산해서 사물을 판별할 수 있지 않을까?하는 생각을 해볼 수 있다. 하지만 이러한 생각은 좋은 방법이 아니다.  왜냐하면 고양이가 아닌 다른 사물을 분류할때는 다시 처음과정부터 반복해야하기 때문이다.

![사진3](https://user-images.githubusercontent.com/77263283/131355325-6a391829-c86d-4516-8828-623233ad0fe1.png)

우리는 세상의 모든 사물에게 적용할 수 있는 사물인식 알고리즘을 생각해 내야하는데 이것을 가능하게 해주는 개념은 '데이터 기반 접근'이다.

데이터 기반 접근은 방대한 데이터를 수집하고 이 데이터들로 머신러닝 Classifier에게 학습시킨뒤 어떠한 방법으로 이 특징들을 요약하여 테스트셋에 적용하여 결과를 예측하는 방법을 말한다.

이 방법에는 두가지 function이 필요한데, 하나는 학습 함수이다. 학습에서는 이미지와 그 카테고리 분류값을 input으로 넣고 classifier모델을 output으로 하는 function이다. 그리고 다른 function은 테스트 function이다. 테스트 function에서는 학습된 classifier 모델을 input으로 넣고 이미지에대한 예측 값을 output으로 하는 function이다.

여기서 우리가 생각할 수 있는 가장 단순한 알고리즘은 '최근접 이웃'알고리즘이다. 이 알고리즘은 학습 단계에서 아무것도 하지 않는다. 단지 학습 데이터샛을 '기억'만 한다. 테스트 단계에서는 새로운 이미지를 넣어 기존의 이미지와 비교하여 가장 유사한 이미지를 찾고 이를 바탕으로 이미지의 카테고리 값을 예측해준다.

여기서 한가지 짚고 넘어갈 점은 classifier 모델은 어떻게 이미지들을 비교하는 걸까?

이때 사용되는 metrics에는 '맨해튼 거리'라고 불리는 L1과, '유클리디안 거리'라 불리는 L2가 있다.

![사진4](https://user-images.githubusercontent.com/77263283/131355362-25a0722c-9df5-4380-9ebe-e1b47b0f977e.png)

L1은 단지 두 이미지의 같은 위치에있는 픽셀값을 빼고 그 값을 절댓값을 취해 모두 더한 값을 뜻한다.

L2는 픽셀값 차에 절댓값을 취한뒤, 제곱하여 루트를 씌운 값이다.

거리 metrics를 어떻게 선택하냐에 따라 다른 결과가 도출되기 때문에 거리 metrics를 선택하는 것은 중요한 일이다. L1은 좌표 시스템에 영향을 크게 받기 때문에 개개의 벡터가 의미를 가지고있다면 L1 distance를 사용하는 것이 효율적인 반면, L2 distance는 좌표 시스템에 별로 영향을 받지 않기 때문에 개별 벡터가 일반적인 벡터거나 의미를 갖지 않는다면 L2 distance가 더 잘어울린다.

최근접 이웃 알고리즘에서 중요한 또하나는 결과를 도출해내는데 비교하는 이웃의 수이다. 이것을 K라고 하는데 주변에 가까운 K개의 이웃에서 다수결에 따라 레이블을 결정하는 것이다.

![사진5](https://user-images.githubusercontent.com/77263283/131355502-9c35afa3-4770-4819-a929-ad1274a8de9c.png)

위 사진에서 K=1일때, 녹색 영역중앙에 노란색 영역이 있는데, 이는 좋지 않다. 또한 빨간색 영역과 초록색 영역이 하나의 데이터 때문에 파란색 영역에 삐죽하게 침범하고 있다. 이 데이터는 노이즈나 잘못된 데이터일수 있다. K=3,K=5일때를 보면 녹색 영역 중앙의 노란 점이 녹색 영역이 된것을 볼수 있는데 이것은 하나의 이웃만 고려하지 않고 K개의 이웃을 고려하여 더 많은 수의 레이블로 결정되기 때문이다. 또한 파란색 영역에 침범한 빨강,초록 영역의 경계도 완만해진것을 볼 수 있다.

위에서 말한 distance metrics와 K는 '하이퍼파라미터'라고 불린다.

그렇다면 우리는 어떻게 하이퍼파라미터를 선택할까?

단순한 방법은 여러 하이퍼파라미터를 시도해보고 가장 좋은 값을 찾는 것이다. 하이퍼파라미터는 데이터에 의존적이기 때문에 데이터마다 적합한 값이 다르기 때문이다.

이런 단순한 방법에서 우리는 하이퍼파라미터 튜닝을 어떻게 적용하고 어떻게 평가할까?

![사진6](https://user-images.githubusercontent.com/77263283/131355525-2b877187-4ca4-4d80-9a95-9f749a1e7f98.png)

가장 먼저 생각할 수 있는 단순한 방법은 우리의 전체 데이터셋에 하이퍼파라미터를 적용하는 것이다. 이 방법은 우리의 데이터셋에게는 높은 정확도와 높은 성능을 낼 수 있지만 새로운 데이터에 대해서는 좋은 성능을 내지 못한다. 머신 러닝의 궁극적인 목표는 학습 데이터에 맞춰지는 것이아니라 이전에 보지 못했던 데이터에 대해 좋은 성능을 내는 것이다.

다른 아이디어는 우리의 데이터 셋을 학습 데이터와 테스트 데이터로 나누어 하이퍼파라미터의 튜닝을 진행하는 것이다. 학습 데이터에 여러 하이퍼파라미터를 적용하고 테스트데이터로 이 classifier model을 테스트하여 가장 성능이 좋은 하이퍼파라미터를 찾는것이다. 하지만 이 방법 또한 우리의 테스트 데이터에 대해서는 잘 작동할지라도 새로운 데이터에 대해서는 좋은 성능을 내지 못한다. 테스트 데이터 세트에 잘 동작하는 하이퍼파라미터를 선택한다하더라도 우리의 테스트데이터셋은 더이상 새로운 데이터의 특징을 대표하지 못하기 때문이다.

세번째 방법은 우리의 데이터 세트를 학습 데이터세트, 검증 데이터세트, 테스트 데이터 세트처럼 세 세트로 나누는 것이다.  다른 하이퍼파라미터로 학습된 classifier를 검증 데이터세트에서 검증하고 가장 좋은 하이퍼파라미터를 선택해 테스트데이터세트를 사용해 단 한번 테스트 해보는 것이다. 검증 데이터세트와 테스트 데이터세트를 엄격하게 분리하는 것은 매우 중요한 일이다. **테스트 데이터는 오직 단 한번만 사용되어야 한다.** 

![사진7](https://user-images.githubusercontent.com/77263283/131355548-ff7f4b42-369a-480c-a62a-2ce0f31d2315.png)

세번째 방법에서 더 나아가 'cross validation'방법이 있다. 이는 학습데이터를 여러 fold로 나누는 것이다. 위의 예제는 학습 데이터를 5개의 fold로 나누어 4개의 fold를 학습에 사용하고 나머지 하나의 fold를 검증에 사용한다. 이렇게 여러 하이퍼파라미터 세트를 검증하고 가장 좋은 성능을 내는 하이퍼파라미터를 채택해 마지막으로 테스트데이터에 적용시킨다. 이 방법은 계산이 많이 필요한 딥러닝에서는 많이 쓰이지 않는다.

 

![사진8](https://user-images.githubusercontent.com/77263283/131355576-15933574-cb84-4c16-8c70-1d2e7425c027.png)

K-최근접 이웃 알고리즘은 이미지 classification에서 쓰이지 않는다. 왜냐하면 이 알고리즘을 테스트 데이터 세트에 적용시켰을때 많은 시간이 걸리고, L1 distance나 L2 distance가 이미지에서 distance metrics로 적합하지 않기 때문이다. 예를들어 위의 원본 사진에 세가지 변형을 준 사진들의 L2 distance를 계산해 보면 모두 같은 값이 나온다.

![사진9](https://user-images.githubusercontent.com/77263283/131355600-c77738fd-8d86-4952-831c-8b7cb5e347d1.png)

다른 문제점은 K-최근접 이웃은 '차원의 저주'라 불리는 문제점이 있다. K-최근접 이웃 알고리즘은 공간위의 파티션을 나누기 위해 학습 데이터의 점들을 사용한다. 그러기 위해서는 우리는 공간을 충분히 덮을만한 학습 데이터 세트의 양이 필요하다. 하지만 공간의 차원이 늘어날수록 필요한 학습 데이터 세트는 급수적으로 늘어나게 된다.   

위의 그림을 보면, 1차원 평면에서 공간을 조밀하게 덮기 위해 단지 4개의 데이터만 필요하다. 하지만 2차원에서는 4^2인 16개의 데이터가 필요하고, 3차원에서는 4^3인 64개의 데이터가 필요하다. 이렇게 K-최근접 이웃기반 알고리즘이 잘 작동하기 위해서는 조밀한 학습 데이터세트를 가져야 한다.

이런 문제점들로 K-최근접 이웃기만 알고리즘은 이미지 데이터에서 많이 쓰이지 않는다. 

딥러닝의 가장 기초적인 linear classification을 살펴보자.

딥러닝에서 neural network은 레고 블럭으로 비유될수 있다. 레고를 구성하는 각각의 컴포넌트들중에서 가장 기본적인 것이 linear classifier이다. 

![사진10](https://user-images.githubusercontent.com/77263283/131355654-34e7e904-0d8b-4ad4-8b33-49240f797940.png)

위에서 말한 K-최근접이웃 기반 알고리즘과 달리 input으로는 고양의 이미지를 X로 사용하며, 파라미터(가중치)는 'W'라고 한다.  그리고 이 두 데이터 X와 W를 사용하여 10개의 카테고리에 대응하는 스코어를 출력하는 함수가 있다. 고양이에 대한 높은 점수는 입력값인 X가 고양이 카테고리에 부합할 확률이 높다는 뜻이다. 

K-최근접 이웃기반 알고리즘에는 파라미터가 없었고 단지 학습 학습 세트 전체를 기억할 뿐이었다. 그 학습 데이터 세트는 테스트에서 쓰인다. 반면에 linear classifier과 같은 parametric적 접근에서는 학습 데이터세트를 요약해서 파라미터인 W에 모아준다. 그래서 test time에서는 더이상 학습 데이터세트를 사용할 필요가 없고 단지 파라미터 W만 필요하다. 이러한 방법들은 model들이 작은 기긱에서 돌아갈수 있게 한다.

딥러닝에서 가장 중요한 것은 데이터와 파라미터를 알맞게 조합하여 결과를 출력하는 함수 F를 생각해 내는 것이다. 보통 간단한 방법으로 단지 입력 X와 파라미터W를 곱하는 것이다.

![사진11](https://user-images.githubusercontent.com/77263283/131355676-523ca225-f0fa-4bab-8e15-0d8d761e86c1.png)

위의 사진처럼 함수 F와 입력값 x, 파라미터 W는 F(x,W)=Wx로 표현될수 있다. 입력 이미지 값인 32x32x3의 벡터를 열벡터로 쭉 늘인다음 10개 카테고리의 스코어를 갖기 위해 10x3072 형태의 W와 곱한다. 그럼 10x1 형태의 열벡터가 함수의 값으로 출력되는데 각각의 행은 각 카테고리의 스코어를 뜻한다.

![사진12](https://user-images.githubusercontent.com/77263283/131355706-aa78b1aa-eec6-4991-be5a-940a48e4e576.png)

가끔 bias값을 더해주는데 이 b값은 10개의 상수벡터이고 학습 데이터와는 독립적인 값이다. 이 bias값은 단지 데이터에 독립적인 우선권을 준다. 만약 데이터세트에서 강아지보다 고양이가 많을때, 즉 데이터가 불균형할때 고양이에게 대응되는 bias값이 더 커질것이다.

![사진13](https://user-images.githubusercontent.com/77263283/131355738-f9476876-bd81-432d-89d2-1e765c443cfe.png)

위의 예제는 고양이 이미지를 세개의 카테고리중 하나로 분류하는 예제이다. 4개의 픽셀로 된 input이미지를 4x1의 형태로 늘여 3x4형태의 W와 곱하고 bias값을 더해서 3x1의 결과 벡터를 출력한다. 우리는 여기서 linear classifier가 template matching 접근이라는 것을 알 수 있다. W의 각 열은 각 카테고리의 template과 매칭된다. 여기서 이미지와 W의 내적은 클래스의 템플릿과 이미지의 픽셀의 유사도를 계산하는 것과 비슷하다. 

이런 템플릿 매칭 관점에서 linear classifier의 매트릭스 W에서 각 행을 시각화 하면 이미지의 템플릿을 볼 수 있다. 이 시각화한 이미지를 통해 우리는 linear classifier가 이미지를 이해하기 위해 어떻게 동작하고 있는지를 알 수 있다.

![사진14](https://user-images.githubusercontent.com/77263283/131355899-8d302534-3b89-431c-829f-4323ddbf4339.png)

위의 이미지는 실제로 matric를 시각화해서 나타낸 것이다. 왼쪽의 템플릿은 비행기 클레스에 대응하는 템플릿이다. 이 이미지의 배경색은 파란색이고 가운데에 파란 얼룩처럼 형태가 보인다. 이런 템플릿을 이용해서 classifier는 비행기를 분류한다. 여기서 linear classifier의 문제점은 단지 하나의 템플릿만을 비교하여 분류를 진행한다는 것이다. 그래서 카테고리마다 다양한 특징들이 있더라고 각 클래스들의 다양성을 평균내어 카테고리를 인식하는데 하나의 템플릿을 사용한다.

![사진15](https://user-images.githubusercontent.com/77263283/131423974-29191105-2487-4020-9ab3-de3584070488.png)

이미지를 공간상의 점으로 볼때 linear classifier은 하나의 카테고리를 나머지 카테고리랑 분리시키기위해 구분선을 그린다. 이런 고차원 공간의 관점에서 보면 linear classifier의 문제점을 알 수 있다.

![사진16](https://user-images.githubusercontent.com/77263283/131423997-f761509c-4bc6-4bf9-9c13-ed36b188999f.png)

위의 예제들을 보면 선하나로 두개의 카테고리를 분류하는 것이 불가능한것을 볼 수 있다.

linear classifier에는 이러한 문제점이 있지만 이 알고리즘은 매우 단순한 알고리즘이고 이애하기 쉬운 알고리즘이다.
