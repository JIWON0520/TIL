## lecture 2 - Image Classification pipeline

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1f85fe66-3ab1-4613-8288-640d0a1143c5/Untitled.png)

컴퓨터가 이미지를 보고 사물을 인식하는 것은 매우 어려운 일이다.

우리는 의미론적으로 저 사진을 보면 '고양이'라고 인식하지만 컴퓨터는 단지  큰 그리드의 숫자로 인식하기 때문이다. 그리고 카메라의 방향이 바뀌거나 빛의 위치, 고양이의 자세 등이 바뀌면 우리는 고양이를 잘 인식하지만 컴퓨터에겐 힘든 일이다. 사진의 사소한 것만 바뀌어도 개개의 픽셀은 바뀔것이기 때문이다. 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b1bef15e-e3e6-48d0-8a3e-2ccbdb1a4f2e/Untitled.png)

우리는 사물의 경계선이 사물을 판별하는데 중요한것을 알수 있듯이, 컴퓨터도 이런 경계선을 계산해서 사물을 판별할 수 있지 않을까?하는 생각을 해볼 수 있다. 하지만 이러한 생각은 좋은 방법이 아니다.  왜냐하면 고양이가 아닌 다른 사물을 분류할때는 다시 처음과정부터 반복해야하기 때문이다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/49046b52-a4c3-4589-ba51-30d2d9d8edac/Untitled.png)

우리는 세상의 모든 사물에게 적용할 수 있는 사물인식 알고리즘을 생각해 내야하는데 이것을 가능하게 해주는 개념은 '데이터 기반 접근'이다.

데이터 기반 접근은 방대한 데이터를 수집하고 이 데이터들로 머신러닝 Classifier에게 학습시킨뒤 어떠한 방법으로 이 특징들을 요약하여 테스트셋에 적용하여 결과를 예측하는 방법을 말한다.

이 방법에는 두가지 function이 필요한데, 하나는 학습 함수이다. 학습에서는 이미지와 그 카테고리 분류값을 input으로 넣고 classifier모델을 output으로 하는 function이다. 그리고 다른 function은 테스트 function이다. 테스트 function에서는 학습된 classifier 모델을 input으로 넣고 이미지에대한 예측 값을 output으로 하는 function이다.

여기서 우리가 생각할 수 있는 가장 단순한 알고리즘은 '최근접 이웃'알고리즘이다. 이 알고리즘은 학습 단계에서 아무것도 하지 않는다. 단지 학습 데이터샛을 '기억'만 한다. 테스트 단계에서는 새로운 이미지를 넣어 기존의 이미지와 비교하여 가장 유사한 이미지를 찾고 이를 바탕으로 이미지의 카테고리 값을 예측해준다.

여기서 한가지 짚고 넘어갈 점은 classifier 모델은 어떻게 이미지들을 비교하는 걸까?

이때 사용되는 metrics에는 '맨해튼 거리'라고 불리는 L1과, '유클리디안 거리'라 불리는 L2가 있다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8f825b18-4448-4215-9ea6-ae7aebbfda08/Untitled.png)

L1은 단지 두 이미지의 같은 위치에있는 픽셀값을 빼고 그 값을 절댓값을 취해 모두 더한 값을 뜻한다.
