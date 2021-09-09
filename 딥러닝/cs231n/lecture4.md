## Lecture 4: Backpropagation and Neural Networks

![화면 캡처 2021-09-09 191505](https://user-images.githubusercontent.com/77263283/132705775-f18fd108-356f-442d-80c6-a8dda3b48e69.png)

우리는 lecture3에서 파라미터 W를 업데이트 하기 위해서 경사하강법(SGD)을 사용한다고 배웠다. loss가 최소가 되는 방향을 알기 위해서는 loss functiond에서 W의 미분값을 구하면 된다. 만약 주어진 함수가 엄청 복잡하다면 우리는 어떻게 미분값을 구할 수 있을까? 우리는 이러한 복잡한 함수에 대해 computational graph를 그려서 미분값을 구할 수 있다.

computational graph란 위의 그림처럼 각 노드를 계산 순서에 따라 표시한 그래프이다. 위의 그래프는 우리가 다뤘던 linear classifier의 식을 그래프화 한것이다. 

이런 computational graph를 이용하면 우리는 '역전파(back propagation)'라고 불리는 기술을 사용할 수 있다. 역전파는 chain rule을 재귀적으로 사용해서 모든 변수의 gradient를 구할 수 있는 테크닉이다.

![화면 캡처 2021-09-09 192656](https://user-images.githubusercontent.com/77263283/132705805-58998967-a236-47a0-bc7f-64bedf2dae77.png)

back propagation의 동작과정을 보기위해 간단한 예시를 들어보자.

함수 f가 있다. f는 x,y,z로 구성되며 식은 (x+y)z로 표현된다. 우리가 처음으로 해야될것은 이 함수에 대한 computational graph를 그리는 것이다. 위의 그래프를 보면 첫번째 노드는 x와y의 합을 나타내는 노드이고 그다음 노드는 이 합노드의 결과롸 z를 곱해주는 노드이다. 그리고 x,y,z의 값에 따라 노드에 계산 값을 적어준다.  그리고 합 노드를 q라하고 맨 마지막 노드를 f라 하자.  q에대한 x와y의 gradient는 각각1이고 f에대한 q와z의 gradient는 z와q이다. 

우리가 최종적으로 구하고자하는 것은 f에대한 x,y,z의 gradient이다. 우리는 그래프의 맨 마지막 노드부터 뒤로 연속적인 chain rule을 적용하면 된다.

![화면 캡처 2021-09-09 193712](https://user-images.githubusercontent.com/77263283/132705821-8d93f070-220a-4c6c-9d40-93c8efcdab25.png)

먼저 맨 마지막 노드 변수인 f의 gradient값부터 구해보자. f에대한 f의 gradient값은 1이다. 

![화면 캡처 2021-09-09 193934](https://user-images.githubusercontent.com/77263283/132705832-f8491c68-cbe9-4f25-801d-5edbd1068532.png)

그리고 뒤의 노드로 가서 f에대한 z의 gradient를 구해보자. df/dz=q라는 것을 알고 있다. 따라서 z의 gradient는 3이다. 

![화면 캡처 2021-09-09 194307](https://user-images.githubusercontent.com/77263283/132705843-220fb414-467f-45a3-b3b5-a8e4f2d00cc8.png)

그리고 df/dq를 구해보자. df/dq는 z라는 것을 알고 있으므로 q의 gradient값은 -4이다. 

![화면 캡처 2021-09-09 194416](https://user-images.githubusercontent.com/77263283/132705853-1e0c7c89-e80e-4711-b4aa-67a0d7ce2aea.png)

그다음 우리는 또 뒤로가서 f에대한 y의 gradient를 구하고싶다. 하지만 y는 f와 직접적으로 연결되어있지 않다.  그래서 우리는 중간노드 q에 대한 y의 gradient값을 구하고, f에대한 q의 gradient값을 구함으로써 y가 f에 미치는 영향을 구할 수 있다. 즉, df/dy=dq/dy*df/dq로 표기될 수 있다. dq/dy는 1인것을 알고 있고, df/dq또한 -4인것을 알수 있으므로 df/dy는 -4이다.

![화면 캡처 2021-09-09 195410](https://user-images.githubusercontent.com/77263283/132705859-9a49044a-a83a-479d-a68b-0c9010955b76.png)

마지막으로 df/dx도 같은 방법으로 구할수 있다. 

back propagation의 과정을 살펴보면 직관적으로 알 수 있는 패턴이 있다.

![화면 캡처 2021-09-09 230934](https://user-images.githubusercontent.com/77263283/132705875-432fc22f-897b-4797-8b24-8cd19b990bd1.png)

add gate는 두개의 input값의 gradient를 동등하게 배분하는 역할을 한다. 위의 예시처럼 두개의 input이 add gate를 거치면 두 input의 gradient는 같은 값을 가진다. 

![화면 캡처 2021-09-09 231120](https://user-images.githubusercontent.com/77263283/132705893-67174d45-b947-4f09-89be-c070315e21d7.png)

그렇다면 max gate는 어떤 역할을 할까? 위의 예시를보면 max gate를 거치는 두 input의 gradient는 0 또는 input값 그대로이다. max gate는 input값 둘중 큰 값에만 영향을 받기때문이다. max gate는 더 큰 값만 선택해 다음 노드로 전달해주기 때문에 라우터 역할을 한다. 

![화면 캡처 2021-09-09 232355](https://user-images.githubusercontent.com/77263283/132705904-d77b22bd-e693-4116-9185-2781a7e9f672.png)

그렇다면 mul gate는 어떤 역할을 할까? mul gate를 통과하는 input의 local gradient는 다른 변수(브랜치) 값이다. 즉 mul gate는 switcher역할을 한다.

![화면 캡처 2021-09-09 232744](https://user-images.githubusercontent.com/77263283/132705915-6079de0e-a034-43ca-9eb1-cb8231a0c3b7.png)

그리고 위의 그림처럼 한 노드가 여러개의 노드에 연결되어 있는경우에는 각 노드로부터 역전파된 gradient들을 합하여 총 gradient를 계산한다. 하나의 노드가 두개의 노드와 연결되어있다면 하나의 노드가 조금이라도 바뀌면 두 노드 모두에게 영향을 미칠것이다. 이렇게 된다면 역전파를 통해 들어온 두개의 gradient모두도 이 노드에 영향을 미칠 것이다.
