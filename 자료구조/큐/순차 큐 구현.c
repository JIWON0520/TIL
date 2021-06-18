#include<stdio.h>
#include<stdlib.h>
#define _CRT_SECURE_NO_WARNINGS
#define Q_SIZE 4

typedef char element;

typedef struct {
	element queue[Q_SIZE];
	int front, rear;
}QueueType;

//공백 순차 큐를 생성하는 연산
QueueType* createQueue() {
	QueueType* Q;
	Q = (QueueType*)malloc(sizeof(QueueType));
	Q->front = -1;
	Q->rear = -1;
	return Q;
}

//순차 큐가 공백 상태인지 검사하는 연산
int IsEmpty(QueueType* Q) {
	if (Q->front == Q->rear) {
		printf("Queue is empty!");
		return 1;
	}
	else return 0;
}

//순자 큐가 포화 상태인지검사하는 연산
int isFull(QueueType* Q) {
	if (Q->rear == Q_SIZE - 1) {
		printf("Queue is full!");
		return 1;
	}
	else return 0;
}
//순차 큐의 rear에 원소를 삽입하는 연산
void enQueue(QueueType* Q, element item) {
	if (isFull(Q)) return;
	else {
		Q->rear++;
		Q->queue[Q->rear] = item;
	}
}

//순차 큐의 front에서 원소를 삭제하는 연산
element deQueue(QueueType* Q) {
	if (IsEmpty(Q)) return NULL;
	else {
		Q->front++;
		return Q->queue[Q->front];
	}
}
//순차 큐의 가장 앞에 있는 원소를 검색하는 연산
element peek(QueueType* Q) {
	if (IsEmpty(Q)) exit(1);
	else {
		return Q->queue[Q->front + 1];
	}
}
//순차 뮤의 원소를 출력하는 연산
void printQ(QueueType* Q) {
	int i;
	printf("Queue : [");
	for (i = Q->front + 1;i <= Q->rear; i++) {
		printf("%3c", Q->queue[i]);
	}
	printf("]");
}
void main(void) {
	QueueType* Q1 = createQueue();// 큐 생성
	element data;
	printf("\n ***** 순차 큐 연산 ***** \n");
	printf("\n 삽입 A>>"); enQueue(Q1, 'A'); printQ(Q1);
	printf("\n 삽입 B>>");  enQueue(Q1, 'B'); printQ(Q1);
	printf("\n 삽입 C>>");  enQueue(Q1, 'C'); printQ(Q1);
	data = peek(Q1); printf(" peek item : %c \n", data);

	printf("\n 삭제 >>"); data = deQueue(Q1); printQ(Q1);
	printf("\t삭제 데이터 : %c", data);
	printf("\n 삭제 >>"); data = deQueue(Q1); printQ(Q1);
	printf("\t삭제 데이터 : %c", data);
	printf("\n 삭제 >>");  data = deQueue(Q1); printQ(Q1);
	printf("\t삭제 데이터 : %c", data);

	printf("\n 삽입 D>>"); enQueue(Q1, 'D'); printQ(Q1);
	printf("\n 삽입 E>>"); enQueue(Q1, 'E'); printQ(Q1);
}
