#include<stdio.h>
#include<stdlib.h>
#define _CRT_SECURE_NO_WARNINGS
#define cQ_SIZE 4

typedef char element;

typedef struct {
	element queue[cQ_SIZE];
	int front, rear;
}QueueType;

//공백 원형 큐를 생성하는 연산
QueueType* createQueue() {
	QueueType* cQ;
	cQ = (QueueType*)malloc(sizeof(QueueType));
	cQ->front = 0;
	cQ->rear = 0;
	return cQ;
}

//원형 큐가 공백 상태인지 검사하는 연산
int isEmpty(QueueType* cQ) {
	if (cQ->front == cQ->rear) {
		printf("Circluar Queue is empty!");
		return 1;
	}
	else return 0;
}

//원형 큐가 포화 상태인지검사하는 연산
int isFull(QueueType* cQ) {
	if ((cQ->rear + 1) % cQ_SIZE == cQ->front) {
		printf("Circular Queue is full");
		return 1;
	}
	else return 0;
}
//원형 큐의 rear에 원소를 삽입하는 연산
void enQueue(QueueType* cQ, element item) {
	if (isFull(cQ)) return;
	else {
		cQ->rear = (cQ->rear + 1) % cQ_SIZE;
		cQ->queue[cQ->rear] = item;
	}
}
//원형 큐의 front에서 원소를 삭제하는 연산
element deQueue(QueueType* cQ) {
	if (isEmpty(cQ)) return NULL;
	else {
		cQ->front = (cQ->front + 1) % cQ_SIZE;
		return cQ->queue[cQ->front];
	}
}
//원형 큐의 가장 앞에 있는 원소를 검색하는 연산
element peek(QueueType* cQ) {
	if (isEmpty(cQ)) exit(1);
	else {
		return cQ->queue[(cQ->front + 1) % cQ_SIZE];
	}
}
//원형 큐의 원소를 출력하는 연산
void printQ(QueueType* cQ) {
	int i,first, last;
	first = (cQ->front + 1) % cQ_SIZE;
	last = (cQ->rear + 1) % cQ_SIZE;
	printf("Circular Queue: [");
	i = first;
	while (i != last) {
		printf("%3c", cQ->queue[i]);
		i = (i + 1) % cQ_SIZE;
	}
	printf("]");
}
void main(void) {
	QueueType* cQ = createQueue();
	element data;
	printf("\n ***** 원형 큐의 연산 *****\n");
	printf("\n 삽입 A>>"); enQueue(cQ, 'A'); printQ(cQ);
	printf("\n 삽입 B>>"); enQueue(cQ, 'B'); printQ(cQ);
	printf("\n 삽입 C>>"); enQueue(cQ, 'C'); printQ(cQ);
	data = peek(cQ); printf(" peek item : %c \n", data);

	printf("\n 삭제 >>"); data = deQueue(cQ); printQ(cQ);
	printf("\t 삭제 데이터 : %c", data);
	printf("\n 삭제 >>"); data = deQueue(cQ); printQ(cQ);
	printf("\t 삭제 데이터 : %c", data);
	printf("\n 삭제 >>"); data = deQueue(cQ); printQ(cQ);
	printf("\t 삭제 데이터 : %c", data);

	printf("\n 삽입 D>>"); enQueue(cQ, 'D'); printQ(cQ);
	printf("\n 삽입 E>>"); enQueue(cQ, 'E'); printQ(cQ);

}
