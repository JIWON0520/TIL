#include<stdio.h>
#include<stdlib.h>
#include<string.h>


//단순 연결 리스트의 노드 구조를 구조체로 정의
typedef struct ListNode {
	char data[4];
	struct ListNode* link;
}listNode;

//리스트 시작을 나타내는 head 노드를 구조체로 정의
typedef struct {
	listNode* head;
}linkedList_h;

//공백 연결 리스트를 생성하는 연산
linkedList_h* createLinkedList_h(void) {
	linkedList_h* L;
	L = (linkedList_h*)malloc(sizeof(linkedList_h));
	L->head = NULL;
	return L;
}

//연결 리스트 전체 메모리를 해제하는 연산
void freeLinkedList_h(linkedList_h* L) {
	listNode* p;
	while (L->head != NULL) {
		p = L->head;
		L->head = L->head->link;
		free(p);
		p = NULL;
	}
}
//연결 리스트를 순서대로 출력하는 연산
void printList(linkedList_h* L) {
	listNode* p;
	printf("L=(");
	p = L->head;
	while (p != NULL) {
		printf("%s", p->data);
		p = p->link;
		if (p != NULL) printf(",");
	}
	printf(")\n");
}
//첫 번째 노드로 삽입하는 연산
void insertFirstNode(linkedList_h* L, char* x) {
	listNode* newNode;
	newNode = (listNode*)malloc(sizeof(listNode));
	strcpy_s(newNode->data, 4,x);
	newNode->link = L->head;
	L->head = newNode;
}
//노드를 pre뒤에 삽입하는 연산
void insertMiddleNode(linkedList_h* L,listNode* pre, char* x) {
	listNode* newNode;
	newNode = (listNode*)malloc(sizeof(listNode));
	strcpy_s(newNode->data, 4, x);
	if (L == NULL) {
		newNode->link = NULL;
		L->head = newNode;
	}
	else if (pre == NULL) {
		L->head = newNode;
	}
	else {
		newNode->link = pre->link;
		pre->link = newNode;
	}
}
//마지막 노드로 삽입하는 연산
void insertLastNode(linkedList_h* L, char* x) {
	listNode* newNode;
	listNode* temp;
	newNode = (listNode*)malloc(sizeof(listNode));
	strcpy_s(newNode->data, 4, x);
	newNode->link = NULL;
	if (L->head==NULL) {
		L->head = newNode;
		return;
	}
	temp = L->head;
	while (temp->link != NULL)
		temp = temp->link;  //현재 리스트의 마지막 노드를 찾음
	temp->link = newNode;
}
//리스트에서 노드 p를 삭제하는 연산!
void deleteNode(linkedList_h* L, listNode* p) {
	listNode* pre;
	if (L->head == NULL) return;
	if (L->head->link == NULL) {
		free(L->head);
		L->head == NULL;
		return;
	}
	else if (p == NULL)return;
	else {
		pre = L->head;
		while (pre->link != p) {
			pre = pre->link;
		}
		pre->link = p->link;
		free(p);
	}
}
//리스트에서 x 노드를 탐색하는 연산
listNode* searchNode(linkedList_h* L, char* x) {
	listNode* temp;
	temp = L->head;
	while (temp != NULL) {
		if (strcmp(temp->data, x) == 0) return temp;
		else temp = temp->link;
	}
	return temp;
}
int main() {
	linkedList_h* L;
	listNode* p;
	L = createLinkedList_h();
	printf("[1]리스트에 [월],[수],[일]노드 삽입하기!\n");
	insertLastNode(L, "월");
	insertLastNode(L, "수");
	insertLastNode(L, "일");
	printList(L);

	printf("[2]리스트에서 [수] 노드 탐색하기!\n");
	p = searchNode(L, "수");
	if (p == NULL)printf("찾는 데이터가 없습니다.\n");
	else printf("[%s]를 찾았습니다.\n",p->data);

	printf("[3]리스트에서 [수]뒤에[금]노드 삽입하기!\n");
	insertMiddleNode(L, p, "금");
	printList(L);

	printf("[4]리스트에서 [일]노드 삭제하기!\n");
	p = searchNode(L, "일");
	deleteNode(L, p);
	printList(L);

	freeLinkedList_h(L);
	return 0;
}