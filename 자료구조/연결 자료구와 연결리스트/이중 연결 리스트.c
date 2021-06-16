#include<stdio.h>
#include<stdlib.h>
#include<string.h>

//���� ���� ����Ʈ�� ��� ������ ����ü�� ����
typedef struct ListNode {
	struct ListNode* llink;
	char data[4];
	struct ListNode* rlink;
}listNode;

//����Ʈ�� ������ ��Ÿ���� head��带 ����ü�� ����
typedef struct {
	listNode* head;
}linkedList_h;

//���� ���� ���� ����Ʈ�� �����ϴ� ����
linkedList_h* createLinkedList_h(void) {
	linkedList_h* DL;
	DL = (linkedList_h*)malloc(sizeof(linkedList_h));
	DL->head = NULL;
	return DL;
}

//���� ���� ����Ʈ�� ������� ����ϴ� ����
void printList(linkedList_h* DL) {
	listNode* p;
	printf("DL=(");
	p = DL->head;
	while (p != NULL) {
		printf("%s", p->data);
		p = p->rlink;
		if (p != NULL)
			printf(",");
	}
	printf(")\n");
}

//pre�ڿ� ��带 �����ϴ� ����
void insertNode(linkedList_h* DL, listNode* pre, char* x) {
	listNode* newNode;
	newNode = (listNode*)malloc(sizeof(listNode));
	strcpy_s(newNode->data, 4, x);
	if (DL->head == NULL) {
		newNode->rlink = NULL;
		newNode->llink = NULL;
		DL->head = newNode;
	}
	else {
		newNode->rlink = pre->rlink;
		newNode->llink = pre;
		pre->rlink = newNode;
		if (newNode->rlink != NULL)
			newNode->rlink->llink = newNode;
	}
}
//���� ���� ����Ʈ���� old��带 �����ϴ� ����
void deleteNode(linkedList_h* DL, listNode* old) {
	if (DL->head == NULL) return;
	else if (old == NULL) return;
	else {
		old->llink->rlink = old->rlink;
		old->rlink->llink = old->rlink;
		free(old);
	}
}
//����Ʈ���� X ��带 Ž���ϴ� ����
listNode* searchNode(linkedList_h* DL, char* x) {
	listNode* temp;
	temp = DL->head;
	while (temp != NULL) {
		if (strcmp(temp->data, x) == 0) return temp;
		else temp = temp->rlink;
	}
	return temp;
}
int main() {
	linkedList_h* DL;
	listNode* p;
	DL = createLinkedList_h();
	printf("[1]���� ���� ����Ʈ �����ϱ�!\n");
	printList(DL);

	printf("[2]���� ���� ����Ʈ�� [��]��� �����ϱ�!\n");
	insertNode(DL, NULL,"��");
	printList(DL);

	printf("[3]���� ���� ����Ʈ�� [��] ��� �ڿ� [��] ��� �����ϱ�!\n");
	p = searchNode(DL, "��");
	insertNode(DL, p, "��");
	printList(DL);

	printf("[4]���� ���� ����Ʈ�� [��]��� �ڿ� [��] ��� �����ϱ�!\n");
	p = searchNode(DL, "��");
	insertNode(DL, p, "��");
	printList(DL);

	printf("[5]���� ���� ����Ʈ���� [��]��� �����ϱ�!\n");
	p = searchNode(DL, "��");
	deleteNode(DL, p);
	printList(DL);

	return 0;
}