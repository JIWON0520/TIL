#include<stdio.h>
#include<stdlib.h>
#include<string.h>


//�ܼ� ���� ����Ʈ�� ��� ������ ����ü�� ����
typedef struct ListNode {
	char data[4];
	struct ListNode* link;
}listNode;

//����Ʈ ������ ��Ÿ���� head ��带 ����ü�� ����
typedef struct {
	listNode* head;
}linkedList_h;

//���� ���� ����Ʈ�� �����ϴ� ����
linkedList_h* createLinkedList_h(void) {
	linkedList_h* L;
	L = (linkedList_h*)malloc(sizeof(linkedList_h));
	L->head = NULL;
	return L;
}

//���� ����Ʈ ��ü �޸𸮸� �����ϴ� ����
void freeLinkedList_h(linkedList_h* L) {
	listNode* p;
	while (L->head != NULL) {
		p = L->head;
		L->head = L->head->link;
		free(p);
		p = NULL;
	}
}
//���� ����Ʈ�� ������� ����ϴ� ����
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
//ù ��° ���� �����ϴ� ����
void insertFirstNode(linkedList_h* L, char* x) {
	listNode* newNode;
	newNode = (listNode*)malloc(sizeof(listNode));
	strcpy_s(newNode->data, 4,x);
	newNode->link = L->head;
	L->head = newNode;
}
//��带 pre�ڿ� �����ϴ� ����
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
//������ ���� �����ϴ� ����
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
		temp = temp->link;  //���� ����Ʈ�� ������ ��带 ã��
	temp->link = newNode;
}
//����Ʈ���� ��� p�� �����ϴ� ����!
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
//����Ʈ���� x ��带 Ž���ϴ� ����
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
	printf("[1]����Ʈ�� [��],[��],[��]��� �����ϱ�!\n");
	insertLastNode(L, "��");
	insertLastNode(L, "��");
	insertLastNode(L, "��");
	printList(L);

	printf("[2]����Ʈ���� [��] ��� Ž���ϱ�!\n");
	p = searchNode(L, "��");
	if (p == NULL)printf("ã�� �����Ͱ� �����ϴ�.\n");
	else printf("[%s]�� ã�ҽ��ϴ�.\n",p->data);

	printf("[3]����Ʈ���� [��]�ڿ�[��]��� �����ϱ�!\n");
	insertMiddleNode(L, p, "��");
	printList(L);

	printf("[4]����Ʈ���� [��]��� �����ϱ�!\n");
	p = searchNode(L, "��");
	deleteNode(L, p);
	printList(L);

	freeLinkedList_h(L);
	return 0;
}