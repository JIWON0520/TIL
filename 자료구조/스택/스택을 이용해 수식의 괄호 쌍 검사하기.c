#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef char element;

typedef struct stackNode {
	char data;
	struct stackNode* link;
}stackNode;

stackNode* top;

//스택이 공백 상태인지 확인하는 연산
int isEmpty() {
	if (top == NULL) return 1;
	else return 0;
}

//스택에 원소를 삽입하는 연산
void push(element item) {
	stackNode* temp = (stackNode*)malloc(sizeof(stackNode));
	temp->data = item;
	temp->link = top;
	top = temp;
}
//스택의 top에서 원소를 삭제하는 연산
element pop() {
	element item;
	stackNode* temp = top;
	if (isEmpty()) {
		printf("\n\n Stack is Empty!n");
		return 0;
	}
	else {
		item = temp->data;
		top = temp->link;
		free(temp);
		return item;
	}
}
int testPair(char* exp) {
	char symbol, open_pair;
	int i,length = strlen(exp);
	top = NULL;
	for (i = 0; i < length; i++) {
		symbol = exp[i];
		switch (symbol) {
		case '[':
		case '{':
		case '(':
			push(symbol); break;
		case ']':
		case '}':
		case ')':
			if (top == NULL) return 0;
			else {
				open_pair = pop();
				if ((open_pair == '[' && symbol != ']') || (open_pair == '{' && symbol != '}') || (open_pair == '(' && symbol != ')'))
					return 0;
				else break;
			}
		}
	}
	if (top == NULL) return 1;
	else return 0;
}
void main(void) {
	char* express =
		"{(A+B)-3}*5+[{cos(x+y)+7}-1]*4";
	printf("%s", express);

	if (testPair(express) == 1)
		printf("\n\n수식의 괄호가 맞게 사용	되었습니다!\n");
	else
		printf("\n\n 수식의 괄호가 틀렸습니	다!\n");
}
