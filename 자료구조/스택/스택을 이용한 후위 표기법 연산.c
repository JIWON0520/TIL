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
element evalPostfix(char* exp) {
	int opr1, opr2, value, i = 0;
	int length = strlen(exp);
	char symbol;
	top = NULL;

	for (i = 0; i < length; i++) {
		symbol = exp[i];
		if (symbol != '+' && symbol != '-' && symbol != '*' && symbol != '/') {
			value = symbol-'0';
			push(value);
		}
		else {
			opr2 = pop();
			opr1 = pop();
			switch (symbol) {
			case '+':push(opr1 + opr2); break;
			case '-':push(opr1 - opr2); break;
			case '*':push(opr1 * opr2); break;
			case '/':push(opr1 / opr2); break;
			}
		}
	}
	return pop();
}

void main(void) {
	int result;
	char* express = "35*62/-";
	printf("후위 표기식  %s", express);

	result = evalPostfix(express);
	printf("\n\n 연산 결과 => %d", result);

}

