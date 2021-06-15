#include <stdio.h>

int N_big_number(int numbers[], int n) {
	int max,max_idx;

	for (int i = 0; i < n; i++) {
		max = 0;
		for (int j = 0; j < 10; j++) {
			if (max < numbers[j]) {
				max = numbers[j];
				max_idx = j;
			}
		}
		numbers[max_idx] = 0;
	}
	return max;
}
int main(void) {
	int numbers[10] = { 1,3,5,4,7,6,12,8,9,2 };
	int n;
	scanf_s("%d", &n);
	printf("%d", N_big_number(numbers, n));
	return 0;
}