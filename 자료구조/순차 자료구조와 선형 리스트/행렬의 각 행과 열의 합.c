#include <stdio.h>
void matrix_addtion() {
	int arr[3][3] = { {1,10,14},{20,3,5},{15,12,0} };

	for (int i = 0; i < 3; i++) {
		int sum_row = 0, sum_col = 0;
		for (int j = 0; j < 3; j++) {
			sum_row += arr[i][j];
			sum_col += arr[j][i];
		}
		printf("%d행의 합:%d\n", i + 1, sum_row);
		printf("%d열의 합:%d\n", i + 1, sum_col);
	}
}

int main() {
	matrix_addtion();
	return 0;
}