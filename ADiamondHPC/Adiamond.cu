#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void imprimir_matriz(int *);
int *cargar_matriz(char *);
void cambiar(int *, int, int);
int *aDiamante(int *, int);
void rutas(int *m1, int *m2, int veces);

int nodos;
const int INF = 999999999;

int main(int argc, char const *argv[]) {

	int *m1 = cargar_matriz("data1.txt");
	int *m2  = NULL;

	imprimir_matriz(m1);
	printf("\n");

	int i;
	for(i = 0; i < 2; i++) {
		m2 = aDiamante(m1, nodos);
		m1 = m2;
	}

	imprimir_matriz(m2);
	printf("\n");

	free(m1);
	return 0;
}

void imprimir_matriz(int *mat) {
	int i, j;
	for (i = 0; i < nodos; i++) {
		for (j = 0; j < nodos; j++) {
			printf("[%d]", mat[i+j*nodos]);
		}
		printf("\n");
	}
}

int *cargar_matriz(char *archivo) {

	FILE *f;
	int fil, col, arcos, peso, *mat = NULL;
	char caracter;

	if ((f = fopen(archivo, "r")) == NULL) {
		printf("Error al cargar el archivo\n");
		exit(1);
	}

	while (!feof(f)) {

		while(fscanf(f, "%c", &caracter) != EOF && caracter != 'p')
	    	while(fscanf(f, "%c", &caracter) != EOF && caracter != '\n');

		fscanf(f, "%*s %d %d\n", &nodos, &arcos);

		if (caracter == 'p')
			printf("Nodos: %d, Arcos: %d\n", nodos, arcos);

		mat = (int *) malloc((nodos * nodos) * sizeof(int));

		while (fscanf(f, "%c %d %d %d\n", &caracter, &fil, &col, &peso) != EOF) {
			mat[(fil - 1) + (col - 1) * nodos] = peso;
			mat[(col - 1) + (fil - 1) * nodos] = peso;
		}

	}

	cambiar(mat, nodos, INF);
	fclose(f);
	return mat;
}

int *aDiamante(int *mat1, int tam) {
	int *result = (int *) malloc((tam * tam) * sizeof(int));
	int i, j, k;
	for (i = 0; i < tam; i++) {
			for (j = 0; j < tam; j++) {
			    result[i + j * tam] = INF;
					for (k = 0; k < tam; k++) {
					    int aux = mat1[i + k * tam] + mat1[k + j * tam];
						  if (aux < result[i + j * tam])
							result[i + j * tam] = aux;
					}
			}
	}
	 return result;
}

void cambiar(int *mat, int tam, int val) {
	int i, j;
	for (i = 0; i < tam; i++)	{
		for (j = 0; j < tam; j++) {
			if (mat[i + j * tam] == 0)
				mat[i + j * tam] = val;
		}
	}
}
