#include <stdio.h>
#include <stdlib.h>

float **crear_matriz(int, int, FILE *);
float **multiplicar_matriz(float **, float **, int, int, int, int);
void imprimir_matriz(float **, int, int);

int main(){

    FILE *arch1;
    FILE *arch2;

    int filaA, columnaA, filaB, columnaB;
    float **ma1, **ma2, **ma3;

    arch1 = fopen("matriz1.txt", "r");
    fscanf(arch1, "%d %d", &filaA, &columnaA);

    arch2 = fopen("matriz2.txt", "r");
    fscanf(arch2, "%d %d", &filaB, &columnaB);

    ma1 = crear_matriz(filaA, columnaA, arch1);
    ma2 = crear_matriz(filaB, columnaB, arch2);

    imprimir_matriz(ma1, filaA, columnaA);
    imprimir_matriz(ma2, filaB, columnaB);

    if (columnaA == filaB){
        ma3 = multiplicar_matriz(ma1, ma2, filaA, columnaA, filaB, columnaB);
        imprimir_matriz(ma3, filaA, columnaB);
    }
    else{
        printf("No se pueden multiplicar las matrices... \n");
        return 0;
    }

    free(*ma1); free(*ma2); free(*ma3);
    free(ma1); free(ma2); free(ma3);

    return 0;
}

float **crear_matriz(int fila, int columna, FILE *f){
    int i, j;
    float caracter, **matriz;

    printf("Allocate memory \n");
    matriz = (float **)malloc(fila*sizeof(float*));
    for (i = 0; i < fila; i++)
        matriz[i] = (float *)malloc(columna*sizeof(float));

    printf("Fill matrix \n");
    for (i = 0; i < fila; i++){
        for (j = 0; j < columna; j++){
            caracter = fgetc(f);
            fscanf(f, "%f", &caracter);
            matriz[i][j] = caracter;
        }
    }
    return matriz;
}

float **multiplicar_matriz(float **matriz1, float **matriz2, int fila1, int columna1, int fila2, int columna2){
    int i, j, k;
    float **matriz, s = 0;
    int chunck = 50;

    printf("Allocate memory \n");
    matriz = (float **)malloc(fila1*sizeof(float*));
    for (i = 0; i < fila1; i++)
        matriz[i] = (float *)malloc(columna2*sizeof(float));

  //printf("LLego al pragma \n");
    #pragma omp parallel shared(matriz1, matriz2, matriz) private(i, j, k) num_threads(4)
    {
  
        #pragma omp for schedule(dynamic, chunk)
        printf("Matrix multiplication \n");
        for (i = 0; i < fila1; i++){
            for (j = 0; j < columna2; j++){
                for (k = 0; k < fila2; k++){
                    s = s + matriz1[i][k] * matriz2[k][j];
                }
                matriz[i][j] = s;
                s = 0;
            }
        }
        return matriz;
    }
}
    


void imprimir_matriz(float **matriz, int fila, int columna){
    int i, j;

    printf("Print matrix \n");
    for (i = 0; i < fila; i++){
        for (j = 0; j < columna; j++){
            printf("[%.2f]", matriz[i][j]);
        }
        printf("\n");
    }
}
