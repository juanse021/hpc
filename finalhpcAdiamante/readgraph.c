#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <malloc.h>
#include <time.h>

void imprimir_matriz(int *);
int *cargar_matriz(char *);
void cambiar(int *, int, int);


int nodos;
const int INF = 999999999;

__global__ void aDiamante(int *mat1, int tam, int *result) {

    const int col = blockIdx.x * blockDim.x + threadIdx.x; 
    const int fil = blockIdx.y * blockDim.y + threadIdx.y;
    int index = col + fil * tam;
   	
	result[index] = INF;
	
	for (int i = 0; i < tam; i++) {
	    for (int j = 0; j < tam; j++) {
	        result[index] = INF;
	        for (int k = 0; k < tam; k++) {
	            int aux = mat1[i + k * tam] + mat1[k + j * tam];
	            if (aux < result[index])
	                result[index] = aux;
	        }
	    }
	}
}


int main(int argc, char const *argv[]) {

    cudaError_t error = cudaSuccess;
    char *filename = "data1.txt";
    
    clock_t startGPU, endGPU;

	int *h_m1 = cargar_matriz(filename);
	int *h_m2 = NULL, *d_m1 = NULL, *d_m2 = NULL;
	
	int size = (nodos * nodos) * sizeof(int); 
	
	error = cudaMalloc((void**)&d_m1, size);
	if (error != cudaSuccess) {
	    printf("Error ... d_m1 \n");
	    return -1;
	}

    error = cudaMalloc((void**)&d_m2, size);
    if (error != cudaSuccess) {
        printf("Error ...d_m2 \n");
        return -1;
    }

	startGPU = clock();
	error = cudaMemcpy(d_m1, h_m1, size, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
	    printf("Error ... h_m1 a d_m1 \n");
	    return -1;
	}
	
	int blocksize = 32;
	dim3 dimBlock(blocksize, blocksize, 1);
	dim3 dimGrid(ceil(nodos/float(blocksize)), ceil(nodos/float(blocksize)), 1);	
	
	aDiamante<<<dimGrid, dimBlock>>>(d_m1, nodos, d_m2);
    cudaDeviceSynchronize();
    
    h_m2 = (int *)malloc(size);
    error = cudaMemcpy(h_m2, d_m2, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        printf("Error ...d_m2 a h_m2 \n");
        return -1;    
    }
    
    endGPU = clock();
    
    double timeGPU = ((double)(endGPU - startGPU)) / CLOCKS_PER_SEC;
    printf("El tiempo de ejecucion en GPU es: %.10f \n", timeGPU);

	free(h_m1);
	free(h_m2);
	
	cudaFree(d_m1);
	cudaFree(d_m2);
	
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


void cambiar(int *mat, int tam, int val) {
	int i, j;
	for (i = 0; i < tam; i++)	{
		for (j = 0; j < tam; j++) {
			if (mat[i + j * tam] == 0)
				mat[i + j * tam] = val;
		}
	}
}