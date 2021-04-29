#include <stdio.h>
#include <cuda.h>

__global__ void ford_fulkerson(int vertices, int edges, int *graph, int *result){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    //Testing
    printf("Ford Fulkerson from GPU\n");
    *result = 1;
}

__global__ void push_relabel(int *result){
    //Testing
    printf("Push Relabel from GPU\n");
    *result = 1;
}

void printCPU(int *graph, int vertices){
    for(int i=0; i<vertices; ++i){
        for(int j=0; j<vertices; ++j){
            printf("%d ", graph[i * vertices + j]);
        }
        printf("\n");
    }
}

__global__ void printGPU(int *graph, int vertices){
    for(int i=0; i<vertices; ++i){
        for(int j=0; j<vertices; ++j){
            printf("%d ", graph[i * vertices + j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv){

    FILE *fin, *fout;
    
    char *inputfilename = argv[1];
    fin = fopen( inputfilename , "r");
    
    if ( fin == NULL )  {
        printf( "%s failed to open.", inputfilename );
        return 0;
    }

    char *outputfilename = argv[2]; 
    fout = fopen(outputfilename, "w");

    int vertices, edges, source, sink;
    fscanf(fin, "%d %d %d %d", &vertices, &edges, &source, &sink);

    // Declaring Variable
    int *cpu_graph, *gpu_graph;
    int *cpu_result, *gpu_result;

    // Initialising Variables
    cpu_graph = (int *) malloc(vertices * vertices * sizeof(int));
    cpu_result = (int *) malloc(sizeof(int));
    cudaMalloc(&gpu_graph, vertices * vertices * sizeof(int));
    cudaMalloc(&gpu_result, sizeof(int));

    memset(cpu_graph, 0, vertices * vertices * sizeof(int));

    // Taking Input & Generating Graph as Ajacency Matrix
    for(int i=0; i<edges; ++i){
        int u, v, c;
        fscanf(fin, "%d %d %d", &u, &v, &c);
        cpu_graph[(u - 1) * vertices + (v - 1)] = c;
    }

    cudaMemcpy(gpu_graph, cpu_graph, vertices * vertices * sizeof(int), cudaMemcpyHostToDevice);
    
    // Rigved's Part
    ford_fulkerson<<<1, 1>>>(vertices, edges, gpu_graph, gpu_result);
    
    // Sumit's Part
    // push_relabel<<<1, 1>>>(gpu_result);
  
    
    // Writing result back to CPU
    cudaMemcpy(cpu_result, gpu_result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Max Flow is: %d\n", *cpu_result);

    cudaDeviceSynchronize();
    fclose(fin);
    fclose(fout);
    return 0;
}