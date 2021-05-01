#include <stdio.h>
#include <cuda.h>

void find_augmenting_path(int vertices, int edges, int source, int sink, int *graph, int *path, int *path_length, int *queue, int *visited, int *min_flow){

    int start = 0, end = 1;

    queue[start * 2] = source;
    queue[start * 2 + 1] = -1;

    int temp_path_length = 0;
    (*path_length) = -1;

    int prev;

    while(start < end){
        
        int loop_start = start, loop_end = end;

        for(int i=loop_start; i<loop_end; ++i){
            int u = queue[i * 2];

            if(u == sink){
                *path_length = temp_path_length;
                prev = queue[i*2 + 1];
            }
            
            for(int j=0; j<vertices; ++j){

                int c = graph[u * vertices * 2 + j * 2 + 0];
                int f = graph[u * vertices * 2 + j * 2 + 1];

                if(c-f > 0 && visited[j] == 0){
                    visited[j] = 1;
                    queue[end * 2] = j;
                    queue[end * 2 + 1] = u;
                    ++end;
                }
            }

            ++start;
            visited[u] = true;
        }
        
        ++temp_path_length;
    }

    if(*path_length != -1){
        path[*path_length] = sink;
        
        for(int i=(*path_length) - 1; i>=0; --i){
            path[i] = prev;
            
            int u = path[i];
            int v = path[i + 1];

            if(graph[u * vertices * 2 + v * 2 + 0] - graph[u * vertices * 2 + v * 2 + 1] < *min_flow){
                *min_flow = graph[u * vertices * 2 + v * 2 + 0] - graph[u * vertices * 2 + v * 2 + 1];
            }

            for(int j=0; j<vertices; ++j){
                if(queue[j*2] == prev){
                    prev = queue[j*2 + 1];
                    break;
                }
            }
        }

        // for(int i=0; i<(*path_length); ++i){
        //     printf("%d => ", path[i]);
        // }
        // printf("%d\n", path[*path_length]);
    }

    //printf("Path Length: %d\n", *path_length);
}

__global__ void update_residual_graph(int vertices, int edges, int *graph, int *path, int *path_length, int *min_flow){
    int edge = blockIdx.x * blockDim.x + threadIdx.x;

    if(edge < *path_length){
        int u = path[edge];
        int v = path[edge + 1];
        
        graph[u * vertices * 2 + v * 2 + 1] += *min_flow;
        graph[v * vertices * 2 + u * 2 + 0] += *min_flow;

        //printf("After updation edge (%d, %d) becomes (%d, %d)\n", u, v, graph[u * vertices * 2 + v * 2 + 0], graph[u * vertices * 2 + v * 2 + 1]);
        //printf("After updation reverse edge (%d, %d) becomes (%d, %d)\n", v, u, graph[v * vertices * 2 + u * 2 + 0], graph[v * vertices * 2 + u * 2 + 1]);
    }
}

__global__ void find_max_flow(int vertices, int edges, int source, int *graph, int *result){
    int v = blockIdx.x * blockDim.x + threadIdx.x;

    if(v < vertices){
        if(graph[source * vertices * 2 + v * 2 + 1] > 0){
            // printf("Result:%d Vertex:%d Value:%d\n", *result, v, graph[source * vertices * 2 + v * 2 + 1]);
            atomicAdd(result, graph[source * vertices * 2 + v * 2 + 1]);
        }
    }
}

void printCPU(int *graph, int vertices){
    for(int i=0; i<vertices; ++i){
        for(int j=0; j<vertices; ++j){
            printf("%d ", graph[i * vertices + j]);
        }
        printf("\n");
    }
}

void printCPUFF(int *graph, int vertices){
    for(int i=0; i<vertices; ++i){
        for(int j=0; j<vertices; ++j){
            printf("(%d %d)", graph[i * vertices * 2 + j*2 + 0], graph[i * vertices * 2+ j*2 + 1]);
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

__global__ void printGPUFF(int *graph, int vertices){
    for(int i=0; i<vertices; ++i){
        for(int j=0; j<vertices; ++j){
            printf("(%d %d)", graph[i * vertices * 2 + j*2 + 0], graph[i * vertices * 2 + j*2 + 1]);
        }
        printf("\n");
    }
}

void edmonds_karp(int vertices, int edges, int source, int sink, int *cpu_graph, int *gpu_graph, int *cpu_result, int *gpu_result){
    int *cpu_path_length, *gpu_path_length;
    int *cpu_path, *gpu_path, *cpu_min_flow, *gpu_min_flow;
    int *cpu_queue, *cpu_visited;
    
    cpu_path_length = (int *) malloc(sizeof(int));
    cpu_path = (int *) malloc(vertices * sizeof(int));
    cpu_queue = (int *) malloc(2 * vertices * sizeof(int));
    cpu_visited = (int *) malloc(vertices * sizeof(int));
    cpu_min_flow = (int *) malloc(sizeof(int));

    cudaMalloc(&gpu_min_flow, sizeof(int));
    cudaMalloc(&gpu_path_length, sizeof(int));
    cudaMalloc(&gpu_path, vertices * sizeof(int));

    cudaMemcpy(gpu_graph, cpu_graph, 2 * vertices * vertices * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMemset(gpu_result, 0, sizeof(int));

    while(1){
        memset(cpu_queue, 0, 2 * vertices * sizeof(int));
        memset(cpu_visited, 0, vertices * sizeof(int));

        *cpu_min_flow = INT_MAX;
        find_augmenting_path(vertices, edges, source, sink, cpu_graph, cpu_path, cpu_path_length, cpu_queue, cpu_visited, cpu_min_flow);
        
        if(*cpu_path_length == -1){
            break;
        }

        cudaMemcpy(gpu_path_length, cpu_path_length, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_path, cpu_path, vertices * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_min_flow, cpu_min_flow, sizeof(int), cudaMemcpyHostToDevice);
        
        int blocks = (ceil)((double) *cpu_path_length / 1024);
        update_residual_graph<<<blocks, 1024>>>(vertices, edges, gpu_graph, gpu_path, gpu_path_length, gpu_min_flow);
        
        cudaMemcpy(cpu_graph, gpu_graph, 2 * vertices * vertices * sizeof(int), cudaMemcpyDeviceToHost);
    }

    int blocks = (ceil)((double) vertices / 1024);
    find_max_flow<<<blocks, 1024>>>(vertices, edges, source, gpu_graph, gpu_result);

    cudaMemcpy(cpu_result, gpu_result, sizeof(int), cudaMemcpyDeviceToHost);
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
    int *cpu_graph, *gpu_graph, *cpu_ff_graph, *gpu_ff_graph;
    int *cpu_result, *gpu_result;

    // Initialising Variables
    cpu_graph = (int *) malloc(vertices * vertices * sizeof(int));
    cpu_ff_graph = (int *) malloc(2 * vertices * vertices * sizeof(int));
    cpu_result = (int *) malloc(sizeof(int));

    cudaMalloc(&gpu_graph, vertices * vertices * sizeof(int));
    cudaMalloc(&gpu_ff_graph, 2 * vertices * vertices * sizeof(int));
    cudaMalloc(&gpu_result, sizeof(int));

    memset(cpu_graph, 0, vertices * vertices * sizeof(int));
    memset(cpu_ff_graph, 0, 2 * vertices * vertices * sizeof(int));

    // Taking Input & Generating Graph as Ajacency Matrix
    for(int i=0; i<edges; ++i){
        int u, v, c;
        fscanf(fin, "%d %d %d", &u, &v, &c);
        cpu_graph[(u - 1) * vertices + (v - 1)] = c;
        cpu_ff_graph[(u - 1) * vertices * 2 + (v - 1) * 2 + 0] = c;
        cpu_ff_graph[(u - 1) * vertices * 2 + (v - 1) * 2 + 1] = 0;
    }

    cudaMemcpy(gpu_graph, cpu_graph, vertices * vertices * sizeof(int), cudaMemcpyHostToDevice);
    
    // Rigved's Part
    edmonds_karp(vertices, edges, source - 1, sink - 1, cpu_ff_graph, gpu_ff_graph, cpu_result, gpu_result);
    
    // Sumit's Part
    // push_relabel<<<1, 1>>>(gpu_result);  
    
    // Writing result back to CPU
    //cudaMemcpy(cpu_result, gpu_result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Max Flow is: %d\n", *cpu_result);

    fclose(fin);
    fclose(fout);
    return 0;
}