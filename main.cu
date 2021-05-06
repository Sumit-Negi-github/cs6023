#include <stdio.h>
#include <cuda.h>
#include <queue>
#include <chrono>



// HELPER FUNCTIONS

void printCPU(int *graph, int vertices){
    for(int i=0; i<vertices; ++i){
        for(int j=0; j<vertices; ++j){
            printf("%d ", graph[i * vertices + j]);
        }
        printf("\n");
    }
}

void printCPUIntArray(int *a, int vertices){
    for (int i=0; i<vertices; ++i){
        printf("%d ", a[i]);
    }
    printf("\n");
}

__global__ void printGPU(int *graph, int vertices){
    for(int i=0; i<vertices; ++i){
        for(int j=0; j<vertices; ++j){
            printf("%d ", graph[i * vertices + j]);
        }
        printf("\n");
    }
}

__global__ void printGPUInt(int *a){
    printf("%d\n", *a);
}

__global__ void printGPUIntArray(int *a, int vertices){
    for (int i=0; i<vertices; ++i){
        printf("%d ", a[i]);
    }
    printf("\n");
}

__global__ void printGPUBool(bool *a, int vertices){
    for (int i=0; i<vertices; ++i){
        printf("%d ", a[i]);
    }
    printf("\n");
}




// EDMONDS KARP MAX FLOW PARALLEIZED ALGORITHM

__global__ void kernel_find_augmenting_path(int vertices, int *residual_graph, bool *frontier, int *visited, int *previous,  bool *frontier_empty){

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(frontier[id]){

        frontier[id] = false;

        for(int v=0; v<vertices; ++v){

            if(v == id || residual_graph[id * vertices + v] <= 0) continue;

            if(atomicCAS(&visited[v], 0, 1) == 0){
                frontier[v] = true;
                previous[v] = id;
                *frontier_empty = false;
            }
        }
    }
}

bool find_augmenting_path(int vertices, int source, int sink, int *residual_graph, bool *visited, int *previous){
    
    memset(previous, -1, vertices * sizeof(int));
    memset(visited, false, vertices * sizeof(bool));
	
	std::queue<int> queue;

	queue.push(source);
    previous[source] = -1;
    visited[source] = true;

	while (!queue.empty())
	{
		int u = queue.front();
		queue.pop();

        if(u == sink){
            return true;
        }

		for (int v=0; v<vertices; ++v)
		{
			if (u != v && residual_graph[u * vertices + v] > 0 && !visited[v])
			{
				queue.push(v);
                visited[v] = true;
                previous[v] = u;
			}
		}
    }

    return false;
}

__global__ void kernel_update_residual_graph(int vertices, int *residual_graph, int *path, int *min_flow){
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    int v = path[id];
    int u = path[id + 1];

    residual_graph[u * vertices + v] -= *min_flow;
    residual_graph[v * vertices + u] += *min_flow;
}

void update_residual_graph(int vertices, int *residual_graph, int *path, int path_length, int min_flow){
    
    for(int i=0; i<path_length - 1; ++i){
        
        int v = path[i];
        int u = path[i + 1];

        residual_graph[u * vertices + v] -= min_flow;
        residual_graph[v * vertices + u] += min_flow;
    }
}

__global__ void kernel_find_max_flow(int vertices, int source, int *graph, int *residual_graph, int *result){
    int v = blockIdx.x * blockDim.x + threadIdx.x;

    if(graph[source * vertices + v] > 0){
        atomicAdd(result, graph[source * vertices + v] - residual_graph[source * vertices + v ]);
    }
}

void find_max_flow(int vertices, int source, int *graph, int *residual_graph, int *result){
   
    for(int v=0; v<vertices; ++v){
        if(graph[source * vertices + v] > 0){
            *result += graph[source * vertices + v] - residual_graph[source * vertices + v ];
        }
    }
}

int * sequential_edmonds_karp(int vertices, int edges, int source, int sink, int *graph, int *result){

    if (source == sink) {
        *result = -1;
        return graph;
    }

    int *residual_graph = (int *) malloc(vertices * vertices * sizeof(int));

    for(int i=0; i<vertices; ++i){
        for(int j=0; j<vertices; ++j){
            residual_graph[i * vertices + j] = graph[i * vertices + j];
        }
    }

    bool *visited = (bool *) malloc(vertices * sizeof(bool));
    int *previous = (int *) malloc(vertices * sizeof(int));
    int *path = (int *) malloc(vertices * sizeof(int));

    // int count = 0;

    while(find_augmenting_path(vertices, source, sink, residual_graph, visited, previous)){

        // printf("Iteration: %d\n", ++count);
        
        int curr = sink;
        int prev;
        int path_length = 0;
        int min_flow = INT_MAX;

        path[path_length++] = sink;

        // Store Path & Min Flow
        while((prev = previous[curr]) != -1 ){
            
            path[path_length++] = prev;
            
            if(residual_graph[prev * vertices + curr] < min_flow){
                min_flow = residual_graph[prev * vertices + curr];
            }

            curr = prev;
        }
        
        update_residual_graph(vertices, residual_graph, path, path_length, min_flow);
    }

    *result = 0;
    find_max_flow(vertices, source, graph, residual_graph, result);

    return residual_graph;
}

int * parallel_edmonds_karp(int vertices, int edges, int source, int sink, int *cpu_graph, int *cpu_result){

    if (source == sink) {
        *cpu_result = -1;
        return cpu_graph;
    }

    int *gpu_graph;
    int *cpu_residual_graph, *gpu_residual_graph;
    
    cpu_residual_graph = (int *) malloc(vertices * vertices * sizeof(int));

    cudaMalloc(&gpu_graph, vertices * vertices * sizeof(int));
    cudaMalloc(&gpu_residual_graph, vertices * vertices * sizeof(int));

    for(int i=0; i<vertices; ++i){
        for(int j=0; j<vertices; ++j){
            cpu_residual_graph[i * vertices + j] = cpu_graph[i * vertices + j];
        }
    }

    cudaMemcpy(gpu_graph, cpu_graph, vertices * vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_residual_graph, cpu_residual_graph, vertices * vertices * sizeof(int), cudaMemcpyHostToDevice);

    bool *gpu_frontier;
    int *gpu_visited;
    int *cpu_previous, *gpu_previous;
    bool *cpu_frontier_empty, *gpu_frontier_empty;
    int *cpu_min_flow, *gpu_min_flow;
    int *cpu_path_length, *gpu_path_length;
    int *cpu_path, *gpu_path;
    int *gpu_result;

    cpu_previous = (int *) malloc(vertices * sizeof(int));
    cpu_frontier_empty = (bool *) malloc(sizeof(bool));
    cpu_min_flow = (int *) malloc(sizeof(int));
    cpu_path_length = (int *) malloc(sizeof(int));
    cpu_path = (int *) malloc(vertices * sizeof(int));

    cudaMalloc(&gpu_frontier, vertices * sizeof(bool));
    cudaMalloc(&gpu_visited, vertices * sizeof(int));
    cudaMalloc(&gpu_previous, vertices * sizeof(int));
    cudaMalloc(&gpu_frontier_empty, sizeof(bool));
    cudaMalloc(&gpu_min_flow, sizeof(int));
    cudaMalloc(&gpu_path_length, sizeof(int));
    cudaMalloc(&gpu_path, vertices * sizeof(int));
    cudaMalloc(&gpu_result, sizeof(int));
    
    // int count = 0;

    do{

        // printf("Iteration: %d\n", ++count);

        cudaMemset(gpu_frontier, false, vertices * sizeof(bool));
        cudaMemset(gpu_visited, 0, vertices * sizeof(int));
        cudaMemset(gpu_previous, -1, vertices * sizeof(int));

        cudaMemset(&gpu_frontier[source], true, sizeof(bool));
        cudaMemset(&gpu_visited[source], 1, sizeof(int));

        *cpu_frontier_empty = false;

        while(!(*cpu_frontier_empty)){

            cudaMemset(gpu_frontier_empty, true, sizeof(bool));
            
            kernel_find_augmenting_path<<<vertices, 1>>>(vertices, gpu_residual_graph, gpu_frontier, gpu_visited, gpu_previous, gpu_frontier_empty);

            cudaMemcpy(cpu_frontier_empty, gpu_frontier_empty, sizeof(bool), cudaMemcpyDeviceToHost);
        }

        cudaMemcpy(cpu_previous, gpu_previous, vertices * sizeof(int), cudaMemcpyDeviceToHost);

        if(cpu_previous[sink] == -1){
            break;
        }

        int curr = sink;
        int prev;
        *cpu_path_length = 1;
        cpu_path[0] = sink;
        *cpu_min_flow = INT_MAX;

        // Store Path & Min Flow
        while((prev = cpu_previous[curr]) != -1 ){
            
            cpu_path[(*cpu_path_length)++] = prev;
            
            if(cpu_residual_graph[prev * vertices + curr] < *cpu_min_flow){
                *cpu_min_flow = cpu_residual_graph[prev * vertices + curr];
            }

            curr = prev;
        }

        cudaMemcpy(gpu_min_flow, cpu_min_flow, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_path, cpu_path, vertices * sizeof(int), cudaMemcpyHostToDevice);
    
        kernel_update_residual_graph<<<(*cpu_path_length) - 1, 1>>>(vertices, gpu_residual_graph, gpu_path, gpu_min_flow);

        cudaMemcpy(cpu_residual_graph, gpu_residual_graph, vertices * vertices * sizeof(int), cudaMemcpyDeviceToHost);
    
    }while(cpu_previous[sink] != -1);
    
    cudaMemset(gpu_result, 0, sizeof(int));
    kernel_find_max_flow<<<vertices, 1>>>(vertices, source, gpu_graph, gpu_residual_graph, gpu_result);
    cudaMemcpy(cpu_result, gpu_result, sizeof(int), cudaMemcpyDeviceToHost);

    return cpu_residual_graph;
}




// DININC'S MAX FLOW PARALLELIZED ALGORITHM

__global__ void kernel_find_level_path(int vertices, int sink, int *residual_graph, bool *frontier, int *level, int lev, bool *frontier_empty){

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(frontier[id]){
        
        frontier[id] = false;

        for(int v=0; v<vertices; ++v){

            if(v == id || residual_graph[id * vertices + v] <= 0) continue;

            if(atomicCAS(&level[v], -1, lev) == -1){
                frontier[v] = true;
                *frontier_empty = false;
            }        
        }
    }
}

bool find_level_path(int vertices, int source, int sink, int *residual_graph, int *level){
    
    memset(level, -1, vertices * sizeof(int));
	level[source] = 0;
	
	std::queue<int> queue;
	queue.push(source);

	while (!queue.empty())
	{
		int u = queue.front();
		queue.pop();

		for (int v=0; v<vertices; ++v)
		{
			if (u != v && residual_graph[u * vertices + v] > 0 && level[v] < 0)
			{
				level[v] = level[u] + 1;
				queue.push(v);
			}
		}
    }
	
    return level[sink] < 0 ? false : true ;
}

int sendFlow(int vertices, int u, int sink, int *residual_graph, int *level, int flow){

	if (u == sink) return flow;

	for (int v=0; v<vertices; ++v){

		if (residual_graph[(u * vertices) + v] > 0){

			if (level[v] == level[u]+1){

			 	int curr_flow = min(flow, residual_graph[u * vertices + v]);

			    int min_cap = sendFlow(vertices, v, sink, residual_graph, level, curr_flow);

			    if (min_cap > 0)
			    {
                    residual_graph[u * vertices + v] -= min_cap;
                    residual_graph[v * vertices + u] += min_cap;
				    return min_cap;
			    }
			}
		}
	}
	return 0;
}

int * sequential_dinic(int vertices, int source, int sink, int *graph, int *result){

    if (source == sink) {
        *result = -1;
        return graph;
    }

    int *residual_graph = (int *) malloc(vertices * vertices * sizeof(int));
    int *level = (int *) malloc(vertices * sizeof(int));

    for(int i=0; i<vertices; ++i){
        for(int j=0; j<vertices; ++j){
            residual_graph[i * vertices + j] = graph[i * vertices + j];
        }
    }
    
    *result = 0;

    // int count = 0;

	while (find_level_path(vertices, source, sink, residual_graph, level)){

        // printf("Iteration %d\n", ++count);

		while (int flow = sendFlow(vertices, source, sink, residual_graph, level, INT_MAX)){
            // printf("Flow: %d\n", flow);
            *result += flow;
        }
	}

    return residual_graph;
}

int * parallel_dinic(int vertices, int source, int sink, int *cpu_graph, int *cpu_result){

    if (source == sink) {
        *cpu_result = -1;
        return cpu_graph;
    }

    int *cpu_residual_graph, *gpu_residual_graph;
    int *cpu_level, *gpu_level;
    bool *gpu_frontier;
    bool *cpu_frontier_empty, *gpu_frontier_empty;
    
    cpu_residual_graph = (int *) malloc(vertices * vertices * sizeof(int));
    cpu_level = (int *) malloc(vertices * sizeof(int));
    cpu_frontier_empty = (bool *) malloc(sizeof(bool));

    cudaMalloc(&gpu_residual_graph, vertices * vertices * sizeof(int));
    cudaMalloc(&gpu_level, vertices * sizeof(int));
    cudaMalloc(&gpu_frontier, vertices * sizeof(bool));
    cudaMalloc(&gpu_frontier_empty, sizeof(bool));

    for(int i=0; i<vertices; ++i){
        for(int j=0; j<vertices; ++j){
            cpu_residual_graph[i * vertices + j] = cpu_graph[i * vertices + j];
        }
    }

    cudaMemcpy(gpu_residual_graph, cpu_residual_graph, vertices * vertices * sizeof(int), cudaMemcpyHostToDevice);
    
    *cpu_result = 0;

    // int count = 0;
    int empty_runs = 0;

    do{

        cudaMemset(gpu_frontier, false, vertices * sizeof(bool));
        cudaMemset(gpu_level, -1, vertices * sizeof(int));
        cudaMemset(&gpu_frontier[source], true, sizeof(bool));
        cudaMemset(&gpu_level[source], 0, sizeof(int));

        int lev = 1;
        *cpu_frontier_empty = false;
        
        while(!(*cpu_frontier_empty)){
            
            cudaMemset(gpu_frontier_empty, true, sizeof(bool));
            
            kernel_find_level_path<<<vertices, 1>>>(vertices, sink, gpu_residual_graph, gpu_frontier, gpu_level, lev, gpu_frontier_empty);

            cudaMemcpy(cpu_frontier_empty, gpu_frontier_empty, sizeof(bool), cudaMemcpyDeviceToHost);

            ++lev;
        }

        cudaMemcpy(cpu_level, gpu_level, vertices * sizeof(int), cudaMemcpyDeviceToHost);

        if(empty_runs == 2){
            memset(cpu_level, -1, vertices * sizeof(int));
            cpu_level[source] = 0;
            
            std::queue<int> queue;
            queue.push(source);

            while (!queue.empty())
            {
                int u = queue.front();
                queue.pop();

                for (int v=0; v<vertices; ++v)
                {
                    if (u != v && cpu_residual_graph[u * vertices + v] > 0 && cpu_level[v] < 0)
                    {
                        cpu_level[v] = cpu_level[u] + 1;
                        queue.push(v);
                    }
                }
            }
        }

        if(cpu_level[sink] == -1){
            break;
        }

        // printf("Iteration %d\n", ++count);

        while (int flow = sendFlow(vertices, source, sink, cpu_residual_graph, cpu_level, INT_MAX)){
            empty_runs = 0;
            // printf("Flow: %d\n", flow);
            *cpu_result += flow;
        }

        ++empty_runs;

        // printf("Total Flow: %d\n", *cpu_result);

        cudaMemcpy(gpu_residual_graph, cpu_residual_graph, vertices * vertices * sizeof(int), cudaMemcpyHostToDevice);

    }while(!(cpu_level[sink] == -1));

    return cpu_residual_graph;
}




// MAIN 

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
        cpu_graph[(u - 1) * vertices + (v - 1)] += c;
    }
    
    // Rigved's Part
    
    int *residual_graph;
    
    auto start1 = std::chrono::high_resolution_clock::now();
    residual_graph = sequential_edmonds_karp(vertices, edges, source - 1, sink - 1, cpu_graph, cpu_result);
    auto stop1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);
    printf("Max Flow is: %d.\n", *cpu_result);
    printf("Time taken by SEQUENTIAL EDONDS KARP is %lld microseconds.\n", duration1.count());

    auto start2 = std::chrono::high_resolution_clock::now();
    residual_graph = parallel_edmonds_karp(vertices, edges, source - 1, sink - 1, cpu_graph, cpu_result);
    auto stop2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);
    printf("Max Flow is: %d.\n", *cpu_result);
    printf("Time taken by PARALLELIZED EDONDS KARP is %lld microseconds.\n", duration2.count());
    
    auto start3 = std::chrono::high_resolution_clock::now();
    residual_graph = sequential_dinic(vertices, source - 1, sink - 1, cpu_graph, cpu_result);
    auto stop3= std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(stop3 - start3);
    printf("Max Flow is: %d.\n", *cpu_result);
    printf("Time taken by SEQUENTIAL DINIC is %lld microseconds.\n", duration3.count());

    auto start4 = std::chrono::high_resolution_clock::now();
    residual_graph = parallel_dinic(vertices, source - 1, sink - 1, cpu_graph, cpu_result);
    auto stop4 = std::chrono::high_resolution_clock::now();
    auto duration4 = std::chrono::duration_cast<std::chrono::microseconds>(stop4 - start4);
    printf("Max Flow is: %d.\n", *cpu_result);
    printf("Time taken by PARALLELIZED DINIC %lld microseconds.\n", duration4.count());

    fprintf(fout, "%d\n\n", *cpu_result);

    for(int i=0; i<vertices; ++i){
        for(int j=0; j<vertices; ++j){
            if(cpu_graph[i * vertices + j] > 0){
                int f = cpu_graph[i * vertices + j] - residual_graph[i * vertices + j];
                if(f < 0){
                    fprintf(fout, "%d ", 0);    
                }
                else{
                    fprintf(fout, "%d ", f);
                }
            }
            else{
                fprintf(fout, "%d ", 0);
            }
        }
        fprintf(fout, "\n");
    }

    fclose(fin);
    fclose(fout);
    return 0;
}