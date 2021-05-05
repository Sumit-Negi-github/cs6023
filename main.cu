#include <stdio.h>
#include <cuda.h>
#include <queue>




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

__global__ void kernel_find_augmenting_path(int vertices, int *residual_graph, bool *frontier, bool *visited, int *previous, int *curr_sync_count, int *next_sync_count, int *ownership, int *temp_ID){

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(frontier[id]){

        frontier[id] = false;

        for(int v=0; v<vertices; ++v){

            if(v == id || residual_graph[id * vertices + v] <= 0) continue;

            if(!visited[v] && atomicCAS(&ownership[v], 0, 1) == 0){
                frontier[v] = true;
                previous[v] = id;
                visited[v] = true;
                atomicAdd(next_sync_count, 1);
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

void sequential_edmonds_karp(int vertices, int edges, int source, int sink, int *graph, int *result){

    if (source == sink) {
        *result = -1;
        return;
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
}

void parallel_edmonds_karp(int vertices, int edges, int source, int sink, int *cpu_graph, int *cpu_result){

    if (source == sink) {
        *cpu_result = -1;
        return;
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
    bool *gpu_visited;
    int *cpu_previous, *gpu_previous;
    int *cpu_curr_sync_count, *gpu_curr_sync_count; 
    int *cpu_next_sync_count, *gpu_next_sync_count;
    int *cpu_sync_vertex, *gpu_sync_vertex;
    bool *gpu_allow;
    int *cpu_min_flow, *gpu_min_flow;
    int *gpu_ownership;
    int *cpu_path_length, *gpu_path_length;
    int *cpu_path, *gpu_path;
    int *gpu_result;
    int *gpu_temp_id;

    cpu_previous = (int *) malloc(vertices * sizeof(int));
    cpu_curr_sync_count = (int *) malloc(sizeof(int));
    cpu_next_sync_count = (int *) malloc(sizeof(int));
    cpu_sync_vertex = (int *) malloc(sizeof(int));
    cpu_min_flow = (int *) malloc(sizeof(int));
    cpu_path_length = (int *) malloc(sizeof(int));
    cpu_path = (int *) malloc(vertices * sizeof(int));

    cudaMalloc(&gpu_frontier, vertices * sizeof(bool));
    cudaMalloc(&gpu_visited, vertices * sizeof(bool));
    cudaMalloc(&gpu_previous, vertices * sizeof(int));
    cudaMalloc(&gpu_curr_sync_count, sizeof(int));
    cudaMalloc(&gpu_next_sync_count, sizeof(int));
    cudaMalloc(&gpu_sync_vertex, sizeof(int));
    cudaMalloc(&gpu_allow, sizeof(bool));
    cudaMalloc(&gpu_min_flow, sizeof(int));
    cudaMalloc(&gpu_ownership, vertices * sizeof(int));
    cudaMalloc(&gpu_path_length, sizeof(int));
    cudaMalloc(&gpu_path, vertices * sizeof(int));
    cudaMalloc(&gpu_result, sizeof(int));
    cudaMalloc(&gpu_temp_id, sizeof(int));

    *cpu_sync_vertex = sink;
    *cpu_min_flow = INT_MAX;
    // printGPU<<<1, 1>>>(gpu_residual_graph, vertices);

    do{

        *cpu_curr_sync_count = 1;
        *cpu_next_sync_count = 0;
        *cpu_path_length = 1;
        cpu_path[0] = sink;

        cudaMemset(gpu_frontier, false, vertices * sizeof(bool));
        cudaMemset(gpu_visited, false, vertices * sizeof(bool));
        cudaMemset(gpu_previous, -1, vertices * sizeof(int));
        cudaMemset(gpu_ownership, 0, vertices * sizeof(int));

        cudaMemset(&gpu_frontier[source], true, sizeof(bool));
        cudaMemset(&gpu_visited[source], true, sizeof(bool));

        cudaMemcpy(gpu_curr_sync_count, cpu_curr_sync_count, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_min_flow, cpu_min_flow, sizeof(int), cudaMemcpyHostToDevice);

        
        // Find Shortest Augmenting Path
        while(*cpu_curr_sync_count){
            // printf("Iteration\n");
            cudaMemset(gpu_next_sync_count, 0, sizeof(int));
            cudaMemset(gpu_temp_id, 0, sizeof(int));
            
            kernel_find_augmenting_path<<<vertices, 1>>>(vertices, gpu_residual_graph, gpu_frontier, gpu_visited, gpu_previous, gpu_curr_sync_count, gpu_next_sync_count, gpu_ownership, gpu_temp_id);
            // next_count<<<vertices, 1>>>(gpu_frontier, gpu_next_sync_count);

            // cudaDeviceSynchronize();

            cudaMemcpy(cpu_curr_sync_count, gpu_next_sync_count, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(gpu_curr_sync_count, cpu_curr_sync_count, sizeof(int), cudaMemcpyHostToDevice);
        }

        cudaMemcpy(cpu_previous, gpu_previous, vertices * sizeof(int), cudaMemcpyDeviceToHost);

        if(cpu_previous[sink] == -1){
            break;
        }

        int curr = sink;
        int prev;

        // Store Path & Min Flow
        while((prev = cpu_previous[curr]) != -1 ){
            
            cpu_path[(*cpu_path_length)++] = prev;
            
            if(cpu_residual_graph[prev * vertices + curr] < *cpu_min_flow){
                *cpu_min_flow = cpu_residual_graph[prev * vertices + curr];
            }

            curr = prev;
        }

        // for(int i=0; i<*cpu_path_length; ++i){
        //     printf("%d ", cpu_path[i]);
        // }
        // printf("\nMin Flow: %d\n", *cpu_min_flow);

        cudaMemcpy(gpu_min_flow, cpu_min_flow, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_path_length, cpu_path_length, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_path, cpu_path, vertices * sizeof(int), cudaMemcpyHostToDevice);
    
        kernel_update_residual_graph<<<(*cpu_path_length) - 1, 1>>>(vertices, gpu_residual_graph, gpu_path, gpu_min_flow);

        cudaMemcpy(cpu_residual_graph, gpu_residual_graph, vertices * vertices * sizeof(int), cudaMemcpyDeviceToHost);

        // printCPU(cpu_residual_graph, vertices);
        // printf("\n\n");
    
    }while(cpu_previous[sink] != -1);
    
    cudaMemset(gpu_result, 0, sizeof(int));
    kernel_find_max_flow<<<vertices, 1>>>(vertices, source, gpu_graph, gpu_residual_graph, gpu_result);
    cudaMemcpy(cpu_result, gpu_result, sizeof(int), cudaMemcpyDeviceToHost);
}




// DININC'S MAX FLOW PARALLELIZED ALGORITHM

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

__global__ void kernel_find_level_path(int vertices, int sink, int *residual_graph, bool *frontier, bool *visited, int *level, int lev, bool *frontier_empty){

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(frontier[id]){
        
        frontier[id] = false;

        for(int v=0; v<vertices; ++v){

            if(v == id || residual_graph[id * vertices + v] <= 0) continue;

            if(!visited[v]){
                frontier[v] = true;
                visited[v] = true;
                level[v] = lev;
                *frontier_empty = false;
            }        
        }
    }
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

void sequential_dinic(int vertices, int source, int sink, int *cpu_graph, int *cpu_result){

    if (source == sink) {
        *cpu_result = -1;
        return;
    }

    int *gpu_graph;
    int *cpu_residual_graph, *gpu_residual_graph;
    int *cpu_level, *gpu_level;
    
    cpu_residual_graph = (int *) malloc(vertices * vertices * sizeof(int));
    cpu_level = (int *) malloc(vertices * sizeof(int));

    cudaMalloc(&gpu_graph, vertices * vertices * sizeof(int));
    cudaMalloc(&gpu_residual_graph, vertices * vertices * sizeof(int));
    cudaMalloc(&gpu_level, vertices * sizeof(int));

    for(int i=0; i<vertices; ++i){
        for(int j=0; j<vertices; ++j){
            cpu_residual_graph[i * vertices + j] = cpu_graph[i * vertices + j];
        }
    }
    
    *cpu_result = 0;

    // int count = 0;

	while (find_level_path(vertices, source, sink, cpu_residual_graph, cpu_level)){
		
        // printf("Iteration %d\n", ++count);

		while (int flow = sendFlow(vertices, source, sink, cpu_residual_graph, cpu_level, INT_MAX)){
            // printf("Flow: %d\n", flow);
            *cpu_result += flow;
        }
	}
}

void parallel_dinic(int vertices, int source, int sink, int *cpu_graph, int *cpu_result){

    if (source == sink) {
        *cpu_result = -1;
        return;
    }

    int *gpu_graph;
    int *cpu_residual_graph, *gpu_residual_graph;
    int *cpu_level, *gpu_level;
    bool *gpu_frontier;
    bool *gpu_visited;
    bool *cpu_frontier_empty, *gpu_frontier_empty;
    
    cpu_residual_graph = (int *) malloc(vertices * vertices * sizeof(int));
    cpu_level = (int *) malloc(vertices * sizeof(int));
    cpu_frontier_empty = (bool *) malloc(sizeof(bool));

    cudaMalloc(&gpu_graph, vertices * vertices * sizeof(int));
    cudaMalloc(&gpu_residual_graph, vertices * vertices * sizeof(int));
    cudaMalloc(&gpu_level, vertices * sizeof(int));
    cudaMalloc(&gpu_frontier, vertices * sizeof(bool));
    cudaMalloc(&gpu_visited, vertices * sizeof(bool));
    cudaMalloc(&gpu_frontier_empty, sizeof(bool));
    
    // *cpu_sync_vertex = sink;

    for(int i=0; i<vertices; ++i){
        for(int j=0; j<vertices; ++j){
            cpu_residual_graph[i * vertices + j] = cpu_graph[i * vertices + j];
        }
    }

    cudaMemcpy(gpu_residual_graph, cpu_residual_graph, vertices * vertices * sizeof(int), cudaMemcpyHostToDevice);
    
    *cpu_result = 0;

    // int count = 0;

    do{

        *cpu_frontier_empty = false;

        cudaMemset(gpu_frontier, false, vertices * sizeof(bool));
        cudaMemset(gpu_visited, false, vertices * sizeof(bool));

        cudaMemset(&gpu_frontier[source], true, sizeof(bool));
        cudaMemset(&gpu_visited[source], true, sizeof(bool));
        cudaMemset(gpu_level, -1, vertices * sizeof(int));
        cudaMemset(&gpu_level[source], 0, sizeof(int));

        int lev = 1;
        
        while(!(*cpu_frontier_empty)){
            
            cudaMemset(gpu_frontier_empty, true, sizeof(bool));
            
            kernel_find_level_path<<<vertices, 1>>>(vertices, sink, gpu_residual_graph, gpu_frontier, gpu_visited, gpu_level, lev, gpu_frontier_empty);

            cudaMemcpy(cpu_frontier_empty, gpu_frontier_empty, sizeof(bool), cudaMemcpyDeviceToHost);

            ++lev;
        }

        cudaMemcpy(cpu_level, gpu_level, vertices * sizeof(int), cudaMemcpyDeviceToHost);

        if(cpu_level[sink] < 0){
            break;
        }

        // printf("Iteration %d\n", ++count);

        while (int flow = sendFlow(vertices, source, sink, cpu_residual_graph, cpu_level, INT_MAX)){
            // printf("Flow: %d\n", flow);
            *cpu_result += flow;
        }

        cudaMemcpy(gpu_residual_graph, cpu_residual_graph, vertices * vertices * sizeof(int), cudaMemcpyHostToDevice);

    }while(!(cpu_level[sink] < 0));
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
        cpu_graph[(u - 1) * vertices + (v - 1)] += c;
        cpu_ff_graph[(u - 1) * vertices * 2 + (v - 1) * 2 + 0] += c;
        cpu_ff_graph[(u - 1) * vertices * 2 + (v - 1) * 2 + 1] += 0;
    }

    // cudaMemcpy(gpu_graph, cpu_graph, vertices * vertices * sizeof(int), cudaMemcpyHostToDevice);
    
    // Rigved's Part
    sequential_edmonds_karp(vertices, edges, source - 1, sink - 1, cpu_graph, cpu_result);
    // edmonds_karp(vertices, edges, source - 1, sink - 1, cpu_graph, cpu_result);
    // sequential_dinic(vertices, source - 1, sink - 1, cpu_graph, cpu_result);
    // parallel_dinic(vertices, source - 1, sink - 1, cpu_graph, cpu_result);

    // Sumit's Part
    // push_relabel<<<1, 1>>>(gpu_result);

    // Writing result back to CPU
    //cudaMemcpy(cpu_result, gpu_result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Max Flow is: %d\n", *cpu_result);

    fclose(fin);
    fclose(fout);
    return 0;
}