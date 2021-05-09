/* *********************************************    MAX FLOW ALGORITHM IMPLEMENTATION   *********************************************  */


// Including Required Header files 

#include <stdio.h>
#include <cuda.h>
#include <queue>
#include <chrono>



/* *********************************************    Helper Functions   *********************************************  */

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


void matrixprinter( int *matrix , int x, int y)
{
    /* A general function to print any 1d array /2d array  .
       Arguments :   If array is 1-d of size n    ======>  pass x = n and y = 0
                     If array is 2-d of size n*m  ======>  pass x = n and y = m
 
      ( However , I have not used any 2-d array in the code, but this is just for visualisation of adjacency matrix )
    */
 
    if(y != 0)                    // 2-d array
    {
        for(int i=0; i<x; i++)
        {
            for(int j=0; j<y; j++)
                printf("%d ",matrix[i*y + j]);
            printf("\n");
        }
    }
    else                         // 1-d array
    {
        for(int i=0; i<x; i++)
            printf("%d ",matrix[i]) ;
        printf("\n");
    }
 
}

// Function to print the flow in the edges
void final_edge_flow_printer(int *cpu_graph, int *cpu_res_graph, int vertices)
{
    /* Function to print the final edge flow in the edges of the graph
       Arguments : cpu_graph : Graph on the cpu
                   cpu_res_graph :  Residual graph on cpu
                   vertices : Number of vertices in the graph
    */
 
    for(int i=0; i<vertices; i++)
    {
      for(int j=0; j<vertices; j++)
      {
          int diff = cpu_graph[i*vertices+j] - cpu_res_graph[i*vertices+j] ;
          int maximum = diff>0 ? diff : 0;
          printf("%d ", maximum);
      }
      printf("\n");
    }
}


// Function to reverify that our algorithm has computed correct
void sink_flow_calc(int *cpu_graph, int *cpu_res_graph, int vertices , int sink)
{
    /* Function to compute the final flow to the sink vertex
       Arguments : cpu_graph : Graph on the cpu
                   cpu_res_graph :  Residual graph on cpu
                   vertices : Number of vertices in the graph
                    sink  : sink vertex
    */
 
   int total = 0 ;
    for(int i=0; i<vertices; i++)
    {      
            int diff = cpu_graph[i*vertices+sink] - cpu_res_graph[i*vertices+sink] ;
            int maximum = diff>0 ? diff : 0;
            total += maximum ;
             
    }
    printf("Total flow to the sink is %d\n", total) ;
}



/*  ***************************   EDMONDS KARP MAX FLOW PARALLELIZED ALGORITHM    ***************************  */
// EDMONDS KARP MAX FLOW PARALLEIZED ALGORITHM

__global__ void kernel_find_augmenting_path(int vertices, int *residual_graph, bool *frontier, int *visited, int *previous,  bool *frontier_empty){

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < vertices && frontier[id]){

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

__global__ void kernel_update_residual_graph(int vertices, int *residual_graph, int *path_length, int *path, int *min_flow){
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < *path_length - 1){

        int v = path[id];
        int u = path[id + 1];

        residual_graph[u * vertices + v] -= *min_flow;
        residual_graph[v * vertices + u] += *min_flow;

    }
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

    if(v < vertices && graph[source * vertices + v] > 0){
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
           
            kernel_find_augmenting_path<<<(ceil)((double)vertices / 1024), 1024>>>(vertices, gpu_residual_graph, gpu_frontier, gpu_visited, gpu_previous, gpu_frontier_empty);

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
        cudaMemcpy(gpu_path_length, cpu_path_length, sizeof(int), cudaMemcpyHostToDevice);
    
        kernel_update_residual_graph<<<(ceil)((double)((*cpu_path_length) - 1) / 1024), 1024>>>(vertices, gpu_residual_graph, gpu_path_length, gpu_path, gpu_min_flow);

        cudaMemcpy(cpu_residual_graph, gpu_residual_graph, vertices * vertices * sizeof(int), cudaMemcpyDeviceToHost);
    
    }while(cpu_previous[sink] != -1);
    
    cudaMemset(gpu_result, 0, sizeof(int));
    kernel_find_max_flow<<<(ceil)((double)vertices / 1024), 1024>>>(vertices, source, gpu_graph, gpu_residual_graph, gpu_result);
    cudaMemcpy(cpu_result, gpu_result, sizeof(int), cudaMemcpyDeviceToHost);

    return cpu_residual_graph;
}



/*  *******************************   DININC'S MAX FLOW PARALLELIZED ALGORITHM    *******************************  */
// DININC'S MAX FLOW PARALLELIZED ALGORITHM

__global__ void kernel_find_level_path(int vertices, int sink, int *residual_graph, bool *frontier, int *level, int lev, bool *frontier_empty){

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id < vertices && frontier[id]){
        
        frontier[id] = false;

        for(int v=0; v<vertices; ++v){

            if(v == id || residual_graph[id * vertices + v] <= 0) continue;

            // if(atomicCAS(&level[v], -1, lev) == -1){
            //     frontier[v] = true;
            //     *frontier_empty = false;
            // }
            
            if(level[v] == -1){
                level[v] = lev;
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
            
            kernel_find_level_path<<<(ceil)((double)vertices/1024), 1024>>>(vertices, sink, gpu_residual_graph, gpu_frontier, gpu_level, lev, gpu_frontier_empty);

            cudaMemcpy(cpu_frontier_empty, gpu_frontier_empty, sizeof(bool), cudaMemcpyDeviceToHost);

            ++lev;
        }

        cudaMemcpy(cpu_level, gpu_level, vertices * sizeof(int), cudaMemcpyDeviceToHost);


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



/*  *********************************************    PUSH RELABEL --Sequential Max Flow Algorithm    *********************************************  */

// Function computing Maxflow sequentially (Without GPU)
int * sequential_push_relabel(int *cpu_graph, int *cpu_result,  int vertices,  int source, int sink )
{
    /*  This  function will compute the maxflow sequentially. All the task will be done on cpu
        Arguments : cpu_graph : graph on cpu
                    cpu_result = This will finally store the final maxflow computed by the algorithm
                    source : source vertex 
                    sink : sink vertex  
                    vertices  : Number of vertices in the graph
 
        Return Value : Residual Graph (in the form of matrix)
    */
 
    //printf("\n Function Invoked : Sequential Push Relabel \n");

    int *residual_graph ;
    int *height ;
    int *excess_flow ;
 
    residual_graph = (int *)malloc(vertices*vertices*sizeof(int));
    height = (int *)malloc(vertices*sizeof(int));
    excess_flow = (int *)malloc(vertices*sizeof(int));
    
    
    // Initializing the values in the residual graph
    for(int i=0; i<vertices; i++)
    {
        for(int j=0; j<vertices; j++)
        {
            residual_graph[i * vertices + j ] = cpu_graph[ i * vertices + j ] ;
        }
    }


    // Assigning height as 0 to all the vertices  . 
    for(int i=0; i<vertices; i++)
    {
        height[i] = 0;
    }

    // Source is initialized as height equal to number of vertices
    height[source] = vertices;


    // Initializing  excess flow of all the vertices as 0 . 
    for(int i=0; i<vertices; i++)
    {
        excess_flow[i] = 0;
    }


    // This for loop will push the flow in all the outgoing edges from the source. Also, flow equal to the capacity of edge will be pushed.
    for( int i = 0; i<vertices; i++)
    {
        
        if(cpu_graph[source*vertices+i] > 0)
        {
            int k = cpu_graph[source*vertices+i] ;

            residual_graph[source*vertices + i] = 0;
          
            residual_graph [i*vertices + source] =cpu_graph[i*vertices+source] +  k;
          
            excess_flow[i] = k;

        }
    }

 
    int flag  = 1;
 
    while (flag)
    {
   
        int push_flag = 0, relabel_flag = 0;
     
        for(int i=0; i < vertices; i++)
        {
            int min_height_neighbour = INT_MAX;
          
            if( i != source && i!= sink && excess_flow[i] > 0 )
            {
                for(int j =0; j<vertices; j++)
                {
                    if( i!=j && residual_graph[i*vertices+j] > 0)
                    {
                        if (height[j] < height[i])
                        {
                            // do push operation on [i,j] ;
                         
                            push_flag = 1;
                            
                            int possible_flow = (residual_graph[i*vertices+j] < excess_flow[i] ? residual_graph[i*vertices+j] : excess_flow[i]);
                         
                            excess_flow[i] -= possible_flow;
                         
                            excess_flow[j] += possible_flow;
                         
                            residual_graph[i*vertices+j] -= possible_flow;
                         
                            residual_graph[j*vertices+i] += possible_flow;
                         
                            break;
                        }
                        if( height[j] < min_height_neighbour)
                        {
                            min_height_neighbour = height[j] ;
                         
                            relabel_flag = 1;
                        }
                    }
                }
             
              if (push_flag == 0 &&  relabel_flag == 1)
                {

                    // Do relabel operation on vertex i
                 
                    height[i] = min_height_neighbour + 1;
                    
                }
                
             if(push_flag == 1 || relabel_flag == 1)
                  break;
             
            }
            
        }
        if (push_flag == 1 || relabel_flag == 1)
                continue;
        else
          {
                flag = 0;
          }
     
    }

    //printf("Final edge flow in graph:\n");
    //final_edge_flow_printer(cpu_graph, residual_graph, vertices);
 
    *cpu_result = excess_flow[sink];
    return residual_graph ;
}



/*  *********************************************    PUSH RELABEL --Parallelized Max Flow Algorithm    *********************************************  */

// Push Relabel Kernel on the GPU
__global__ void kernel_push_relabel(int *gpu_res_graph, int *gpu_node_excess_flow, int *gpu_height, int vertices , int source, int sink)
{
    
    /*  Push Relabel kernel  : This will be envoked with number of threads equal to number of vertices in the graph.
        Arguments :   gpu_res_graph : Residual graph on the GPU  
                      gpu_node_excess_flow : Excess flow remaining on vertices  
                      gpu_height : height of the vertices 
                      vertices : Number of vertices in the graph
                      source : source vertex
                      sink : sink vertex
    */
 
    //printf("\n Kernel invoked : PUSH Relabel --Parallelized version : \n");


    // Calculating the id of the thread
    int id = blockIdx.x * blockDim.x + threadIdx.x ;
    
    //Below if block will be executed by thread other than source and sink vertex
    if( id!= source && id!= sink )
    {
        // Every thread reaches inside executes the below while block for cycle times ( can be adjusted ).
        int cycle = vertices;
     
        while (cycle > 0)
        {
            
            if ( gpu_node_excess_flow[id] > 0 && gpu_height[id]<vertices)
            {
                // h_prime will basically contains height of neighbour with minimum height .
                // The vertex corresponding to h_prime is v_prime
                // e_prime will contain the excess flow of the vertex corresponding to thread
             
                int e_prime = gpu_node_excess_flow[id] ;
                int h_prime = INT_MAX;
                int v_prime ;
             
                for(int j=0; j<vertices; j++)
                    {
                        if( gpu_res_graph[id*vertices +j] > 0)
                        {
                            int temp_height = gpu_height[j];
                            if( temp_height < h_prime )
                            {
                                h_prime = temp_height ;
                                v_prime = j ;
                            }
                        }
                    }
                

                // if  smallest height neighbour has height smaller than vertex(corresponding to thread id) , then send the flow to that neighbour.
                if (gpu_height[id] > h_prime)
                {
                    int k = e_prime ;
                    k = e_prime < gpu_res_graph[id*vertices+v_prime] ? e_prime :gpu_res_graph[id*vertices+v_prime];
                 
                    atomicAdd(&gpu_res_graph[v_prime*vertices + id ], k);
                    atomicSub(&gpu_res_graph[id*vertices + v_prime ], k) ;
                    atomicAdd( &gpu_node_excess_flow[v_prime], k);
                    atomicSub( &gpu_node_excess_flow[id], k);                    
                }
                else             // otherwise update the height of vertex(id)
                {
                    gpu_height[id] = h_prime + 1;
                }
            }
         
            cycle--;
        }
    }  
} 


// Global relabelling step 1 kernel on GPU
__global__ void GR_step1_kernel( int *gpu_graph, int *gpu_node_excess_flow , int *gpu_res_graph, int *gpu_height, int vertices )
{
    /*  This is step 1 of Global Relabelling (Hence named GR_step1_kernel)
        This will be envoked with number of threads equal to number of vertices in the graph.
 
        Arguments : gpu_graph  : original graph on the the gpu
                    gpu_node_excess_flow : Excess flow remaining on vertices 
                    gpu_res_graph : Residual graph on the GPU 
                    gpu_height : height of the vertices 
                    vertices  : Number of vertices in the graph
    */
 
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if ( id < vertices)
    {
        for( int j = id+1 ; j<vertices; j++)
        {
          
                if(  (gpu_height[id] > gpu_height[j] + 1) && (gpu_graph[id*vertices+j] > 0 ) )
                {
                    gpu_node_excess_flow[id] = gpu_node_excess_flow[id] - gpu_res_graph[id*vertices+j];

                    atomicAdd(&gpu_node_excess_flow[j] , gpu_res_graph[id*vertices+j]);

                    atomicAdd(&gpu_res_graph[j*vertices+id] , gpu_res_graph[id*vertices+j]);

                    gpu_res_graph[id*vertices+j] = 0;
                }
                else if ( (gpu_height[j] > gpu_height[id] + 1 ) && (gpu_graph[j*vertices+id] > 0 ) ) 
                {
                    atomicSub(&gpu_node_excess_flow[j] , gpu_res_graph[j*vertices+id]);

                    gpu_node_excess_flow[id] = gpu_node_excess_flow[id] + gpu_res_graph[j*vertices+id];

                    gpu_res_graph[id*vertices+j] = gpu_res_graph[id*vertices+j] + gpu_res_graph[j*vertices+id];

                    atomicSub(&gpu_res_graph[j*vertices+id], gpu_res_graph[j*vertices+id]);
                } 
                
        }
    }
}



// BFS Parallelized kernel on the GPU
__global__ void GR_step2_kernel(int *gpu_res_graph, int *queue, int *duplicate_queue, int *visited, int *gpu_height, int vertices, int *reiterate, int sink)
{

   /*  This is step 2 of Global Relabelling (Hence named GR_step2_kernel) .
       This kernel is basically do BFS(breadth first search) in Residual Graph with starting vertex as sink. 
       This kernel will assign height labels to the vertex equal to the length of path from sink to that vertex .
 
       This will be envoked with number of threads equal to number of vertices in the graph.
 
        Arguments : gpu_res_graph : Residual graph on the GPU 
                    queue : This is an array which keeps track of the nodes in the current level of BFS. For those nodes value will be 1 , for other 0.
                    duplicate_queue : This is exactly same before starting of kernel (i.e. before every level) . 
                                      This is just for avoiding atomics as queue gets updated inside the kernel.
                    gpu_height : height of the vertices 
                    vertices  : Number of vertices in the graph
                    reiterate : This will be set ( = 1) if there is atleast a node in next level of BFS. i.e. tells whether to continue BFS or not.
                    sink : sink vertex
    */
    
	int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id < vertices)
  {
      if (duplicate_queue[id] == 1 && visited[id] == 0)
      {

            if(id == sink)
            gpu_height[sink] = 0;
            visited[id] = 1;

            queue[id] = 0;
       
            int neighbor_height = gpu_height[id] +1;

            for (int i = 0; i < vertices; i++) 
            {
                if(visited[i] == 0 && duplicate_queue[i] !=1)
                {
                      if(gpu_res_graph[i*vertices+id] > 0 )
                      {         
                            gpu_height[i] = neighbor_height;      
                            queue[i] = 1;
                            *reiterate= 1;
                      } 
                } 
            }
        }
    }
}



// Global Relabel Function on the CPU
void global_relabel(int *gpu_graph, int *cpu_node_excess_flow, int *gpu_node_excess_flow, int *gpu_res_graph, int *gpu_height, int *cpu_height, int *cpu_excess_total, int source , int sink, int vertices)
 {
    
    //printf("function invoked: global relabel\n");

    /* Global relabelling step 1 */
      GR_step1_kernel<<<vertices,1>>>( gpu_graph, gpu_node_excess_flow , gpu_res_graph, gpu_height, vertices );
      cudaDeviceSynchronize();


      // Copying changed contents from gpu to cpu
      cudaMemcpy(cpu_height, gpu_height, vertices*sizeof(int) , cudaMemcpyDeviceToHost);
      cudaMemcpy(cpu_node_excess_flow, gpu_node_excess_flow, vertices*sizeof(int) , cudaMemcpyDeviceToHost);


    /* Global Relabelling Step 2 : Assign new heights by  BFS from sink to all vertices*/

    // Creating visited array as we do in BFS, to track care of visited vertices. hvisited is for cpu and dvisited is for gpu
    // queue and duplicate queue will contain 1 value for vertex  present at a level in BFS . For other , it will be 0.
  
    int *dvisited , *hvisited , *queue, *duplicate_queue;
    
    // Allocating memory to the variables 
    hvisited = (int *)malloc(vertices*sizeof(int));
    cudaMalloc(&dvisited,vertices*sizeof(int));
  
    cudaMalloc(&queue,vertices*sizeof(int));
    cudaMalloc(&duplicate_queue,vertices*sizeof(int));

    // Initially all the vertices are unvisited
    for(int i=0; i<vertices; i++)
      {
            hvisited[i] = 0;
      }
  
    cudaMemcpy(dvisited, hvisited, vertices * sizeof(int), cudaMemcpyHostToDevice);
  
    // hvisited is set to 1 , because we are gonna start the BFS. Hence its like we have pushed starting point in queue. 
    hvisited[sink] = 1;
  
    cudaMemcpy(queue, hvisited, vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(duplicate_queue, hvisited, vertices * sizeof(int), cudaMemcpyHostToDevice);
    
    // Creating variable for tracking till when to iterate in the loop. reiterate = 1, if still more levels to go, otherwise 0.  reiterate for cpu. diterate for gpu.
    int *dreiterate, *hreiterate, *zero;
  
    hreiterate = (int *)malloc(sizeof(int));
    cudaMalloc(&dreiterate,sizeof(int));
  
    zero = (int *)malloc(sizeof(int));
    *zero = 0;
  
    // Initially we have to iterate.
    *hreiterate = 1;

    while(*hreiterate)
    {    
        cudaMemcpy(dreiterate,zero,sizeof(int),cudaMemcpyHostToDevice);

        // kernel call to compute BFS at a particular level. 
     
        GR_step2_kernel<<<vertices,1>>>(gpu_res_graph, queue, duplicate_queue, dvisited, gpu_height, vertices, dreiterate, sink);
        cudaDeviceSynchronize();
     
     
        cudaMemcpy(duplicate_queue, queue, vertices * sizeof(int), cudaMemcpyDeviceToHost);
        
        cudaMemcpy(cpu_height, gpu_height, vertices * sizeof(int), cudaMemcpyDeviceToHost);
     
        cudaMemcpy(hreiterate, dreiterate, sizeof(int), cudaMemcpyDeviceToHost);

    }


    cudaMemcpy(cpu_height,gpu_height,vertices*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(hvisited,dvisited,vertices*sizeof(int),cudaMemcpyDeviceToHost);



    /* Global Relabelling Step 3 : Setting excess flow of those vertices to 0 from where we no path to sink in residual graph */
  
    for(int i=0 ; i<vertices ; i++)
    {
            if( hvisited[i] == 0 )
            {
                    cpu_height[i] = vertices;
                    *cpu_excess_total -= cpu_node_excess_flow[i] ;
                    cpu_node_excess_flow[i] = 0;
            }
    }
    
}



int * push_relabel (int *cpu_graph, int *gpu_graph , int *cpu_result, int vertices,  int source, int sink)
{
    /*  This  function will call the push relabel function
 
        Arguments : cpu_graph : graph on cpu
                    gpu_graph  : graph on the the gpu
                    cpu_result : will point to location storing the final maxflow value
                    source : source vertex 
                    sink : sink vertex  
                    vertices  : Number of vertices in the graph

        Return value : Residual Graph (In the form of matrix)
    */


    //printf("\n Function Invoked : CPU push Relabel caller \n");

    int *cpu_res_graph, *gpu_res_graph;
    int *cpu_height , *gpu_height;
    int *cpu_node_excess_flow, *gpu_node_excess_flow;
    int *cpu_excess_total;

 
 
    cpu_res_graph = (int *)malloc(vertices*vertices*sizeof(int));
    cpu_height = (int *)malloc(vertices*sizeof(int));
    cpu_node_excess_flow = (int *)malloc(vertices*sizeof(int));
    cpu_excess_total = (int *)malloc(sizeof(int));
    
    cudaMalloc(&gpu_res_graph, vertices*vertices*sizeof(int));
    cudaMalloc(&gpu_height, vertices*sizeof(int));
    cudaMalloc(&gpu_node_excess_flow, vertices*sizeof(int));

    
    // Initializing the values in the residual graph
    for(int i=0; i<vertices; i++)
    {
        for(int j=0; j<vertices; j++)
        {
            cpu_res_graph[i * vertices + j ] = cpu_graph[ i * vertices + j ] ;
        }
    }


    // Assigning height as 0 to all the vertices  . 
    for(int i=0; i<vertices; i++)
    {
        cpu_height[i] = 0;
    }

    // Source is initialized as height equal to number of vertices
    cpu_height[source] = vertices;


    // Initializing  excess flow of all the vertices as 0 . 
    for(int i=0; i<vertices; i++)
    {
        cpu_node_excess_flow[i] = 0;
    }


    *cpu_excess_total = 0;

    // This for loop will push the flow in all the outgoing edges from the source. Also, flow equal to the capacity of edge will be pushed.
    for( int i = 0; i<vertices; i++)
    {
        
        if(cpu_graph[source*vertices+i] > 0)
        {
            int k = cpu_graph[source*vertices+i] ;

            cpu_res_graph[source*vertices + i] = 0;
          
            cpu_res_graph [i*vertices + source] =cpu_graph[i*vertices+source] +  k;
          
            cpu_node_excess_flow[i] = k;
          
            *cpu_excess_total += k;

        }
    }
    
    // Copying the contents from cpu to gpu
    cudaMemcpy(gpu_res_graph, cpu_res_graph, vertices*vertices*sizeof(int) , cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_node_excess_flow, cpu_node_excess_flow, vertices*sizeof(int) , cudaMemcpyHostToDevice);
    

    while ( cpu_node_excess_flow[source] + cpu_node_excess_flow[sink] < *cpu_excess_total)
    {
        // Copying contents from cpu to gpu
        cudaMemcpy(gpu_height, cpu_height, vertices*sizeof(int) , cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_node_excess_flow, cpu_node_excess_flow, vertices*sizeof(int) , cudaMemcpyHostToDevice);


        // Push relabel Kernel call  to push the flow
        kernel_push_relabel<<<vertices,1>>>(gpu_res_graph, gpu_node_excess_flow, gpu_height, vertices , source, sink );
        cudaDeviceSynchronize();


        // Global relabel function call. It is also parallelized . Inside the function global relabel, kernel calls have been  made.
        global_relabel(gpu_graph, cpu_node_excess_flow, gpu_node_excess_flow, gpu_res_graph, gpu_height, cpu_height, cpu_excess_total, source , sink, vertices);

    }
    

    cudaDeviceSynchronize();
    cudaMemcpy( cpu_res_graph, gpu_res_graph, vertices * vertices * sizeof(int) , cudaMemcpyDeviceToHost) ;

    // Final maxflow reached at sink 
    *cpu_result = cpu_node_excess_flow[sink];
    
    return cpu_res_graph;
}


/* *********************************************  Main  ********************************************* */

int main(int argc, char **argv){
    

    FILE *fin, *fout;
    
    char *inputfilename = argv[1];
    //char *inputfilename = "sample.txt";
    fin = fopen( inputfilename , "r");
    
    if ( fin == NULL )  {
        printf( "%s failed to open.", inputfilename );
        return 0;
    }

    char *outputfilename = argv[2]; 
    //char *outputfilename = "output.txt";
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

    // Some condition check's that could save computation time 
    if(source == sink)
    {
      printf("source and sink are same, so no need to apply algorithm");
      return 0;
    }

    if(edges == 0)
    {
        printf("No edges in the graph . So maxflow = 0");
        return 0;
    }

    
    int *residual_graph ;
    

    // Sequential Edonds Karp 
    
    /*
    auto start1 = std::chrono::high_resolution_clock::now();
    residual_graph = sequential_edmonds_karp(vertices, edges, source - 1, sink - 1, cpu_graph, cpu_result);
    auto stop1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start1);
    printf("Max Flow is: %d.\n", *cpu_result);
    printf("Time taken by SEQUENTIAL EDONDS KARP is %lld microseconds.\n", duration1.count());
    printf("\n");
    */



    // Parallelized Edonds Karp
    
    /*
    auto start2 = std::chrono::high_resolution_clock::now();
    residual_graph = parallel_edmonds_karp(vertices, edges, source - 1, sink - 1, cpu_graph, cpu_result);
    auto stop2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);
    printf("Max Flow is: %d.\n", *cpu_result);
    printf("Time taken by PARALLELIZED EDONDS KARP is %lld microseconds.\n", duration2.count());
    printf("\n");
    */


    // Sequential Dinic 
    
    /*
    auto start3 = std::chrono::high_resolution_clock::now();
    residual_graph = sequential_dinic(vertices, source - 1, sink - 1, cpu_graph, cpu_result);
    auto stop3= std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(stop3 - start3);
    printf("Max Flow is: %d.\n", *cpu_result);
    printf("Time taken by SEQUENTIAL DINIC is %lld microseconds.\n", duration3.count());
    printf("\n");
    */



    // Parallelized Dinic
    
    /*
    auto start4 = std::chrono::high_resolution_clock::now();
    residual_graph = parallel_dinic(vertices, source - 1, sink - 1, cpu_graph, cpu_result);
    auto stop4 = std::chrono::high_resolution_clock::now();
    auto duration4 = std::chrono::duration_cast<std::chrono::microseconds>(stop4 - start4);
    printf("Max Flow is: %d.\n", *cpu_result);
    printf("Time taken by PARALLELIZED DINIC is %lld microseconds.\n", duration4.count());
    printf("\n");
    */


    // Sequential Push_relabel
    
    /*
    auto start5 = std::chrono::high_resolution_clock::now();
    residual_graph = sequential_push_relabel(cpu_graph, cpu_result, vertices, source-1, sink-1 ) ;
    auto stop5 = std::chrono::high_resolution_clock::now();
    auto duration5 = std::chrono::duration_cast<std::chrono::microseconds>(stop5 - start5);
    printf("Max Flow is: %d.\n", *cpu_result);
    printf("Time taken by PUSH RELABEL --Sequential version is %lld microseconds.\n", duration5.count());
    printf("\n");
    */



    // Parallelized Push Relabel
    
    
    auto start6 = std::chrono::high_resolution_clock::now();
    residual_graph = push_relabel(cpu_graph, gpu_graph , cpu_result, vertices,  source - 1, sink - 1);
    auto stop6 = std::chrono::high_resolution_clock::now();
    auto duration6 = std::chrono::duration_cast<std::chrono::microseconds>(stop6 - start6);
    printf("Max Flow is: %d.\n", *cpu_result);
    printf("Time taken by PUSH RELABEL --Parallelized version is  %lld microseconds.\n", duration6.count());
    printf("\n");

    
    // Print final maxflow computed in the output file 
    fprintf(fout, "%d\n\n", *cpu_result);
    

    // Print the flow between vertices in the output file . It will be printed in (vertices * vertices) size  matrix
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
