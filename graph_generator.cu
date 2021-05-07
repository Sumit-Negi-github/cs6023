/* ********************************************  Graph Generator  ******************************************** */

#include<cuda.h>
#include<stdio.h>
#include<stdlib.h>
#define max_num_edge (nvertices*(nvertices-1))/2

int graph[10001][10001];

int main () {
    /*
          nvertices : Number of vertices in the graph
          nedges : Number of edges in the graph
          maxcapacity : maxcapacity possible for any edge
          source : source vertex in the graph
          sink : sink vertex in the graph
    */

    int nvertices, nedges, row, col, capacity, maxcapacity, source, sink;
    srand(time(0));

    // Set number of vertices

    nvertices = rand()%10001;                                 // For randomly selection of number of vertices
    //nvertices = 100;                                        // For manually setting the number of vertices


    // Set number of edges 

    nedges = rand() % ((nvertices*(nvertices-1)));             // For randomly selection of number of edges
    //nedges = 8;                                              // For manually setting the number of edges


    // Set maximum capacity possible for an edge
    maxcapacity = 10;                                          // Set maxcapacity possible for any edge


    printf("Number of vertices is :%d\n",nvertices);
    printf("Number of edges is :%d\n",nedges);
    

    FILE *filepointer;

    // Initializing filepointer with output file for creating graph
    filepointer = fopen("sample.txt","w");

    fprintf(filepointer, "%d %d\n",nvertices,nedges);

    source = (rand() % nvertices) + 1;
    sink = (rand() % nvertices) + 1;

    fprintf(filepointer , "%d %d\n" , source, sink);


    //int graph[nvertices][nvertices];

    for( int i=0; i <= nvertices ; i++)
    {
        for( int j=0; j <= nvertices ; j++)
        {
            graph[i][j] = 0;
        }
    }


    int edgesprepared = 0;
    while(edgesprepared < nedges)
    {
        row = (rand() % nvertices) + 1;
        col = (rand() % nvertices) + 1;
        capacity = ( rand()%(maxcapacity+1) ) + 1;
     
        if ( row != col && graph[row][col] == 0 )
        {
            fprintf(filepointer , "%d %d %d\n", row , col , capacity);
            graph[row][col] = 1;
            edgesprepared ++;
        }
     
    }

    fclose(filepointer);
    return 0;
}
