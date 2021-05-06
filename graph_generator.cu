#include<cuda.h>
#include<stdio.h>
#include<stdlib.h>
#define max_num_edge (nvertices*(nvertices-1))/2

int graph[10001][10001];

int main () {
    int nvertices, nedges, row, col, capacity, maxcapacity, source, destination;
    srand(time(0));

    // Set number of vertices
    nvertices = (rand() % 10000) + 1;

    // Set number of edges 
    nedges = rand() % (nvertices*(nvertices-1));

    // Set maximum capacity possible for an edge
    maxcapacity = 10;

    printf("Number of vertices is :%d\n",nvertices);
    printf("Number of edges is :%d\n",nedges);
    
    FILE *filepointer;

    filepointer = fopen("sample12.txt","w");

    fprintf(filepointer, "%d %d\n", nvertices, nedges);

    source = (rand() % nvertices) + 1;
    destination = (rand() % nvertices) + 1;

    fprintf(filepointer , "%d %d\n" , source, destination);

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