%%cu
#include<cuda.h>
#include<stdio.h>
#include<stdlib.h>

int main () {
    int nvertices, nedges, row, col, capacity, maxcapacity, source, destination;

    // Set number of vertices
    nvertices = 6;

    // Set number of edges 
    nedges = 8;

    // Set maximum capacity possible for an edge
    maxcapacity = 10;


    srand(time(0));
    FILE *filepointer;

    // Initializing filepointer with output file for creating graph
    filepointer = fopen("sample1.txt","w");

    fprintf(filepointer, "%d %d\n",nvertices,nedges);

    source = rand() % nvertices + 1;
    destination = rand() % nvertices + 1;

    fprintf(filepointer , "%d %d\n" , source, destination);


    int graph[nvertices][nvertices];

    for( int i=0; i < nvertices ; i++)
    {
        for( int j=0; j < nvertices ; j++)
        {
            graph[i][j] = 0;
        }
    }


    int edgesprepared = 0;
    while(edgesprepared < nedges)
    {
        row = rand()%nvertices + 1;
        col = rand()%nvertices + 1;
        capacity = rand()%(maxcapacity+1);
     
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