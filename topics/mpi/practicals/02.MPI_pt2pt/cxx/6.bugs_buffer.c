/****************************************************************
 *                                                              *
 * This file has been written as a sample solution to an        *
 * exercise in a course given at the CSCS Summer School.        *
 * It is made freely available with the understanding that      *
 * every copy of this file must include this header and that    *
 * CSCS take no responsibility for the use of the enclosed      *
 * teaching material.                                           *
 *                                                              *
 * Purpose: There is 2 bugs related to buffer                   *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/

/* There are three bugs in this application.
 * If compiled with cray compiler and ran with 4 ranks the program works
 * If 16 ranks are used the program fails (easy to find)
 * one bug is hidden and will not make the program fails (easy to find)
 * If compiled with Intel compiler, the program fails (hard to find)
 * DO NOT TRY TO FIX THE BUGS - YOU JUST HAVE TO IDENTIFY THEM
 */

/* NOTE: to compile with Intel just do:
 * module switch PrgEnv-cray PrgEnv-intel
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <vector>

#define MAX_SIZE_STRING 58
#define MAX_SIZE_COMPUTE (1<<20)

void compute_and_send(size_t size, float root, int k, MPI_Request* req)
{
    std::vector<float> data(size);
    int j;

    for(j = 0; j < size; j++) {
        data.push_back(rand()*root);
    }

    MPI_Isend(&data[0], size, MPI_FLOAT, k, 3, MPI_COMM_WORLD, req);
}


int main(int argc, char *argv[])
{
    int my_rank, np, k;
    float* root;
    MPI_Request* req;
    char* buffer;
    unsigned long* size;
    unsigned long rcv_size;
    float* data;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    buffer = (char*)malloc(MAX_SIZE_STRING*sizeof(char));

    if (my_rank == 0) {
        req = (MPI_Request*)malloc(3*(np-1)*sizeof(MPI_Request));
        root = (float*)malloc((np-1)*sizeof(float));
        size = (unsigned long*)malloc((np-1)*sizeof(unsigned long));

        /* Non-blocking send buffer */
        for( k = 1; k < np; k++ ) {
            root[k-1] = sqrt(k);
            sprintf(buffer, "Hello! This rank %d has the data computed with root %.4f.", k, root[k-1]);
            MPI_Isend(buffer, strlen(buffer), MPI_CHAR, k, 1, MPI_COMM_WORLD, &req[k-1]);

            size[k-1] = (unsigned long)(MAX_SIZE_COMPUTE*root[k-1]);
            MPI_Isend(&size[k-1], 1, MPI_UNSIGNED_LONG, k, 2, MPI_COMM_WORLD, &req[k-1+(np-1)]);

        }

        for( k = 1; k < np; k++ ) {
            compute_and_send(size[k-1], root[k-1], k, &req[k-1+2*(np-1)]);
        }

        MPI_Waitall(3*(np-1), req, MPI_STATUS_IGNORE);
        free(req);
        free(buffer);

    } else {
        // receive string
        MPI_Recv(buffer, MAX_SIZE_STRING, MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("%s\n",buffer);

        // receive size
        MPI_Recv(&rcv_size, 1, MPI_UNSIGNED_LONG, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("rank %d received size of %d\n",my_rank, rcv_size);
        data = (float*)malloc(rcv_size*sizeof(float));

        // receive data
        MPI_Recv(data, rcv_size, MPI_FLOAT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        free(data);
    }

    MPI_Finalize();
    return 0;
}
