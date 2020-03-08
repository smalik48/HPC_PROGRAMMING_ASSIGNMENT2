/**
 * @file    parallel_sort.h
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Declares the parallel sorting function.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

#ifndef PARALLEL_SORT_H
#define PARALLEL_SORT_H


#include <vector>
#include <algorithm>
#include <math.h>
#include <numeric>


#include <mpi.h>

using namespace std;

/**
 * @brief   Parallel, distributed sorting over all processors in `comm`. Each
 *          processor has the local input [begin, end).
 *
 * Note that `end` is given one element beyond the input. This corresponds to
 * the API of C++ std::sort! You can get the size of the local input with:
 * int local_size = end - begin;
 *
 * @param begin Pointer to the first element in the input sequence.
 * @param end   Pointer to one element past the input sequence. Don't access this!
 * @param comm  The MPI communicator with the processors participating in the
 *              sorting.
 */
void parallel_sort(int * begin, int* end, MPI_Comm comm);


/*********************************************************************
 *              Declare your own helper functions here               *
 *********************************************************************/

int random_number_generate(int total);

void swap(int&a, int &b);

int partition(int *begin, int *end, int pivot);

//void sub_array_create(int* begin, int* end, vector<int> &smaller_than_pivot, vector<int> &greater_than_pivot, int pivot);

void compute_srcounts(vector<int> &sendcnts, vector<int> &recvcnts, int barrier, vector<int> small_arr, vector<int> large_arr, vector<int> new_size, MPI_Comm comm, int check);

void compute_displace(vector<int> &sdispl, vector<int> &rdispl, const vector<int> &sendcnts, const vector<int> &recvcnts, MPI_Comm comm);

// ...

#endif // PARALLEL_SORT_H
