/**
 * @file    parallel_sort.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements the parallel, distributed sorting function.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

#include "parallel_sort.h"
#include "utils.h"

// implementation of your parallel sorting
void parallel_sort(int * begin, int* end, MPI_Comm comm) {
  // set up MPI
  int rank, p;

  // get total size of processors and current rank
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &p);

  if (p == 1) {
        sort(begin, end);
        return;
    }

  /************************************************
  *  Broadcast the pivot                           *
  ***********************************************/

  // Get the size of the local array, and the size of the global array
    int cur_local_size = end - begin, total;
    MPI_Allreduce(&cur_local_size, &total, 1, MPI_INT, MPI_SUM, comm);

    /*********************************************************************
     *               Broadcast the pivot to each processor               *
     *********************************************************************/

    // Compute the global index of this processor
    int q = total / p; // get the quotient
    int r = total % p; // get the remainder
    int start;
    if (rank < r + 1) start = (q + 1) * rank;
    else start = (q + 1) * r + q * (rank - r);

    //random number generate
    int k = random_number_generate(total);

    // find the root bcasting pivot
    int root;
    if (r == 0) root = k / q;
    else if (k < r * (q + 1)) root = k / (q + 1);
    else root = r + (k - r * (q + 1)) / q;

    //bcast the pivot to all the processor
    int pivot = 0;
    if (k >= start && k < start + cur_local_size)
    	pivot = begin[k - start];
    MPI_Bcast(&pivot, 1, MPI_INT, root, comm);


  /*********************************************************************
     *          Transfer the data using All-to-all communication         *
     *********************************************************************/

  //create two subarrays, one less than pivot and one greater than pivot

  vector<int> smaller_than_pivot;
  vector<int> greater_than_pivot;
  sub_array_create(begin, end, smaller_than_pivot,greater_than_pivot,pivot);

  //run MPI_ALL_Gather for all the sizes of sub arrays to calculate m0 and m1

  vector<int> small_arr(p);
  vector<int> large_arr(p);
  int small = smaller_than_pivot.size();
  int large = greater_than_pivot.size();
  MPI_Allgather(&small, 1, MPI_INT, &small_arr[0], 1, MPI_INT, comm);
  MPI_Allgather(&large, 1, MPI_INT, &large_arr[0], 1, MPI_INT, comm);
  //sum the 2 gathered arrays to know m0 and m1
  int m0 = accumulate(small_arr.begin(), small_arr.end(), 0);
  int m1 = accumulate(large_arr.begin(), large_arr.end(), 0);

  /* Get the new allocation of p processors for m0 and m1 by putting a barrier between existing processor, small number will be sent to 0, 1, ...,
     * barrier - 1 processors, large number will be sent to the rest processors */

  int barrier = ceil(1.0 * p * m0 / total);
  //never allocate 0 processors to a problem
  if (barrier == 0) barrier++;
  if (barrier == p) barrier--;

  /*********************************************************************
     *          Transfer the data using All-to-all communication         *
     *********************************************************************/

  //Compute new array local size for each processor after barrier
  int new_local_size  = 0;
  if(rank<barrier){
    new_local_size = block_decompose(m0, barrier, rank);
  }
  else{
    new_local_size = block_decompose(m1, (p - barrier), (rank - barrier));
  }

  //all gather the new local size so that all the processor can calculate sendcounts and recv sendcounts
  // Compute the new local size vector
  vector<int> new_size(p);
  MPI_Allgather(&new_local_size, 1, MPI_INT, &new_size[0], 1, MPI_INT, comm);

  //compute send and recv sendcounts
  vector<int> sendcnts(p,0),recvcnts(p,0);
  //compute it for the elements smaller than pivot(update sendcnts and recvcnts)
  compute_srcounts(sendcnts, recvcnts, barrier, small_arr, new_size, comm, 0);
  //compute send and recieve displacements
  vector<int> sdispl(p, 0), rdispl(p, 0);
  compute_displace(sdispl, rdispl, sendcnts, recvcnts, comm);

  vector<int> rbuf;
  rbuf.resize(new_local_size);
  //all to all small transfered
  MPI_Alltoallv(&smaller_than_pivot[0], &sendcnts[0], &sdispl[0], MPI_INT, &rbuf[0], &recvcnts[0], &rdispl[0], MPI_INT, comm);

  //compute it for the elements larger than pivot(update sendcnts and recvcnts)
  sendcnts.clear();
  sendcnts.resize(p, 0);
  recvcnts.clear();
  recvcnts.resize(p, 0);
  compute_srcounts(sendcnts, recvcnts, barrier, large_arr, new_size, comm, 1);
  //compute send and recieve displacements
  sdispl.clear();
  rdispl.clear();
  sdispl.resize(p, 0);
  rdispl.resize(p, 0);
  compute_displace(sdispl, rdispl, sendcnts, recvcnts, comm);

  //all to all large transfered
  MPI_Alltoallv(&greater_than_pivot[0], &sendcnts[0], &sdispl[0], MPI_INT, &rbuf[0], &recvcnts[0], &rdispl[0], MPI_INT, comm);

  /*********************************************************************
     *            Create new communicator and recursively sort           *
     *********************************************************************/

  MPI_Comm comm_new;
  MPI_Comm_split(comm, (rank < barrier), rank, &comm_new);
  parallel_sort(&rbuf[0], &rbuf[0] + rbuf.size(), comm_new);
  MPI_Comm_free(&comm_new);

  /*********************************************************************
     *          Transfer the data using All-to-all communication         *
     *********************************************************************/
  //updating the actual address to make changes to the local_elements
  vector<int> tmp_sendcnts(p, 0), tmp_recvcnts(p, 0), tmp_sdispl(p, 0), tmp_rdispl(p, 0);
  vector<int> curr_size(p,0);
  MPI_Allgather(&cur_local_size, 1, MPI_INT, &curr_size[0], 1, MPI_INT, comm);
  compute_srcounts(tmp_sendcnts, tmp_recvcnts, barrier, new_size, curr_size, comm, 0);
  compute_displace(tmp_sdispl, tmp_rdispl, tmp_sendcnts, tmp_recvcnts, comm);
  // All-to-to communication
  MPI_Alltoallv(&rbuf[0], &tmp_sendcnts[0], &tmp_sdispl[0], MPI_INT, begin, &tmp_recvcnts[0], &tmp_rdispl[0], MPI_INT, comm);

}


/*********************************************************************
 *             Implement your own helper functions here:             *
 *********************************************************************/
 // Set the random seed
int random_number_generate(int total){
  srand(0);
  return (rand() % total);
}


//create subarray of greater and smaller values

void sub_array_create(int* begin, int* end, vector<int> &smaller_than_pivot, vector<int> &greater_than_pivot, int pivot){
  for(auto i = begin; i!=end;i++){
    if(*i<=pivot)smaller_than_pivot.push_back(*i);
    else greater_than_pivot.push_back(*i);
  }
}

void compute_srcounts(vector<int> &sendcnts, vector<int> &recvcnts, int barrier, vector<int> arr, vector<int> new_size, MPI_Comm comm, int check){
  int p, rank;
  MPI_Comm_size(comm, &p);
  MPI_Comm_rank(comm, &rank);

  //check whether the sendcnts are computed for larger or smaller array
  int j;
  //denote the recieving processors
  j = (check == 1)? barrier : 0;
  for(int i = 0 ; i < p ; i++ ){
    //start filling the arrays
    int send = arr[i];
    while(send > 0){
      int temp = (send <= new_size[j])? send : new_size[j];
      if(i == rank) sendcnts[j] = temp;
      if(j == rank) recvcnts[i] = temp;
      send -= temp;
      new_size[j] -= temp;
      if(new_size[j]==0)j++;
    }
  }
}

void compute_displace(vector<int> &sdispl, vector<int> &rdispl, const vector<int> &sendcnts, const vector<int> &recvcnts, MPI_Comm comm){
  int p, rank;
  MPI_Comm_size(comm, &p);
  MPI_Comm_rank(comm, &rank);

  for (int i = 1; i < p; i++) {
    sdispl[i] = sdispl[i - 1] + sendcnts[i - 1];
    rdispl[i] = rdispl[i - 1] + recvcnts[i - 1];
  }
}

// ...
