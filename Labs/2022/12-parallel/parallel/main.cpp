// Lab 12
// recuperare primi 30 minuti

// dopo la pausa mostra i codici per risolvere il problema con mpiexec

/*
#include <iostream>
#include <omp.h>

int
main(int argc, char ** argv)
{
  int n_threads;
  int thread_id;

  #pragma omp parallel shared(n_threads) private(thread_id)
  {
    thread_id = omp_get_thread_num();

  #pragma omp single
      n_threads = omp_get_num_threads();

  #pragma omp for schedule(dynamic)
      for(int i = 0; i < 4*n_threads; ++i)
        #pragma omp critical // da aggiungere sempre prima degli output in maniera da non rischiare che i processori si sovrappongano
        std::cout << "Iteration " << i << " run by thread " << thread_id << std::endl;
  }

  return 0;
}




#include <iostream>
#include <vector>
#include <omp.h>

int
main(int argc, char ** argv)
{
  int n_threads;
  int thread_id;

  int n = 10;
  std::vector<int> a(n), b(n);

  for(size_t i = 0; i < n; i++)
    {
      a[i] = i;
      b[i] = 2*i;
    }

  #pragma omp parallel shared(n_threads) private(thread_id)
  {
    thread_id = omp_get_thread_num();

  #pragma omp single
      n_threads = omp_get_num_threads();

  #pragma omp for
      for(size_t i = 0; i < n -1; ++i)
        {
          // if thread X executes i = 9 before thread Y executes i = 8
          a[i] = a[i+1] + b[i];
        }
//        std::cout << "Iteration " << i << " run by thread " << thread_id << std::endl;
  }

  for (auto x : a)
    std::cout << "" << std::endl;

  return 0;
}



#include <iostream>
#include <vector>
#include <omp.h>

int
main(int argc, char ** argv)
{
  // [0,1]
  const size_t n = 1e3;
  const double h = 1.0 / n;


  double sum = 0;

#pragma omp parallel for default(none)  reduction(+: sum)
  for(size_t i = 0; i < n; i++)
    {
      double x = h * (i - 0.5); // midpoint of i-th interval
      double y = 4.0 / ( 1 + x*x );

      sum += y;
    }



    double pi = h*sum;

  std::cout << pi << std::endl;

  return 0;
}




/// MPI



#include <iostream>
#include <vector>
#include <mpi.h>

int
main(int argc, char ** argv)
{
  MPI_Init(&argc, &argv);

  int mpi_rank;
  int mpi_size;

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  std::cout << "Hello world from rank " << mpi_rank << " out of " << mpi_size << std::endl;

  // Assume mpi_size = 2
  size_t n = 10;
  std::std::vector<int> to_send(n);
  std::std::vector<int> to_recive(n);



  if(mpi_size == 0)
    {
      std::fill(to_send.begin(), to_send.end(), 10);


      int destination = 1;
      int tag = 10;
      MPI_Send(to_send.data(), to_send.size(), MPI_INT, destination, tag, MPI_COMM_WORLD);
    }

  else
  {
    to_send == 100;


    MPI_Status status;
    // this is a struct containing:
    // - MPI_SOURCE
    // - MPI_TAG
    // - MPI_ERROR

    int source = 1;
    int tag = 10;
    MPI_Recv(to_recive.data(), to_recive.size(), MPI_INT, source, tag, MPI_COMM_WORLD, &status);
  }

  std::cout << "Rank " << mpi_rank << ": to_send = " << to_send << ", to_receive = " << to_receive << std::endl;

  MPI_Finalize();

  return 0;
}
*/

#include <iostream>
#include <vector>
#include <mpi.h>

int
main(int argc, char ** argv)
{
  MPI_Init(&argc, &argv);

  int mpi_rank;
  int mpi_size;

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);


  //assume mpi_size = 2
  // Eample of non-blcoking communication

  std::vector<int> to_send(100, mpi_rank);

  std::vector<int> to_recive(100);

  const int tag_send = (mpi_rank == 0) ? 10 : 20;
  const int tag_recv = (mpi_rank == 0) ? 20 : 10;

  const int partner_rank_id = 1 - mpi_rank; // 0 --> 1, 1 --> 2

  std::vector<MPI_Request> requests(mpi_size);


  MPI_Irecv(to_receive.data(),
            to_receive.size(),
            MPI_INT,partner_rank_id, MPI_COMM_WORLD, &(requests[0]));

  MPI_Isend(to_send.data(),
            to_send.size(),
            MPI_INT,partner_rank_id, MPI_COMM_WORLD, &(requests[1]));


  // perform more computations...
  int ready;
  MPI_Testall(mpi_size, &request, &ready, MPI_STATUS_IGNORE);

  std::cout << (ready ? "Completed" : "Not completed yet") << std::endl;

  // Now I need to access to to_receive

  std::std::vector<MPI_Status> statuses(mpi_size);
  MPI_Waitall(mpi_size, request.data(), statuses.data());

  // let's check again:
  MPI_Testall(mpi_size, requests.data(), &ready, MPI_STATUS_IGNORE);
  std::cout << (ready ? "Completed" : "Not completed yet") << std::endl;

  MPI_Finalize();

  return 0;
}
