#include <iostream>
#include <string>
#include <random>
#include <chrono>
#include "mpi.h"

void model(int* result, int rsize, MPI_Comm communicator, double mean, double stddev, int seed) {
    // each process will roll 1000/p numbers
    // using a normal distribution
    //std::cout << "Hello from cpp model" << std::endl;
    int commSize, commRank;
    MPI_Comm_size(communicator, &commSize);
    MPI_Comm_rank(communicator, &commRank);
    //std::cout << "Hello from rank " << commRank << " out of " << commSize << std::endl;
    int nrolls = 1000/commSize;
    std::default_random_engine generator;
    generator.seed(seed);
    std::normal_distribution<double> distribution(mean, stddev);
    int *localRes = new int[rsize];
    //std::cout << "will enter the loop" << std::endl;
    for(int i=0; i<rsize; i++){
        localRes[i] = 0;
        //std::cout << "in loop" << std::endl;
    }
    for (int i=0; i<nrolls; ++i) {
        //std::cout << "in loop 2" << std::endl;
        double number = distribution(generator);
        if ((number>=0.0)&&(number<10.0)) ++localRes[int(number)];
    }
    // reduction of the values, result valid only on process 0
    //std::cout << "reduce" << std::endl;
    MPI_Reduce(localRes, result, rsize, MPI_INT, MPI_SUM, 0, communicator);
    //std::cout << "done" << std::endl;
    delete[] localRes;
}

int main(int argc, char* argv[]){
    MPI_Init(NULL, NULL);
    int size = 10;
    double mean = 5.0;
    double stddev = 2.0;
    int *y_obs = new int[size];
    int seed = std::chrono::system_clock::now().time_since_epoch().count();
    model(y_obs, size, MPI_COMM_WORLD, mean, stddev, seed);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank==0){
        for(int i=0; i<size; i++) std::cout << y_obs[i] << ", ";
        std::cout << std::endl;
    }
    MPI_Finalize();
}
