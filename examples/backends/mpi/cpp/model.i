%module model
%{
  #define SWIG_FILE_WITH_INIT

  #include "mpi.h"

  extern void model( int* result, int rsize, MPI_Comm communicator, double mean, double stddev, int seed);
%}

%include mpi4py.i
%mpi4py_typemap(Comm, MPI_Comm);

%include "numpy.i"

%init %{
  import_array();
%}

%apply (int* ARGOUT_ARRAY1, int DIM1) {(int* result, unsigned int rsize)};

extern void model( int* result, int rsize, MPI_Comm communicator, double mean, double stddev, int seed);
