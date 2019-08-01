import numpy as np
from model import model
from abcpy.probabilisticmodels import ProbabilisticModel, InputConnector
from mpi4py import MPI

def meanr(measures, compute):
    ss_res = dot((measures - compute),(measures - compute))
    ymean = mean(measures)
    ss_tot = dot((measures-ymean),(measures-ymean))
    return 1.0-ss_res/ss_tot

class Normal(ProbabilisticModel):

    def __init__(self, parameters, name='Normal'):
        # We expect input of type parameters = [mean, stddev]
        if not isinstance(parameters, list):
            raise TypeError('Input of normal model is of type list')

        if len(parameters) != 2:
            raise RuntimeError('Input list must be of length 2, containing [mean, stddev]')

        input_connector = InputConnector.from_list(parameters)
        super().__init__(input_connector, name)

    def _check_input(self, input_values):
        # Check whether input has correct type or format
        if len(input_values) != 2:
            raise ValueError('Number of parameters is 2')
        return True

    def _check_output(self, values):
        if not isinstance(values, np.ndarray):
            raise ValueError('Output of the model should be a numpy array')

        if values.shape[0] != 10:
            raise ValueError('Output shape should be of dimension 10')
        return True

    def get_output_dimension(self):
        return 10

    def forward_simulate(self, input_values, k, rng=np.random.RandomState(), mpi_comm=None):
        # do i k times
        results = []
        for i in range(k):
            seed = rng.randint(np.iinfo(np.int32).max)
            mean = input_values[0]
            stddev = input_values[1]
            mpi_comm = MPI.COMM_WORLD
            res = model(self.get_output_dimension(), mpi_comm, mean, stddev, seed)
            # model outputs valid values only on rank 0  
            if mpi_comm.Get_rank() == 0:
                results.append(np.array(res))
        # reshape the results and broadcast them to all rank
        result = None
        if mpi_comm.Get_rank()==0:
            result = [np.array([results[i]]).reshape(-1,) for i in range(k)]
        result = mpi_comm.bcast(result)
        return result

# define the model
from abcpy.continuousmodels import Uniform
mean = Uniform([[1], [10]], )
stddev = Uniform([[0], [4]], )
normal_model = Normal([mean, stddev])
fake_obs = normal_model.forward_simulate([5.0, 2.0], 1)
if MPI.COMM_WORLD.Get_rank()==0:
    print(fake_obs)



