import numpy as np
from model import model
from abcpy.probabilisticmodels import ProbabilisticModel, InputConnector
import argparse
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
            # run the model
            print(mpi_comm)
            mpi_comm = MPI.COMM_WORLD
            print(mpi_comm)
            print("will call the model")
            res = model(self.get_output_dimension(), mpi_comm, mean, stddev, seed)  
            # model outputs valid values only on rank 0  
            if mpi_comm.Get_rank() == 0:
                results.append(res)
        # reshape the results and broadcast them to all rank
        result = None
        if mpi_comm.Get_rank()==0:
            result = [np.array([results[i]]).reshape(-1,) for i in range(k)]
        result = mpi_comm.bcast(result)
        return result

class Distance():

    # distance between output vectors
    def distance(self, s1, s2):
        if not isinstance(s1, list):
            raise TypeError('Data is not of allowed types, should be a list')
        if not isinstance(s2, list):
            raise TypeError('Data is not of allowed types, should be a list')

        # compute distance between the statistics
        dist = np.zeros(shape=(s1.shape[0],s2.shape[0]))
        for ind1 in range(0, s1.shape[0]):
            for ind2 in range(0, s2.shape[0]):
                dist[ind1,ind2] = meanr(s1[ind1,:], s2[ind2,:])

        return dist.mean()

    def dist_max(self):
        return np.inf

def setup_backend(process_per_model):
    global backend
    from abcpy.backends import BackendMPI as Backend
    backend = Backend(process_per_model=process_per_model)


def infer_parameters():

    from abcpy.continuousmodels import Uniform
    mean = Uniform([[1], [10]], )
    stddev = Uniform([[0], [4]], )

    # define the model
    normal_model = Normal([mean, stddev])

    print("Will call forward simulate")

    fake_obs = normal_model.forward_simulate([2.0, 5.0], 1)

    print(fake_obs)

    #y_obs = [17, 54, 75, 161, 187, 202, 140, 87, 44, 17]

    # define distance
    #distance_calculator = Distance()

    # define sampling scheme
    #from abcpy.inferences import APMCABC
    #sampler = APMCABC([normal_model], [distance_calculator], backend, seed=1)
    #print('Sampling')
    #T, n_samples = 1, 2
    #eps_arr = np.array([10000])
    #journal = sampler.sample(y_obs, T, eps_arr, n_samples)

    #return journal
    return fake_obs

print("Hello from rank ", MPI.COMM_WORLD.Get_rank())

parser = argparse.ArgumentParser()
parser.add_argument("npm", help="number of mpi process per model", type=int)
args = parser.parse_args()
if args.npm >=MPI.COMM_WORLD.Get_size():
    raise "number of process per model must be lower than number of MPI process (one process is dedicated to the scheduler)"
#setup_backend(args.npm)
print(infer_parameters())
