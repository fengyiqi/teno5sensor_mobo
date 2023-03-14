import os
import torch
from objectives import TENO5Objectives
from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
import time
import warnings
from botorch import fit_gpytorch_mll
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning

os.system("rm -rf runtime_data")
os.system("rm -rf log.txt")
os.system("ulimit -s unlimited")

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

problem = TENO5Objectives(negate=False).to(**tkwargs)

def log(message: str):
    with open("log.txt", "a") as file:
        file.write(f"{message}\n")

def round_x(train_x):
    for i in range(train_x.shape[-1]):
        train_x[:, i] = train_x[:, i].round(decimals=problem._decimals[i])

NOISE_SE = torch.tensor([1e-6, 1e-6], **tkwargs)
torch.random.manual_seed(1)
def generate_initial_data(n=6):
    # generate training data
    train_x = draw_sobol_samples(bounds=problem.bounds,n=n, q=1).squeeze(1)
    for i in range(train_x.shape[-1]):
        train_x[:, i] = train_x[:, i].round(decimals=problem._decimals[i])
    train_obj = problem(train_x)
    return train_x, train_obj


def initialize_model(train_x, train_obj):
    # define models for objective and constraint
    train_x = normalize(train_x, problem.bounds)
    models = []
    for i in range(train_obj.shape[-1]):
        train_y = train_obj[..., i:i+1]
        train_yvar = torch.full_like(train_y, NOISE_SE[i] ** 2)
        models.append(
            FixedNoiseGP(train_x, train_y, train_yvar, outcome_transform=Standardize(m=1))
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model


BATCH_SIZE = 1
NUM_RESTARTS = 10
RAW_SAMPLES = 512

standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
standard_bounds[1] = 1


def optimize_qehvi_and_get_observation(model, train_x, train_obj, sampler):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    with torch.no_grad():
        pred = model.posterior(normalize(train_x, problem.bounds)).mean
                    
    partitioning = FastNondominatedPartitioning(
        ref_point=problem.ref_point, 
        Y=pred,
    )
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=problem.ref_point,
        partitioning=partitioning,
        sampler=sampler,
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    # observe new values 
    new_x =  unnormalize(candidates.detach(), bounds=problem.bounds)
    for i in range(new_x.shape[-1]):
        new_x[:, i] = new_x[:, i].round(decimals=problem._decimals[i])
    log(f"\tnew_x_qehvi {str(new_x)}")
    new_obj_true = problem(new_x)
    # new_obj = new_obj_true + torch.randn_like(new_obj_true) * NOISE_SE
    return new_x, new_obj_true


warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

N_BATCH = 20
MC_SAMPLES = 128

verbose = True

hvs_qehvi = []

# call helper functions to generate initial training data and initialize model

train_x_qehvi, train_obj_qehvi = generate_initial_data(n=2 * (problem.dim + 1))

mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)

# compute hypervolume
bd = DominatedPartitioning(ref_point=problem.ref_point, Y=train_obj_qehvi)
volume = bd.compute_hypervolume().item()

hvs_qehvi.append(volume)

# run N_BATCH rounds of BayesOpt after the initial random batch
for iteration in range(1, N_BATCH + 1):

    t0 = time.monotonic()

    # fit the models
    log(f"iteration {iteration}: fit_gpytorch_mll")
    fit_gpytorch_mll(mll_qehvi)

    # define the qEI and qNEI acquisition modules using a QMC sampler
    qehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

    # optimize acquisition functions and get new observations
    log(f"iteration {iteration}: optimize_qehvi_and_get_observation")
    new_x_qehvi, new_obj_qehvi = optimize_qehvi_and_get_observation( model_qehvi, train_x_qehvi, train_obj_qehvi, qehvi_sampler)
    # update training points
    train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
    train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj_qehvi])

    # update progress
        # compute hypervolume
    bd = DominatedPartitioning(ref_point=problem.ref_point, Y=train_obj_qehvi)
    volume = bd.compute_hypervolume().item()
    hvs_qehvi.append(volume)

    # reinitialize the models so they are ready for fitting on next iteration
    # Note: we find improved performance from not warm starting the model hyperparameters
    # using the hyperparameters from the previous iteration
    mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)

    t1 = time.monotonic()

    log(
        f"\nBatch {iteration:>2}: Hypervolume = "
        f"{hvs_qehvi[-1]:>4.2f}, "
        f"time = {t1-t0:>4.2f}."
    )