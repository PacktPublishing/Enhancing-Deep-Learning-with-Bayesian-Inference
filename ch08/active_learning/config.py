import dataclasses


@dataclasses.dataclass
class Config:
    initial_n_samples: int
    n_total_samples: int
    n_epochs: int
    n_samples_per_iter: int
    # string representation of the acquisition function
    acquisition_type: str
    # number of mc_dropout iterations
    n_iter: int
