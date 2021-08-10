"""Get model's hyperparameters."""

import inspect


class HyperparametersGen():
    """
    Get hyperparameter from a model.
    Normally, these hyperparameters is configurated in __init__, fit and compile methods.
    
    Args:
    -------
        non_hyper: List<str>
            List of obligatory arguments which are not considered as hyperparameters.
    """

    non_hyper = ['X', 'y', 'X_train', 'y_train', 'X_val', 'y_val', 'self']

    def get_hyperparams(self, o):
        
        # Get args from __init__ function
        init_args = inspect.getargspec(o.__init__).args

        # Get args from fit function
        fit_args = inspect.getargspec(o.fit).args

        # Get args from compile function
        compile_args = inspect.getargspec(o.compile).args

        # Concatenate all arguments and remove the non_hyper elements & config object
        args = init_args + fit_args + compile_args
        hyper = [i for i in args if i not in self.non_hyper and not i.startswith("_")]

        return hyper