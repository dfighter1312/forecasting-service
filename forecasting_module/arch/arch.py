"""Generate a JSON file for storing hyperparameter model."""

from json import dumps
from params_gen.hypergen import HyperparametersGen
from params_gen.paramgen import ParametersGen
from models.base_model import BaseModel

class ArchitectureGen():
    """
    Generate the JSON file that shows the model's architecture.
    """

    def collect_param(self, o: BaseModel):
        
        # Call the 2 param/hyperparameters generation from the model
        param_gen = ParametersGen()
        hyper_gen = HyperparametersGen()
        
        # Get the parameters/hyperparameters
        param_dict = param_gen.get_params(o)     # This returns a dictionary
        hyper_lst = hyper_gen.get_hyperparams(o)     # This returns a hyperparam list

        # Concatenate the param_dict and hyper_lst
        arch_dict = param_dict
        arch_dict["hyper"] = hyper_lst

        return arch_dict

    def arch_gen(self, o: BaseModel):
        
        arch_dict = self.collect_param(o)

        # Comment out this line if you don't want to print the model architecture into the console
        print(arch_dict)

        # Encode the dictionary into JSON serialized string
        j = dumps(arch_dict)

        return j

        