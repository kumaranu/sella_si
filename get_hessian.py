from ase.io import read
from newtonnet.utils.ase_interface import MLAseCalculator
from ase import Atoms
import numpy as np


def get_hessian(atoms: Atoms) -> np.ndarray:
    """Calculate the Hessian matrix of an ASE `Atoms` object using a
    machine learning model specified by `ml_path` and `config_path`
    as global variables.
    
    Args:
        atoms: An `Atoms` object.
    
    Returns:
        A 2D numpy array representing the Hessian matrix of the `atoms` object.
    """
    mlcalculator = MLAseCalculator(model_path=ml_path,
                                   settings_path=config_path)
    atoms.set_calculator(mlcalculator)
    mlcalculator.calculate(atoms)
    H = mlcalculator.results['hessian']
    n_atoms = np.shape(H)[0]
    return np.reshape(H, (n_atoms * 3, n_atoms * 3))


# Define global variables to provide the ML-funtion and the config file for it
ml_path = 'inputs/best_model_state.tar'
config_path = 'inputs/config0.yml'

if __name__ == '__main__':
    atoms = read(f'inputs/irc_r.xyz')
    hessian = get_hessian(atoms)
    print(hessian)

