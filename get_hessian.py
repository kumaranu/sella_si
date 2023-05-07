from ase.io import read
from newtonnet.utils.ase_interface import MLAseCalculator
from ase import Atoms
import numpy as np


def get_hessian(atoms: Atoms, ml_path: str, config_path: str) -> np.ndarray:
    """Calculate the Hessian matrix of an ASE `Atoms` object using a
    machine learning model specified by `ml_path` and `config_path`.
    
    Args:
        atoms: An `Atoms` object.
        ml_path: Path to the machine learning model.
        config_path: Path to the configuration file for the machine learning model.
    
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


if __name__ == '__main__':
    atoms = read(f'inputs/irc_r.xyz')
    ml_path = 'inputs/best_model_state.tar'
    config_path = 'inputs/config0.yml'
    hessian = get_hessian(atoms, ml_path, config_path)
    print(hessian)

