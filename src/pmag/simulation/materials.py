import tidy3d as td
from tidy3d.plugins.dispersion import AdvancedFastFitterParam, FastDispersionFitter
import matplotlib.pyplot as plt
import numpy as np
import os
from pmag.config import PATH
import dill

def renamed_pole_residue(material, name):
    pole_args = {
        'name' : name,
        'frequency_range' : material.frequency_range,
        'allow_gain' : material.allow_gain,
        'nonlinear_spec' : material.nonlinear_spec,
        'modulation_spec' : material.modulation_spec,
        'viz_spec' : material.viz_spec,
        'heat_spec' : material.heat_spec,
        'eps_inf' : material.eps_inf,
        'poles' : material.poles,
    }
    new_mat = td.PoleResidue(**pole_args)
    return new_mat

def renamed_medium(material, name):
    """Create a copy of a td.Medium with a new name.

    Parameters
    ----------
    material : td.Medium
        The input medium to copy.
    name : str
        The new name for the medium.

    Returns
    -------
    td.Medium
        A new Medium instance with the given name and otherwise identical properties.
    """
    medium_args = {
        'name': name,
        'frequency_range': material.frequency_range,
        'allow_gain': material.allow_gain,
        'nonlinear_spec': material.nonlinear_spec,
        'modulation_spec': material.modulation_spec,
        'viz_spec': material.viz_spec,
        'heat_spec': material.heat_spec,
        'permittivity': material.permittivity,
        'conductivity': material.conductivity,
        'attrs': getattr(material, 'attrs', None)
    }
    new_medium = td.Medium(**medium_args)
    return new_medium


def load_if_exists(file_path):
    if not os.path.exists(file_path):
        return False, None
    with open(file_path, 'rb') as f:
        return True,dill.load(f)

def get_diamond_mat(wavelengths=[1.55], plot=True):
    file_path = PATH.materials / 'diamond_mat.dill'
    exists, diamond_mat = load_if_exists(file_path)
    if not exists:
        url = "https://refractiveindex.info/data_csv.php?datafile=database/data/main/C/nk/Phillip.yml"
        # note that additional keyword arguments to load_nk_file get passed to np.loadtxt
        fitter = FastDispersionFitter.from_url(url)
        min_wavelength, max_wavelength = np.min(wavelengths), np.max(wavelengths)
        if min_wavelength < 0.4 or max_wavelength > 2.1:
            raise ValueError("For accurate diamond material, wavelengths must be between 0.4 and 2.1 um")

        fitter = fitter.copy(update={"wvl_range": [0.4, 2.1]})
        advanced_param = AdvancedFastFitterParam(weights=(1, 1), num_iters=500, show_progress=True)
        diamond_mat, rms_error = fitter.fit(max_num_poles=3, advanced_param=advanced_param, tolerance_rms=0.01)

        with open(file_path, 'wb') as f:
            dill.dump(diamond_mat, f)
    if plot:
        plt.figure(figsize=(10, 5))
        plt.title('Diamond')
        fitter.plot(diamond_mat)
        plt.show()
    return diamond_mat

def get_nitride_mat(wavelengths=[1.55], plot=True):
    sin = td.material_library['Si3N4']['Luke2015Sellmeier']
    return sin
    
def get_oxide_mat(wavelengths=[1.55], plot=True):
    sio2 = td.material_library["SiO2"]["Palik_Lossless"]
    return sio2

def get_rich_sin_mat(wavelengths=[1.55], richness='mid', plot=True):
    rich_sin_filepath = PATH.materials / 'richsin_data' / f'{richness}_n.csv'
    file_path = PATH.materials / 'richsin_data' / f'{richness}_n.dill'
    exists, rich_sin_mat = load_if_exists(file_path)
    if not exists:
        rich_sin_fitter = FastDispersionFitter.from_file(rich_sin_filepath, delimiter=",")
        
        min_wavelength, max_wavelength = np.min(wavelengths), np.max(wavelengths)
        if min_wavelength < 0.4 or max_wavelength > 2.1:
            raise ValueError("For accurate rich silicon material, wavelengths must be between 0.4 and 2.1 um")

        rich_sin_fitter = rich_sin_fitter.copy(update={"wvl_range": [0.4, 1.6]})
        advanced_param = AdvancedFastFitterParam(weights=(1, 1), num_iters=200, show_progress=True)
        rich_sin_mat, rms_error = rich_sin_fitter.fit(max_num_poles=4, advanced_param=advanced_param, tolerance_rms=2e-3)
        with open(file_path, 'wb') as f:
            dill.dump(rich_sin_mat, f)
    
    if plot:
        plt.figure(figsize=(10, 5))
        plt.title(f'Rich Sin: {richness}')
        rich_sin_fitter.plot(rich_sin_mat)
        plt.show()
    return rich_sin_mat

def get_gap_mat(wavelengths=[1.55], plot=True):
    gap_url = 'https://refractiveindex.info/data_csv.php?datafile=database/data/main/GaP/nk/Bond.yml'
    gap_fitter = FastDispersionFitter.from_url(gap_url)

    min_wavelength, max_wavelength = np.min(wavelengths), np.max(wavelengths)
    if min_wavelength < 0.5 or max_wavelength > 2.1:
        raise ValueError("For accurate GaP material, wavelengths must be between 0.5 and 2.1 um")

    gap_fitter = gap_fitter.copy(update={"wvl_range": [0.5, 2.1]})
    advanced_param = AdvancedFastFitterParam(weights=(1, 1), num_iters=200, show_progress=True)
    gap_mat, rms_error = gap_fitter.fit(max_num_poles=3, advanced_param=advanced_param, tolerance_rms=2e-3)
    if plot:
        plt.figure(figsize=(10, 5))
        plt.title('GaP')
        gap_fitter.plot(gap_mat)
        plt.show()
    return gap_mat

def get_diamond_nonlinear(wavelengths=[1.55], plot=False):
    n_solid = 2.4

    # define the nonlinear parameters
    n_kerr_2 = 2e-8
    kerr_chi3 = 4 * (n_solid**2) * td.constants.EPSILON_0 * td.constants.C_0 * n_kerr_2 / 3
    chi3_model = td.NonlinearSpec(models=[td.NonlinearSusceptibility(chi3=kerr_chi3)], num_iters=10)
    kerr_solid = td.Medium(permittivity=n_solid**2, nonlinear_spec=chi3_model)
    if plot:
        plt.figure(figsize=(10, 5))
        plt.title('Diamond Nonlinear')
        kerr_solid.plot()
        plt.show()
    return kerr_solid
mat_funcs = {
    'diamond': get_diamond_mat,
    'diamond_nonlinear': get_diamond_nonlinear,
    'nitride': get_nitride_mat,
    'oxide': get_oxide_mat,
    'rich_sin_high': lambda wavelengths, plot: get_rich_sin_mat(wavelengths=wavelengths, richness='high', plot=plot),
    'rich_sin_mid': lambda wavelengths, plot: get_rich_sin_mat(wavelengths=wavelengths, richness='mid', plot=plot),
    'rich_sin_low': lambda wavelengths, plot: get_rich_sin_mat(wavelengths=wavelengths, richness='low', plot=plot),
    'gap': get_gap_mat,
    'air': lambda wavelengths=[1.55], plot=True: td.Medium(permittivity=1.0, name='air'),
    'PEC': lambda wavelengths=[1.55], plot=True: td.PECMedium(name='PEC'),
}

def init_materials(mat_names, mat_ids, wavelengths=[1.55], plot=True):
    materials = {'air': mat_funcs['air']()}
    for m, m_id in zip(mat_names, mat_ids):
        print(f"Initializing '{m_id}' material as '{m}'")
        loaded_mat = mat_funcs[m_id](wavelengths=wavelengths, plot=plot)
        if m_id == 'PEC':
            materials[m] = td.PECMedium(name=m)
        elif isinstance(loaded_mat, td.Medium):
            materials[m] = renamed_medium(loaded_mat, m)
        else:
            materials[m] = renamed_pole_residue(loaded_mat, m)
    return materials