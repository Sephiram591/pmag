from scipy.constants import c, epsilon_0
from numpy.polynomial import Polynomial
from matplotlib import pyplot as plt
import numpy as np
import tidy3d as td
from tidy3d.plugins.mode import ModeSolver
from tqdm import tqdm

n2_vals = {
    'diamond': 0.082e-18,
}
def get_nonlinear_coeffs(n2, n0):
    """Returns chi3 and n2_bar"""
    n2_bar = n2/2*epsilon_0*n0*c
    x3 = 4/3*epsilon_0*c*n2*(n0**2)
    return x3, n2_bar

class NeffMixer:
    def __init__(self, freqs, neffs, num_modes, layer_widths, layer_height, meta_data):
        '''Initializes a NeffMixer object.
        inputs:
            freqs (Nx1 array): list of frequencies
            neffs (WxNxM array): list of effective indices
            num_modes (int): M, the number of modes
            layer_widths (Wx1 array): list of widths for the chosen layer
            layer_height (float): height of the chosen layer
            meta_data (dict(str, any)): dictionary of meta data
         '''
        self.freqs = freqs
        self.neffs = neffs
        self.num_modes = num_modes
        self.layer_widths = layer_widths
        self.layer_height = layer_height
        self.meta_data = meta_data

    def plot(self, mode_num=0):
        freqs_grid, wg_widths_grid = np.meshgrid(self.freqs, self.layer_widths)
        plt.figure()
        plt.pcolormesh(freqs_grid, wg_widths_grid, self.neffs[:,:,mode_num], shading='auto', cmap='bwr')
        plt.colorbar(label='neff')
        plt.xlabel('Frequency (THz)')
        plt.ylabel('Waveguide Width (um)')
        plt.title(f'Mode {mode_num} Effective Index')
        plt.show()

    def net_momentum_matrix(self, fixed_freqs, mode_factors, mode_nums=0, unit_ks=1, fit_tol=1e-3, poly_order=3, plot=False, num_freqs=100, center_colorbar=False):
        '''Returns the fwm net momentum matrix for mixing that targets the given output frequency.
        inputs:
            fixed_freqs ((K-2)x1 array): the frequencies of the fixed modes, where K is the number of modes involved in the mixing
            mode_factors (Kx1 array): list of mode factors, where the last two are the mode factors of the independent and dependent modes, respectively
            mode_nums (Kx1 array): list of mode numbers
            unit_ks (Kx1 array): indicates the unit vector magnitude in the direction of net momentum
            fit_tol (float): tolerance for the fit
            plot (bool): whether to plot the net momentum matrix
        outputs:
            f_A (Vx1 array): the frequencies of the independent mode, where V is the number of valid frequencies
            f_B (Vx1 array): the frequencies of the dependent mode, where V is the number of valid frequencies
            net_momentum_matrix (WxV array): the net momentum matrix, where W is the number of widths and L is the number of valid frequencies
        '''
        if np.isscalar(mode_nums):
            mode_nums = np.array([mode_nums]*len(mode_factors))
        if np.isscalar(unit_ks):
            unit_ks = np.array([unit_ks]*len(mode_factors))

        if len(fixed_freqs) != len(mode_factors) - 2:
            raise ValueError("The number of fixed frequencies must be equal to the number of modes minus 2")
        if np.max(mode_nums) > self.num_modes:
            raise ValueError(f"The requested mode number ({np.max(mode_nums)}) is too high for the number of modes in the mixer ({self.num_modes})")
        # Independent mode
        min_freq = np.min(self.freqs)
        max_freq = np.max(self.freqs)
        f_A = np.linspace(min_freq, max_freq, num_freqs)
        # Dependent mode
        f_B = -mode_factors[-1]*(np.sum(mode_factors[:-2]*fixed_freqs) + mode_factors[-2]*f_A)
        if plot:
            plt.figure()
            plt.plot(f_A/1e12, f_B/1e12)
            plt.vlines([min_freq/1e12, max_freq/1e12], min_freq/1e12, max_freq/1e12, color='k')
            plt.hlines([min_freq/1e12, max_freq/1e12], min_freq/1e12, max_freq/1e12, color='k')
            plt.xlabel('Pump A Frequency (THz)')
            plt.ylabel('Pump B Frequency (THz)')
            plt.title('Pump Frequencies')
            plt.legend(['Pumps', 'Mixer Boundary'])
            plt.show()
        # Crop the frequencies to the range of the mixer
        f_A = f_A[(f_B < np.max(self.freqs)) & (f_B > np.min(self.freqs))]
        f_B = f_B[(f_B < np.max(self.freqs)) & (f_B > np.min(self.freqs))]

        if len(f_A) == 0:
            raise ValueError("No valid frequencies found for the mixer")

        net_momentum_matrix = np.zeros((len(self.layer_widths), len(f_A)))
        for w_i, width in enumerate(self.layer_widths):
            k_fits = {}
            for m_n in mode_nums:
                if m_n not in k_fits:
                    k_fit, k_err = Polynomial.fit(self.freqs, 2*np.pi*self.freqs/td.C_0*self.neffs[w_i,:, m_n], poly_order, full=True)
                    k_fits[m_n] = k_fit
                    if np.abs(k_err[0]) > fit_tol:
                        print(f"Warning: k_err {np.abs(k_err[0][0]):.3e} is too large for mode {m_n}")
            # Add the fixed modes to the net momentum matrix
            fixed_iterable = zip(mode_nums[:-2], fixed_freqs, mode_factors[:-2], unit_ks[:-2])
            fixed_ks = np.array([k_fits[mode_num](fixed_freq)*mode_factor*unit_k for mode_num, fixed_freq, mode_factor, unit_k in fixed_iterable])
            net_momentum_matrix[w_i, :] = np.sum(fixed_ks)
            # Add the independent mode to the net momentum matrix
            independent_k = k_fits[mode_nums[-2]](f_A)*mode_factors[-2]*unit_ks[-2]
            net_momentum_matrix[w_i, :] += independent_k
            # Add the dependent mode to the net momentum matrix
            dependent_k = k_fits[mode_nums[-1]](f_B)*mode_factors[-1]*unit_ks[-1]
            net_momentum_matrix[w_i, :] += dependent_k
        
        if plot:
            plt.figure()
            plt.plot(f_A, independent_k)
            plt.plot(f_A, dependent_k)
            for k_i, fixed_k in enumerate(fixed_ks):
                plt.axhline(fixed_k, color='k', linestyle='--')
            plt.legend(['Independent Mode', 'Dependent Mode', 'Fixed Modes'])
            plt.xlabel('Pump A Frequency (THz)')
            plt.ylabel('Momentum')
            plt.title('Mode Momentum')
            plt.show()

            f_A_grid, widths_grid = np.meshgrid(f_A, self.layer_widths)
            plt.figure()
            if center_colorbar:
                vmin = -np.max(np.abs(net_momentum_matrix))
                vmax = np.max(np.abs(net_momentum_matrix))
            else:
                vmin = None
                vmax = None
            plt.pcolormesh(f_A_grid/1e12, widths_grid, net_momentum_matrix, cmap='bwr', shading='auto', vmin=vmin, vmax=vmax)
            plt.title('Net Momentum Matrix')
            plt.xlabel('Pump A Frequency (THz)')
            plt.ylabel('Width (um)')
            plt.colorbar()
            plt.show()

        return f_A, f_B, net_momentum_matrix

    def scale_neffs(self, new_height, plot=False, plot_mode_num=0):
        """
        Scale the frequencies and the widths to a new size, using the height of the layer at the given index
        inputs:
            new_height (float): the new height
        """
        scale_factor = new_height / self.layer_height

        self.layer_widths = self.layer_widths * scale_factor
        self.layer_height = self.layer_height * scale_factor
        self.freqs = self.freqs / scale_factor
        if plot:
            self.plot(plot_mode_num)

def get_neff_mixer(swept_layer_widths, layer_heights, freqs, num_modes, mode_args, chosen_layer_i=0, plotted_modes=0, plot_fields=False, plot_freq=None):
    '''Return a NeffMixer object for the given layer widths, heights, and frequencies
    inputs:
        swept_layer_widths (HxWx1 array): a list of swept widths for each layer
        layer_heights (Hx1 array): heights of the layers
        freqs (Nx1 array): list of frequencies in increasing order
        num_modes (int): number of modes to solve for
        mode_args (dict): dictionary of mode arguments for get_waveguide_modes
        chosen_layer_i (int): index of the layer to use for the mixer
        plotted_modes (int): number of modes to plot
        plot_fields (bool): whether to plot the fields
    outputs:
        neff_mixer (NeffMixer): the NeffMixer object
    '''
    neffs = np.zeros((np.shape(swept_layer_widths)[1], len(freqs), num_modes))
    f_min = freqs[0]
    if plot_freq is None:
        plot_freq = f_min

    if plot_fields:
        fig, axs = plt.subplots(np.shape(swept_layer_widths)[1], 2*plotted_modes, figsize=(9*plotted_modes, 4*np.shape(swept_layer_widths)[1]))

    for sweep_i in tqdm(range(np.shape(swept_layer_widths)[1])):
        layer_widths = swept_layer_widths[:, sweep_i]
        mode_data, mode_solver = get_waveguide_modes(layer_widths, layer_heights, freqs, num_modes, **mode_args)
        if plot_fields:
            for i in range(plotted_modes):
                mode_solver.plot_field("Ex", f=plot_freq, mode_index=i, ax=axs[sweep_i, 2*i])
                mode_solver.plot_field("Ez", f=plot_freq, mode_index=i, ax=axs[sweep_i, 2*i+1])
                axs[sweep_i,i].set_title(f"Width: {layer_widths[chosen_layer_i]:.2f} um, m={i+1}")
        neff = mode_data.n_eff
        neff = np.array(neff)
        neffs[sweep_i, :] = neff

    meta_data = {}
    meta_data['swept_layer_widths'] = swept_layer_widths
    meta_data['layer_heights'] = layer_heights
    meta_data['chosen_layer_i'] = chosen_layer_i
    meta_data['layer_mats'] = mode_args['layer_mats']
    meta_data['bend_radius'] = mode_args['bend_radius']
    meta_data['box_mat'] = mode_args['box_mat']
    meta_data['cladding'] = mode_args['cladding']
    meta_data['target_neff'] = mode_args['target_neff']

    mixer = NeffMixer(freqs, neffs, num_modes, swept_layer_widths[chosen_layer_i, :], layer_heights[chosen_layer_i], meta_data)
    for i in range(plotted_modes):
        mixer.plot(i)
    return mixer

def get_waveguide_modes(layer_widths, layer_heights, freqs, num_modes, layer_mats, bend_radius, material_library, box_mat='air', cladding='air', target_neff=None, plot_geom=True, return_fields=True, web_modes=False):
    '''Returns the modes for the given layer widths, heights, and frequencies
    inputs:
        layer_widths (Hx1 array): a list of swept widths for each layer
        layer_heights (Hx1 array): heights of the layers
        freqs (Nx1 array): list of frequencies
        num_modes (int): number of modes to solve for
        layer_mats (Hx1 array): materials of the layers
        bend_radius (float): radius of the bend
        material_library (dict): dictionary of materials
        box_mat (str): material of the box
        cladding (str): material of the cladding
        target_neff (float): target effective index
        plot_geom (bool): whether to plot the mode solver geometry
        return_fields (bool): whether to return the fields
    outputs:
        mode_data (ModeData): the mode data
        mode_solver (ModeSolver): the mode solver
    '''
    # central frequency
    fmin = freqs[0]
    fmax = freqs[-1]
    freq0 = (fmin + fmax) / 2
    wavelength = td.C_0/freq0
    # max_neff = np.max([material_library[mat].nk_model(wavelength)[0] for mat in layer_mats])


    # automatic grid specification
    run_time = 1e-12
    grid_spec = td.GridSpec.auto(min_steps_per_wvl=25, wavelength=td.C_0/freq0)

    bend_axis= 1
    structures = []
    total_height = np.sum(layer_heights)
    weighted_width = np.sum(layer_widths*layer_heights)/total_height


    Lx, Lz = 2+weighted_width, 2+total_height
    z_min = -total_height/2

    for i, (width, height, mat) in enumerate(zip(layer_widths, layer_heights, layer_mats)):
        z_center = z_min + height/2
        layer = td.Structure(
            geometry=td.Box(size=(width, td.inf, height), center=(bend_radius,0,z_center)),
            medium=material_library[mat],
        )
        structures.append(layer)
        z_min += height

    # Add the box
    box_thickness = (Lz)/2 + 2
    box_z = -total_height/2 - box_thickness/2
    box = td.Structure(
        geometry=td.Box(size=(Lx+1, td.inf, box_thickness), center=(bend_radius, 0, box_z)),
        medium=material_library[box_mat],
    )
    structures.append(box)


    sim = td.Simulation(
        size=(Lx, 1, Lz),
        center=(bend_radius,0,0),
        grid_spec=grid_spec,
        structures=structures,
        run_time=run_time,
        medium=material_library[cladding],
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
    )
    mode_spec = td.ModeSpec(
        num_modes=num_modes,
        target_neff=target_neff,
        bend_radius=bend_radius if bend_radius else None,
        bend_axis=bend_axis,
        track_freq='highest',
        filter_pol='te'
    )
    plane = td.Box(center=(bend_radius, 0, 0), size=(Lx, 0, Lz))
    mode_solver = ModeSolver(
        simulation=sim,
        plane=plane,
        mode_spec=mode_spec,
        freqs=freqs,
        fields=['Ex', 'Ez'] if return_fields else []
    )
    if plot_geom:
        mode_solver.plot()
    
    if web_modes:
        mode_data = td.web.run(mode_solver, "bend_modes")
    else:
        mode_data = mode_solver.solve()
    return mode_data, mode_solver