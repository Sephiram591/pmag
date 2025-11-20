from scipy.constants import c, epsilon_0
from numpy.polynomial import Polynomial
from matplotlib import pyplot as plt
import numpy as np
import tidy3d as td
from tidy3d.plugins.mode import ModeSolver
from tidy3d.components.simulation import AbstractYeeGridSimulation
from tidy3d.components.structure import Structure
from tidy3d.components.grid.grid import Grid, Coords
import xarray as xr
import matplotlib.cm as cm
from tqdm import tqdm

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
        min_n_matrix = np.zeros((len(self.layer_widths), len(f_A)))
        for w_i, width in enumerate(self.layer_widths):
            k_fits = {}
            n_fits = {}
            for m_n in mode_nums:
                if m_n not in k_fits:
                    n_fit, n_err = Polynomial.fit(self.freqs, self.neffs[w_i,:, m_n], poly_order, full=True)
                    n_fits[m_n] = n_fit
                    k_fits[m_n] = lambda f: 2*np.pi*f/td.C_0*n_fit(f)
                    if np.abs(n_err[0]) > fit_tol:
                        print(f"Warning: n_err {np.abs(n_err[0][0]):.3e} is too large for mode {m_n}")
            for m_n, n_fit in n_fits.items():
                if w_i == 0 or w_i == len(self.layer_widths) - 1:
                    plt.plot(self.freqs, n_fit(self.freqs), label=f'Mode {m_n}')
                    plt.scatter(self.freqs, self.neffs[w_i,:, m_n])
            if w_i == 0 or w_i == len(self.layer_widths) - 1:
                plt.legend()
                plt.show()
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
            # Record the smallest effective index involved in the mixing
            fixed_ns = np.array([n_fits[mode_num](fixed_freq) for mode_num, fixed_freq in zip(mode_nums[:-2], fixed_freqs)])
            independent_n = n_fits[mode_nums[-2]](f_A)
            dependent_n = n_fits[mode_nums[-1]](f_B)
            fixed_ns = np.tile(fixed_ns, (len(independent_n),1 ))
            all_ns = np.concatenate((fixed_ns, independent_n[:, np.newaxis], dependent_n[:, np.newaxis]), axis=1)
            min_n_matrix[w_i, :] = np.min(all_ns, axis=1)
        
        if plot:
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
            plt.figure()
            plt.pcolormesh(f_A_grid/1e12, widths_grid, min_n_matrix, cmap='hot', shading='auto')
            plt.title('Minimum Effective Index Matrix')
            plt.xlabel('Pump A Frequency (THz)')
            plt.ylabel('Width (um)')
            plt.colorbar()
            plt.show()

        return f_A, f_B, net_momentum_matrix, min_n_matrix

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

    def get_resonances(self, length, mode_num=0, plot=False, poly_order=3, fit_tol=2e-3):
        '''Returns the resonances for the given length
        inputs:
            length (float): the length of the resonator
        outputs:
            resonances (dict): Dictionary mapping width index to array of resonance frequencies and modes
                resonances[w_i]['resonance_freqs'] (array): the resonance frequencies
                resonances[w_i]['resonance_modes'] (array): the resonance modes
        '''

        # Calculate accumulated phase for each width and frequency
        # accumulated_phase = 2 * pi * n_eff * length / wavelength
        # wavelength = c / f

        # Ensure self.freqs is a 1D array of frequencies (Hz)
        freqs = np.array(self.freqs)
        widths = np.array(self.layer_widths)
        n_effs = np.array(self.neffs)  # shape: (num_widths, num_freqs, num_modes)

        # Only use the selected mode
        n_effs_mode = n_effs[:, :, mode_num]  # shape: (num_widths, num_freqs)

        # Calculate wavelength for each frequency
        wavelengths = c / freqs  # shape: (num_freqs,)

        # Calculate accumulated phase for each width and frequency
        # shape: (num_widths, num_freqs)
        accumulated_phase = 2 * np.pi * n_effs_mode * length / wavelengths[np.newaxis, :]

        # Mode number as a float
        resonance_number = accumulated_phase / (2 * np.pi)  # shape: (num_widths, num_freqs)

        resonances = {}

        if plot:
            plt.figure(figsize=(5, 4.5), dpi=200)
            colors = cm.viridis(np.linspace(0, 1, len(widths)))

        for w_i, width in enumerate(widths):
            # Find integer mode numbers within the range
            min_mode = np.ceil(np.min(resonance_number[w_i, :]))
            max_mode = np.floor(np.max(resonance_number[w_i, :]))
            resonance_modes = np.arange(min_mode, max_mode + 1)
            # Interpolate frequency at these integer mode numbers
            interp_func, interp_err = Polynomial.fit(resonance_number[w_i, :], freqs/1e12, poly_order, full=True)
            if np.abs(interp_err[0]) > fit_tol:
                print(f"Warning: interp_err {np.abs(interp_err[0][0]):.3e} is too large for width {width:.2f} um")
            resonance_freqs = interp_func(resonance_modes)*1e12
            resonances[w_i] = {}
            resonances[w_i]['resonance_freqs'] = resonance_freqs
            resonances[w_i]['resonance_modes'] = resonance_modes

            if plot:
                resonance_wavelengths = c / resonance_freqs
                resonance_neffs = resonance_modes/length*resonance_wavelengths
                plt.scatter(resonance_freqs / 1e12, resonance_neffs, color=colors[w_i], label=f'w = {width:.2f} $\\mu$m', s=1)
                # plt.plot(freqs/1e12, resonance_number[w_i, :], color=colors[w_i])
                # plt.scatter(resonance_freqs / 1e12, resonance_modes, color=colors[w_i], label=f'Width={width:.2f} um', s=1)

        if plot:
            plt.xlabel('Resonance Frequency (THz)', fontsize=15)
            plt.ylabel('Effective Index', fontsize=15)
            # plt.title(f'Resonances vs Mode Number (length={length*1e6:.2f} um)', fontsize=15)
            plt.legend(title=f'Waveguide Width')
            plt.tight_layout()
            plt.show()

        return resonances
        
        

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
        print(f"Plotting fields at {plot_freq/1e12:.2f} THz")
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

def get_overlap_2D(mode_data, mode_freqs, mode_nums, mode_factors, unit_ks, n2_mask=None):
    '''Returns the overlap integral for the given mode numbers and frequencies
    inputs:
        mode_data (ModeData): the mode data
        mode_freqs (Kx1 array): the mode frequencies
        mode_nums (Kx1 array): the mode number to look up in mode_data
        mode_factors (Kx1 array): the mode factors, determining the sign in the nonlinear interation. -1 means the field will be conjugated
        unit_ks (Kx1 array): the unit vector magnitudes in the direction of the mode. 1 means the mode is propagating in the positive direction, 
                            -1 means the mode is propagating in the negative direction. This field will be conjugated in the overlap integral.
                            0 means the field is generated by the conjugate of the polarization of the other modes.
        n2_mask (): the n2 of the mode
    outputs:
        overlap_integral (np.complex64): the overlap integral of Ex
    '''
    polarization_generated_index = None
    field_product = None
    dA = None
    power_product = 1

    for mode_num, mode_freq, mode_factor, unit_k in zip(mode_nums, mode_freqs, mode_factors, unit_ks):
        if unit_k == 0:
            # This is a polarization generated mode
            if polarization_generated_index is None:
                polarization_generated_index = mode_num
            else:
                raise ValueError("Multiple polarization generated modes are not supported")
        else:
            Ex = mode_data.Ex.isel(y=0, mode_index=mode_num).sel(f=mode_freq).values
            if mode_factor*unit_k == -1:
                Ex = np.conj(Ex)
            if dA is None:
                xx, zz = np.meshgrid(mode_data.Ex.x, mode_data.Ex.z, indexing='ij')
                dxx = np.gradient(xx, axis=0)*1e-6
                dzz = np.gradient(zz, axis=1)*1e-6
                dA = dxx*dzz
            if field_product is None:
                field_product = Ex
            else:
                field_product *= Ex
            power_product *= np.sum(np.abs(Ex)**2*dA)
    if polarization_generated_index is not None:
        # Polarization = np.conj(field_product)
        field_product = np.abs(field_product)**2 # equal to field_product*np.conj(field_product)
        power_product *= np.sum(field_product*dA)

    if n2_mask is not None:
        n2_mask = n2_mask.isel(y=0).values
        field_product *= n2_mask

    # print(f'numerator: {np.sum(field_product*dA)}')
    # print(f'denominator: {np.sqrt(power_product)}')

    overlap_integral = np.sum(field_product*dA) / np.sqrt(power_product)

    return overlap_integral

def get_overlap_3D(mode_data, mode_freqs, mode_nums, mode_factors, unit_ks, n2_mask=None):
    '''Returns the overlap integral for the given mode numbers and frequencies
    inputs:
        mode_data (FieldDataset): the mode data
        mode_freqs (Kx1 array): the mode frequencies
        mode_nums (Kx1 array): the mode number to look up in mode_data
        mode_factors (Kx1 array): the mode factors, determining the sign in the nonlinear interation. -1 means the field will be conjugated
        unit_ks (Kx1 array): the unit vector magnitudes in the direction of the mode. 1 means the mode is propagating in the positive direction, 
                            -1 means the mode is propagating in the negative direction. This field will be conjugated in the overlap integral.
                            0 means the field is generated by the conjugate of the polarization of the other modes.
        n2_mask (): the n2 of the mode
    outputs:
        overlap_integral (np.complex64): the overlap integral of Ey
        overlap_integral_2D (np.complex64): the overlap integral of Ey in the yz plane
    '''
    polarization_generated_index = None
    field_product = None
    dA = None
    dV = None
    power_product = 1
    power_product_2D = 1

    for mode_num, mode_freq, mode_factor, unit_k in zip(mode_nums, mode_freqs, mode_factors, unit_ks):
        if unit_k == 0:
            # This is a polarization generated mode
            if polarization_generated_index is None:
                polarization_generated_index = mode_num
            else:
                raise ValueError("Multiple polarization generated modes are not supported")
        else:
            Ey = mode_data.Ey.sel(f=mode_freq).values
            if mode_factor*unit_k == -1:
                Ey = np.conj(Ey)
            if dA is None:
                xx, yy, zz = np.meshgrid(mode_data.Ey.x, mode_data.Ey.y, mode_data.Ey.z, indexing='ij')
                dxx = np.gradient(xx, axis=0)*1e-6
                dyy = np.gradient(yy, axis=1)*1e-6
                dzz = np.gradient(zz, axis=2)*1e-6
                dA = dyy*dzz
                dV = dxx*dyy*dzz
                # print(f"dA: {dA[0,0,0]}, dV: {dV[0,0,0]}")
            if field_product is None:
                field_product = Ey
            else:
                field_product *= Ey

            power_product *= np.sum(np.abs(Ey)**2*dV)
            power_product_2D *= np.sum(np.abs(Ey)**2*dA, axis=(1,2))
            # print(f"power_product: {power_product}, power_product_2D: {power_product_2D}")
            # print(f"field_product: {field_product[0,0,0]}")

    if polarization_generated_index is not None:
        # Polarization = np.conj(field_product)
        field_product = np.abs(field_product)**2 # equal to field_product*np.conj(field_product)
        power_product *= np.sum(field_product*dV)
        power_product_2D *= np.sum(field_product*dA, axis=(1,2))
        # print(f"power_product: {power_product}, power_product_2D: {power_product_2D}")
        # print(f"field_product: {field_product[0,0,0]}")
    if n2_mask is not None:
        n2_mask = n2_mask.values
        field_product *= n2_mask

    # print(f'numerator: {np.sum(field_product*dA)}')
    # print(f'denominator: {np.sqrt(power_product)}')

    overlap_integral = np.sum(field_product*dV) / np.sqrt(power_product)
    overlap_integral_2D = np.sum(field_product*dA, axis=(1,2)) / np.sqrt(power_product_2D)

    return overlap_integral, overlap_integral_2D

def get_waveguide_modes(layer_widths, layer_heights, freqs, num_modes, layer_mats, bend_radius, material_library, box_mat='air', cladding='air', target_neff=None, plot_geom=True, return_fields=True, return_n2=False, web_modes=False):
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
        return_n2 (bool): whether to return the n2 of the mode
        web_modes (bool): whether to run the modes in the web app
    outputs:
        mode_data (ModeData): the mode data
        mode_solver (ModeSolver): the mode solver
        n2 (np.complex64): the n2 of the mode, if return_n2 is True
    '''
    # central frequency
    fmin = freqs[0]
    fmax = freqs[-1]
    freq0 = (fmin + fmax) / 2


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
        center=(bend_radius if bend_radius else 0,0,0),
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
        fields=['Ex', 'Ey', 'Ez'] if return_fields else []
    )
    if plot_geom:
        mode_solver.plot()
    
    if web_modes:
        mode_data = td.web.run(mode_solver, "bend_modes")
    else:
        mode_data = mode_solver.solve()
    if not return_n2:
        return mode_data, mode_solver
    else:
        freq= np.mean(freqs)
        coords = td.Coords(x=mode_solver.grid_snapped.centers.x, y=[-1,1], z=mode_solver.grid_snapped.centers.z)
        grid = td.Grid(boundaries=coords)
        n2 = n2_on_grid(sim=sim, grid=grid, coord_key='centers', freq=freq)
        return mode_data, mode_solver, n2

def get_n2(structure: Structure, frequency: float, coords: Coords):
    '''Returns the n2 of the structure at the given frequency and coordinates'''
    
    n2_vals = {
        'diamond': 0.082e-18,
        'nitride': 0.250e-18,
        'air': 0.0,
        'oxide': 0.0,
        'reflector': 0.0,
        None : 0.0
    }
    return n2_vals[structure.medium.name]
    
def n2_on_grid(
        sim : AbstractYeeGridSimulation,
        grid: Grid,
        coord_key: str = "centers",
        freq: float = None,
        n2_fn: callable = get_n2,
    ) -> xr.DataArray:
    """Get array of permittivity at a given freq on a given grid.

    Parameters
    ----------
    sim : :class:`.AbstractYeeGridSimulation`
        Simulation object to get the permittivity from.
    grid : :class:`.Grid`
        Grid specifying where to measure the permittivity.
    coord_key : str = 'centers'
        Specifies at what part of the grid to return the permittivity at.
        Accepted values are ``{'centers', 'boundaries', 'Ex', 'Ey', 'Ez', 'Exy', 'Exz', 'Eyx',
        'Eyz', 'Ezx', Ezy'}``. The field values (eg. ``'Ex'``) correspond to the corresponding field
        locations on the yee lattice. If field values are selected, the corresponding diagonal
        (eg. ``eps_xx`` in case of ``'Ex'``) or off-diagonal (eg. ``eps_xy`` in case of ``'Exy'``) epsilon
        component from the epsilon tensor is returned. Otherwise, the average of the main
        values is returned.
    freq : float = None
        The frequency to evaluate the mediums at.
        If not specified, evaluates at infinite frequency.
    Returns
    -------
    xarray.DataArray
        Datastructure containing the relative permittivity values and location coordinates.
        For details on xarray DataArray objects,
        refer to `xarray's Documentation <https://tinyurl.com/2zrzsp7b>`_.
    """
    def make_n2_data(coords: Coords):
        """returns epsilon data on grid of points defined by coords"""
        arrays = (np.array(coords.x), np.array(coords.y), np.array(coords.z))
        n2_background = n2_fn(
            structure=sim.scene.background_structure, frequency=freq, coords=coords
        )
        shape = tuple(len(array) for array in arrays)
        n2_array = n2_background * np.ones(shape, dtype=np.float64)
        # replace 2d materials with volumetric equivalents
        for structure in sim.volumetric_structures:
            # Indexing subset within the bounds of the structure
                
            inds = structure.geometry._inds_inside_bounds(*arrays)

            # Get permittivity on meshgrid over the reduced coordinates
            coords_reduced = tuple(arr[ind] for arr, ind in zip(arrays, inds))
            if any(coords.size == 0 for coords in coords_reduced):
                continue

            red_coords = Coords(**dict(zip("xyz", coords_reduced)))
            n2_structure = n2_fn(structure=structure, frequency=freq, coords=red_coords)

            # Update permittivity array at selected indexes within the geometry
            is_inside = structure.geometry.inside_meshgrid(*coords_reduced)
            n2_array[inds][is_inside] = (n2_structure * is_inside)[is_inside]

        coords = dict(zip("xyz", arrays))
        return xr.DataArray(n2_array, coords=coords, dims=("x", "y", "z"))

    # combine all data into dictionary
    if coord_key[0] == "E":
        # off-diagonal components are sampled at respective locations (eg. `eps_xy` at `Ex`)
        coords = grid[coord_key[0:2]]
    else:
        coords = grid[coord_key]
    return make_n2_data(coords)
def n2_on_coords(
        sim : AbstractYeeGridSimulation,
        coords: Coords,
        freq: float = None,
        n2_fn: callable = get_n2,
    ) -> xr.DataArray:
    """Get array of permittivity at a given freq on a given grid.

    Parameters
    ----------
    sim : :class:`.AbstractYeeGridSimulation`
        Simulation object to get the permittivity from.
    coords : :class:`.Coords`
        Grid specifying where to measure the permittivity.
    coord_key : str = 'centers'
        Specifies at what part of the grid to return the permittivity at.
        Accepted values are ``{'centers', 'boundaries', 'Ex', 'Ey', 'Ez', 'Exy', 'Exz', 'Eyx',
        'Eyz', 'Ezx', Ezy'}``. The field values (eg. ``'Ex'``) correspond to the corresponding field
        locations on the yee lattice. If field values are selected, the corresponding diagonal
        (eg. ``eps_xx`` in case of ``'Ex'``) or off-diagonal (eg. ``eps_xy`` in case of ``'Exy'``) epsilon
        component from the epsilon tensor is returned. Otherwise, the average of the main
        values is returned.
    freq : float = None
        The frequency to evaluate the mediums at.
        If not specified, evaluates at infinite frequency.
    Returns
    -------
    xarray.DataArray
        Datastructure containing the relative permittivity values and location coordinates.
        For details on xarray DataArray objects,
        refer to `xarray's Documentation <https://tinyurl.com/2zrzsp7b>`_.
    """


    def make_n2_data(coords: Coords):
        """returns epsilon data on grid of points defined by coords"""
        arrays = (np.array(coords.x), np.array(coords.y), np.array(coords.z))
        n2_background = n2_fn(
            structure=sim.scene.background_structure, frequency=freq, coords=coords
        )
        shape = tuple(len(array) for array in arrays)
        n2_array = n2_background * np.ones(shape, dtype=np.float64)
        # replace 2d materials with volumetric equivalents
        for structure in sim.volumetric_structures:
            # Indexing subset within the bounds of the structure
            if n2_fn(structure=structure, frequency=freq, coords=coords) != 0:
                inds = structure.geometry._inds_inside_bounds(*arrays)

                # Get permittivity on meshgrid over the reduced coordinates
                coords_reduced = tuple(arr[ind] for arr, ind in zip(arrays, inds))
                if any(coords.size == 0 for coords in coords_reduced):
                    continue

                red_coords = Coords(**dict(zip("xyz", coords_reduced)))
                n2_structure = n2_fn(structure=structure, frequency=freq, coords=red_coords)

                # Update permittivity array at selected indexes within the geometry
                is_inside = structure.geometry.inside_meshgrid(*coords_reduced)
                n2_array[inds][is_inside] = (n2_structure * is_inside)[is_inside]

        coords = dict(zip("xyz", arrays))
        return xr.DataArray(n2_array, coords=coords, dims=("x", "y", "z"))

    return make_n2_data(coords)

def fpm_spectrum(resonances, nonlinear_env, net_momentum_env, w_i, plot=False, mlim_a=None):
    """Compute pump resonance pairs and output frequencies for four-wave mixing.

    Inputs:
        resonances (dict): output of get_resonances(). Use `resonances[w_i]`.
        nonlinear_env (dict): contains keys:
            - 'fixed_freqs': array-like of fixed mode frequencies (Hz) with length K-2
            - 'mode_factors': array-like of length K (integers, e.g. [+1, +1, -1, -1])
            - 'unit_ks': array-like of length K (direction indicators in {-1, 0, +1})
        net_momentum_env (dict): included for API symmetry; not used directly here.
        w_i (int): width index selecting which resonance set to use.
        plot (bool): if True, plot m_pump_a and m_pump_b vs output frequency.
        mlim_a (tuple): if not None, restrict the m_pump_a to the given limits.
    Returns:
        m_pump_a (list[int]): resonance numbers for pump A.
        m_pump_b (list[int]): resonance numbers for pump B.
        output_freqs (np.ndarray): computed output frequencies (Hz) for each pair.
    """
    # Extract resonance mapping for the specified width index
    width_res = resonances[w_i]
    res_modes = np.asarray(width_res['resonance_modes']).astype(int)
    res_freqs = np.asarray(width_res['resonance_freqs'])  # Hz

    # Build lookup from mode number to frequency
    mode_to_freq = {int(m): f for m, f in zip(res_modes, res_freqs)}

    fixed_freqs = np.asarray(net_momentum_env['fixed_freqs'])  # length K-2
    mode_factors = np.asarray(nonlinear_env['mode_factors'])
    unit_ks = np.asarray(nonlinear_env['unit_ks'])

    if len(mode_factors) != len(unit_ks):
        raise ValueError("mode_factors and unit_ks must have the same length")
    if len(fixed_freqs) != len(mode_factors) - 2:
        raise ValueError("fixed_freqs length must be len(mode_factors) - 2")

    num_fixed = len(fixed_freqs)

    # For each fixed frequency, find the nearest resonance number by frequency proximity
    fixed_res_numbers = []
    for f in fixed_freqs:
        idx = int(np.argmin(np.abs(res_freqs - f)))
        fixed_res_numbers.append(int(res_modes[idx]))
    fixed_res_numbers = np.asarray(fixed_res_numbers, dtype=int)
    # Compute leftover resonance number required to satisfy momentum conservation in mode-number space
    leftover_resonance_number = int(np.sum(fixed_res_numbers * mode_factors[:num_fixed] * unit_ks[:num_fixed]))
    # Define coefficients for pumps A and B in the resonance-number equation:
    # m_a * (mode_factors[-2] * unit_ks[-2]) + m_b * (mode_factors[-1] * unit_ks[-1]) = leftover_resonance_number
    coeff_a = int(mode_factors[-2] * unit_ks[-2])
    coeff_b = int(mode_factors[-1] * unit_ks[-1])

    if coeff_a == 0 and coeff_b == 0:
        # No pump contributions possible to satisfy leftover; no valid pairs
        return [], [], np.array([])

    available_modes = set(int(m) for m in res_modes.tolist())
    m_pump_a = []
    m_pump_b = []

    # Search all feasible pump pairs within available resonance numbers
    # Solve for m_b given m_a when possible, or vice versa.
    if coeff_b != 0:
        for m_a in available_modes:
            rhs = leftover_resonance_number + coeff_a * m_a
            m_b = - rhs // coeff_b
            if m_b in available_modes:
                m_pump_a.append(int(m_a))
                m_pump_b.append(int(m_b))
    else:
        # coeff_b == 0, require coeff_a * m_a == leftover
        if coeff_a != 0 and (leftover_resonance_number % coeff_a == 0):
            m_a_solution = leftover_resonance_number // coeff_a
            if m_a_solution in available_modes:
                # Any m_b in available_modes is acceptable; pair each with the single m_a
                for m_b in available_modes:
                    m_pump_a.append(int(m_a_solution))
                    m_pump_b.append(int(m_b))
    # Compute output frequency for each valid pump pair using the energy relation
    # f_out = sum_j mode_factor[j] * abs(unit_k[j]) * f_j (fixed + pumps)
    # Fixed modes contribution (constant shift)
    fixed_contrib = float(np.sum(mode_factors[:num_fixed] * np.abs(unit_ks[:num_fixed]) * fixed_freqs))

    output_freqs = []
    f_as = []
    f_bs = []
    for m_a, m_b in zip(m_pump_a, m_pump_b):
        # Lookup pump frequencies from resonance numbers
        f_a = mode_to_freq[m_a]
        f_b = mode_to_freq[m_b]
        f_as.append(f_a)
        f_bs.append(f_b)
        pump_contrib = (mode_factors[-2] * np.abs(unit_ks[-2]) * f_a) + (mode_factors[-1] * np.abs(unit_ks[-1]) * f_b)
        output_freqs.append(fixed_contrib + pump_contrib)
    f_as = np.asarray(f_as)
    f_bs = np.asarray(f_bs)
    output_freqs = np.asarray(output_freqs)

    # Crop the data to the limits of mlim_a
    if mlim_a is not None:
        idx_min = np.argmin(np.abs(np.array(m_pump_a) - mlim_a[0]))
        idx_max = np.argmin(np.abs(np.array(m_pump_a) - mlim_a[1]))
        m_pump_a = m_pump_a[idx_min:idx_max+1]
        m_pump_b = m_pump_b[idx_min:idx_max+1]
        output_freqs = output_freqs[idx_min:idx_max+1]

    if plot and len(output_freqs) > 1:
        fig, ax0 = plt.subplots(1, 1, figsize=(5, 5), dpi=200)
        ax1 = ax0.twiny()
        # ax1.scatter(m_pump_b, output_freqs / 1e12, s=10, color='tab:orange')
        ax1.spines["bottom"].set_position(("outward", 40))  # 40 points below the original
        # if mlim_a is None:
        ax0.set_xlim((m_pump_a[0], m_pump_a[-1]))
        ax1.set_xlim((m_pump_b[0], m_pump_b[-1]))  # Reverse the x-axis direction 
        ax0.scatter(m_pump_a, c/output_freqs*1e6, s=400/len(m_pump_a), color='tab:blue')
        if len(m_pump_a) < 22:
            ax0.set_xticks(m_pump_a)
            ax1.set_xticks(m_pump_b)
        ax0.set_xlabel(r'Mode Number $m_a$', fontsize=15)
        ax1.set_xlabel(r'Mode Number $m_b$', fontsize=15, labelpad=10, loc='center')
        ax1.xaxis.set_label_position("top")
        ax1.xaxis.set_ticks_position("top")
        ax0.set_ylabel(f'Output Wavelength ($\\mu$m)', fontsize=15)
        # ax0.set_title(f'Output Wavelengths for m([Signal, Idler]) = {fixed_res_numbers * mode_factors[:num_fixed] * unit_ks[:num_fixed]}', fontsize=15)
        plt.tight_layout()
        plt.show()



    return m_pump_a, m_pump_b, f_as, f_bs, output_freqs