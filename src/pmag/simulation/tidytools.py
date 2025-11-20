import numpy as np
import tidy3d as td

def validate_sim_for_daily_allowance(simulation, max_grid_points=100e6, max_time_steps=50000):
    """Validate the simulation for daily allowance."""
    error = False
    error_str = ""
    if simulation.num_computational_grid_points > max_grid_points:
        error=True
        error_str = f"Grid size ({simulation.num_computational_grid_points/1e6}e6) exceeds daily allowance. Reduce grid size to below {max_grid_points/1e6}e6\n"
    if simulation.num_time_steps > max_time_steps:
        error=True
        error_str += f"Total time steps ({simulation.num_time_steps}) exceeds daily allowance. Reduce simulation time to below {max_time_steps}"
    if error:
        raise ValueError(error_str)
    print(f"Grid size ({simulation.num_computational_grid_points}) and total time steps ({simulation.num_time_steps}) are within daily allowance.")

def get_fdtd_sim(tidy_component, modeler, inputs, outputs, mode_volume=None, farfield=None, flux=None, output_freqs=None, custom_sources=None, sim_kwargs=None):
    """Returns the FDTD simulation object for a given set of inputs and outputs
    
    Args:
    inputs:
        tidy_component: Tidy3DComponent object
        modeler: Tidy3D ComponentModeler object
        inputs: dict{port: {modes: [mode_nums], amps: [amps], phases: [phases], freqs: [freqs]}, fwidths: [fwidths]}
            freqs: optional, if None, or not provided, the mean of the modeler.freqs is used
        outputs: dict{port: {types: list[str]}}
            types: list of strings, options are 'mode', 'time'
        mode_volume (dict): Contains keys 'layer' (str), 'thickness' (float), downsample (tuple(int, int, int)),
            optional, if None, or not provided, the mode volume is not monitored
        farfield (dict): Contains keys 'downsample' (tuple(int, int, int)), 'surface' (str, 'spherical' or 'cartesian', optional), 'res' (int), 'thetas' (list of floats, overrides res), 'phis' (list of floats, overrides res)
            optional, if None, or not provided, the farfield is not monitored
        flux (dict): Contains keys 'downsample' (tuple(int, int, int)), 'apodization' (dict with keys 'start' and 'width')
        output_freqs (list of floats): optional, if None, or not provided, the mean of the modeler.freqs is used
        custom_sources (list of tidy3d.ModeSource or tidy3d.PointDipole): optional, list of custom sources to add to the simulation
        sim_kwargs (dict): optional, kwargs to pass to the simulation constructor
    """
    sources = []
    if custom_sources is not None:
        if not isinstance(custom_sources, list):
            custom_sources = [custom_sources]
        sources.extend(custom_sources)
    monitors = []
    if sim_kwargs is None:
        sim_kwargs = {}
    # Make sure inputs and outputs exist in modeler.ports
    port_names = set(list(inputs.keys()) + list(outputs.keys()))
    missing_ports = []
    for port_name in port_names:
        port_found = False
        for port in modeler.ports:
            if port.name == port_name:
                port_found = True
                break
        if not port_found:
            missing_ports.append(port_name)
    if missing_ports:
        raise ValueError(f"Ports {missing_ports} not found in modeler.ports")
    if output_freqs is None:
        output_freqs = np.mean(modeler.freqs)

    for port in modeler.ports:
        if port.name in inputs:
            if 'freqs' not in inputs[port.name]:
                inputs[port.name]['freqs'] = [None] * len(inputs[port.name]['modes'])
            if 'fwidths' not in inputs[port.name]:
                inputs[port.name]['fwidths'] = [None] * len(inputs[port.name]['modes'])
            if 'port_offset' not in inputs[port.name]:
                inputs[port.name]['port_offset'] = [(0, 0, 0)] * len(inputs[port.name]['modes'])
            for input_mode, input_amp, input_phase, input_freq, input_fwidth, input_port_offset in zip(inputs[port.name]['modes'], inputs[port.name]['amps'], inputs[port.name]['phases'], inputs[port.name]['freqs'], inputs[port.name]['fwidths'], inputs[port.name]['port_offset']):
                if isinstance(input_mode, str) and ('dipole' in input_mode):
                    freq0 = input_freq if input_freq is not None else np.mean(modeler.freqs)
                    if input_fwidth is None:
                        fdiff = max(modeler.freqs) - min(modeler.freqs)
                        fwidth = max(fdiff, max(output_freqs)/10)
                    else:
                        fwidth = input_fwidth
                    dipole_source = td.PointDipole(
                        center=(port.center[0]+input_port_offset[0], port.center[1]+input_port_offset[1], port.center[2]+input_port_offset[2]),
                        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth, phase=input_phase, amplitude=input_amp),
                        name=port.name+f'@{input_mode}' + f'@{input_freq}' if input_freq is not None else '',
                        polarization=input_mode.split('_')[1]
                    )
                    sources.append(dipole_source)
                else:
                    mode_kwargs = {'freq0': input_freq, 'fwidth': input_fwidth}
                    mode_source = modeler.to_source(port=port, mode_index=input_mode, phase=input_phase, amplitude=input_amp, **mode_kwargs)
                    sources.append(mode_source)
        if port.name in outputs:
            if len(outputs[port.name]['types']) == 0:
                outputs[port.name]['types'] = ['mode']
            for output_type in outputs[port.name]['types']:
                if output_type == 'mode':
                    output_monitor = modeler.to_monitor(port=port, freqs=output_freqs)
                elif 'time' in output_type:
                    fdiff = max(modeler.freqs) - min(modeler.freqs)
                    fwidth = max(fdiff, max(output_freqs)/10)
                    t_start = 2/fwidth
                    output_monitor = td.FieldTimeMonitor(
                                fields=[output_type.split('_')[1]],
                                center=port.center,
                                size=(0, 0, 0),
                                start=t_start,
                                name=port.name + "_" + output_type)
                monitors.append(output_monitor)
    
    sim_center = modeler.simulation.center
    sim_size = modeler.simulation.size
    if mode_volume is not None:
        if 'downsample' not in mode_volume or mode_volume['downsample'] is None:
            mode_volume['downsample']=(4,4,2)
        if 'apodization' in mode_volume:
            apod_spec = td.ApodizationSpec(start=mode_volume['apodization']['start'], width=mode_volume['apodization']['width'])
        else:
            apod_spec = td.ApodizationSpec()
        if 'colocate' in mode_volume:
            colocate = mode_volume['colocate']
        else:
            colocate = True
        monitors.append(td.FieldMonitor(name='field', center=[0,0, tidy_component.get_layer_center(mode_volume['layer'])[2]], size=[td.inf, td.inf, mode_volume['thickness']], interval_space=mode_volume['downsample'], freqs=output_freqs, apodization=apod_spec, colocate=colocate))
    if farfield is not None:
        if 'downsample' not in farfield or farfield['downsample'] is None:
            farfield['downsample']=(4,4,4)
        if 'apodization' in farfield:
            apod_spec = td.ApodizationSpec(start=farfield['apodization']['start'], width=farfield['apodization']['width'])
        else:
            apod_spec = td.ApodizationSpec()
        distance_from_pml = 0.1
        if 'surface' not in farfield or farfield['surface'] == 'spherical':
            if 'thetas' in farfield:
                thetas = farfield['thetas']
            else:
                thetas = np.linspace(0, np.pi, farfield['res'])
            if 'phis' in farfield:
                phis = farfield['phis']
            else:
                phis = np.linspace(0, 2*np.pi, 2*farfield['res'])
            if 'exclude_surfaces' in farfield:
                exclude_surfaces = farfield['exclude_surfaces']
            else:
                exclude_surfaces = None
            monitors.append(td.FieldProjectionAngleMonitor(name='farfield_monitor', center=sim_center, size=(sim_size[0]-2*distance_from_pml, sim_size[1]-2*distance_from_pml, sim_size[2]-2*distance_from_pml), 
                    interval_space=farfield['downsample'], freqs=output_freqs, theta=thetas, phi=phis, apodization=apod_spec, exclude_surfaces=exclude_surfaces))
        elif farfield['surface'] == 'cartesian':
            if 'proj_distance' in farfield:
                proj_distance = farfield['proj_distance']
            else:
                proj_distance = 1e6
            if 'res' in farfield:
                res = farfield['res']
            else:
                res = 100
            if 'x' in farfield:
                x = farfield['x']
            else:
                x = np.linspace(-proj_distance/2, proj_distance/2, res)
            if 'y' in farfield:
                y = farfield['y']
            else:
                y = np.linspace(-proj_distance/2, proj_distance/2, res)
            if 'exclude_surfaces' in farfield:
                exclude_surfaces = farfield['exclude_surfaces']
            else:
                exclude_surfaces = None
            monitors.append(td.FieldProjectionCartesianMonitor(name='farfield_monitor_cartesian', center=sim_center, size=(sim_size[0]-2*distance_from_pml, sim_size[1]-2*distance_from_pml, sim_size[2]-2*distance_from_pml), 
                    interval_space=farfield['downsample'], freqs=output_freqs, x=x, y=y, proj_axis=2, proj_distance=proj_distance, apodization=apod_spec, exclude_surfaces=exclude_surfaces))
    if flux is not None:
        if 'downsample' not in flux or flux['downsample'] is None:
            flux['downsample']=(4,4,4)
        if 'apodization' in flux:
            apod_spec = td.ApodizationSpec(start=flux['apodization']['start'], width=flux['apodization']['width'])
        else:
            apod_spec = td.ApodizationSpec()
        distance_from_pml = 0.1
        surfaces = {'left': 'x-', 'right': 'x+', 'back': 'y-', 'front': 'y+', 'bottom': 'z-', 'top': 'z+'}
        for name, direction in surfaces.items():
            excludes = list({k: v for k, v in surfaces.items() if k != name}.values())
            monitors.append(td.FluxMonitor(name='flux_'+name, center=sim_center, size=(sim_size[0]-2*distance_from_pml, sim_size[1]-2*distance_from_pml, sim_size[2]-2*distance_from_pml), interval_space=flux['downsample'], freqs=output_freqs, apodization=apod_spec, exclude_surfaces=excludes))
    sim = modeler.simulation.copy(update=dict(sources=sources, monitors=monitors, **sim_kwargs))
    return sim