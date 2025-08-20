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

def get_fdtd_sim(tidy_component, modeler, inputs, outputs, z_monitor=None, z_freq=None, z_thickness=0):
    '''Returns the FDTD simulation object for a given set of inputs and outputs
    inputs:
        modeler: ComponentModeler object
        inputs: dict{port: {modes: [mode_nums], amps: [amps], phases: [phases]}}
        outputs: dict{port: {modes: [mode_nums]}}
    '''
    sources = []
    monitors = []
    for port in modeler.ports:
        if port.name in inputs:
            for input_mode, input_amp, input_phase in zip(inputs[port.name]['modes'], inputs[port.name]['amps'], inputs[port.name]['phases']):
                if 'dipole' in input_mode:
                    freq0 = np.mean(modeler.freqs)
                    fdiff = max(modeler.freqs) - min(modeler.freqs)
                    fwidth = max(fdiff, freq0/10)
                    t_start = 2/fwidth
                    dipole_source = td.PointDipole(
                        center=port.center,
                        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth, phase=input_phase, amplitude=input_amp),
                        name=port.name+f'@{input_mode}',
                        polarization=input_mode.split('_')[1]
                    )
                    sources.append(dipole_source)
                    monitors.append(
                        td.FieldTimeMonitor(
                            fields=[input_mode.split('_')[1]],
                            center=port.center,
                            size=(0, 0, 0),
                            start=t_start,
                            name=port.name+f'@{input_mode}',
                        )
                    )
                else:
                    mode_source = modeler.to_source(port=port, mode_index=input_mode, phase=input_phase, amplitude=input_amp)
                    sources.append(mode_source)
        if port.name in outputs:
            mode_monitor = modeler.to_monitor(port=port)
            monitors.append(mode_monitor)
    
    if z_monitor is not None:
        if z_freq is None:
            z_freq=np.mean(modeler.freqs)
        monitors.append(td.FieldMonitor(name='field', center=[0,0, tidy_component.get_layer_center(z_monitor)[2]], size=[td.inf, td.inf, z_thickness], interval_space=(4,4,2), freqs=z_freq))
    
    sim = modeler.simulation.copy(update=dict(sources=sources, monitors=monitors))
    return sim