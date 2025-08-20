import gdsfactory as gf
import numpy as np

@gf.cell
def hybrid_cavity(a1, a2, a3, a4, a5, b1, b4, left_periods, right_periods, center_over_rib, rib_length, left_padding, right_padding):
    '''Builds a hybrid diamond-nitride nanophotonic cavity
    inputs:
        a1: central grating pitch
        a2: adiabatic taper pitch change
        a3: Number of periods to reach final pitch
        a4: SiN thickness
        a5: SiN duty cycle
        b1: diamond waveguide width
        b4: diamond waveguide height
        left_periods: number of periods on the left side of the waveguide
        right_periods: number of periods on the right side of the waveguide
        center_over_rib: if True, the dipole is placed over the rib, else the dipole is placed over a gap between ribs
        rib_length: length of the rib
        left_padding: padding on the left side of the ribs
        right_padding: padding on the right side of the ribs
    outputs:
        c: gdsfactory component
    '''
    ############################################################ Build Geometry ############################################################
    # build nitride ribs
    c = gf.Component()

    max_ribs = max([left_periods, right_periods])
    pitches = a1+a2*(np.linspace(0, max_ribs, max_ribs+1)/a3)**2
    pitches = np.minimum(pitches, a1+a2)
    locations_x = np.cumsum(pitches)
    if center_over_rib:
        locations_x = np.concatenate(([0],locations_x))
        max_ribs += 1
    else:
        locations_x -= a1/2
    min_x = 0
    max_x = 0
    for i in range(0, max_ribs):
        size = (pitches[i]*a5, rib_length)
        if i <= left_periods:  
            center = (-locations_x[i], 0)
            rib = gf.components.rectangle(size=size, layer='NITRIDE', centered=True)
            rib_ref = c << rib
            rib_ref.move(center)
        if i == left_periods:
            min_x = -locations_x[i]-pitches[i]*a5/2
            if left_padding:
                left_pad_center = (min_x-left_padding/2, 0)
                left_pad = gf.components.rectangle(size=(left_padding, rib_length), layer='NITRIDE', centered=True)
                left_pad_ref = c << left_pad
                left_pad_ref.move(left_pad_center)
                min_x -= left_padding

        if i <= right_periods:
            center = (locations_x[i], 0)
            rib = gf.components.rectangle(size=size, layer='NITRIDE', centered=True)
            rib_ref = c << rib
            rib_ref.move(center)
        if i == right_periods:
            max_x = locations_x[i]+pitches[i]*a5/2
            if right_padding:
                right_pad_center = (max_x+right_padding/2, 0)
                right_pad = gf.components.rectangle(size=(right_padding, rib_length), layer='NITRIDE', centered=True)
                right_pad_ref = c << right_pad
                right_pad_ref.move(right_pad_center)
                max_x += right_padding

    # build diamond waveguide
    wg_length = max_x-min_x
    wg_center = ((max_x+min_x)/2, 0)
    wg = gf.components.rectangle(size=(wg_length, b1), layer='DIAMOND', centered=True)
    wg_ref = c << wg
    wg_ref.move(wg_center)

    # Add ports
    port_center = (0,0)
    if left_periods == 0 and left_padding:
        port_center = (-left_padding/2, 0)
    c.add_port("o1", center=port_center, width=b1, layer='DIAMOND', orientation=180)
    c.add_port("o2", center=(max_x-right_padding/2, 0), width=b1, layer='DIAMOND', orientation=0)
    
    return c