import gdsfactory as gf
import numpy as np
gf.clear_cache()
def euler_curve_tapered(width, radius: float = 10, p2=.5):
    c = gf.Component("euler_curve")


    # Create an Euler bend path
    outer_path = gf.path.euler(
        radius=radius,
        angle=180,  # Total bend angle (to reverse direction)
        p=1,  # Shape parameter for Euler bend (1.0 is a good default)
        use_eff=True,
    )
    inner_path = gf.path.euler(
        radius=radius-width,
        angle=180,  # Total bend angle (to reverse direction)
        p=p2,  # Shape parameter for Euler bend (1.0 is a good default)
        use_eff=True,
    )
    inner_path.movey(width)
    all_points = np.concatenate((outer_path.points, inner_path.points[::-1]))
    mid_y = (np.max(all_points[:, 1]) - np.min(all_points[:, 1]))/2
    far_x = (np.max(outer_path.points[:,0]) + np.max(inner_path.points[:, 0]))/2
    c.add_polygon(all_points, layer=(1, 0))
    c.add_port("o1", center =(outer_path.points[0] + inner_path.points[0])/2, layer=(1,0), width=width)
    c.add_port("o2", center =(outer_path.points[-1] + inner_path.points[-1])/2, layer=(1,0), width=width)
    c.add_port("i1", center =(far_x, mid_y), layer=(1,0), width=width)

    return c

def euler_curve_transition(initial_width, curve_width, outer_radius: float = 10, npoints=None, ttype='linear'):
    c = gf.Component("euler_curve")

    initial_section = gf.Section(width=initial_width, offset=0, layer=(1, 0), name="wg")
    X1 = gf.CrossSection(sections=[initial_section])
    curve_section = gf.Section(width=curve_width, offset=0, layer=(1, 0), name="wg")
    X2 = gf.CrossSection(sections=[curve_section])

    Xtrans = gf.path.transition(cross_section1=X1, cross_section2=X2, width_type=ttype)
    Xtrans_rev = gf.path.transition(cross_section1=X2, cross_section2=X1, width_type=ttype)
    # Create an Euler bend path
    path_up = gf.path.euler(
        radius=outer_radius-initial_width/2,
        angle=180,  # Total bend angle (to reverse direction)
        p=1,  # Shape parameter for Euler bend (1.0 is a good default)
        use_eff=True,
        npoints=npoints
    )
    path_down = gf.path.euler(
        radius=outer_radius-initial_width/2,
        angle=180,  # Total bend angle (to reverse direction)
        p=1,  # Shape parameter for Euler bend (1.0 is a good default)
        use_eff=True,
        npoints=npoints
    )
    total_points = len(path_up.points)
    path_up.end_angle = 90
    path_up.points = path_up.points[:total_points//2+1, :]
    path_down.start_angle = 90
    path_down.points = path_down.points[total_points//2:, :]
    transition_up = gf.path.extrude_transition(path_up, Xtrans)
    transition_down = gf.path.extrude_transition(path_down, Xtrans_rev)
    transition_up_ref = c << transition_up
    transition_down_ref = c << transition_down
    c.movey(initial_width/2)
    
    c.add_port("o1", center =path_up.points[0], layer=(1,0), width=initial_width)
    c.add_port("o2", center =path_down.points[-1], layer=(1,0), width=initial_width)
    c.add_port("i1", center=path_up.points[-1], layer=(1,0), width=curve_width)
    return c

def euler_curve_lopsided(initial_width, final_width, outer_radius: float = 10, npoints=None, ttype='linear', name='euler_curve'):
    c = gf.Component(name)

    initial_section = gf.Section(width=initial_width, offset=0, layer=(1, 0), name="wg")
    X1 = gf.CrossSection(sections=[initial_section])
    curve_section = gf.Section(width=final_width, offset=0, layer=(1, 0), name="wg")
    X2 = gf.CrossSection(sections=[curve_section])

    Xtrans = gf.path.transition(cross_section1=X1, cross_section2=X2, width_type=ttype)
    Xtrans_rev = gf.path.transition(cross_section1=X2, cross_section2=X1, width_type=ttype)
    # Create an Euler bend path
    path = gf.path.euler(
        radius=outer_radius-initial_width/2,
        angle=180,  # Total bend angle (to reverse direction)
        p=1,  # Shape parameter for Euler bend (1.0 is a good default)
        use_eff=True,
        npoints=npoints
    )
    transition = gf.path.extrude_transition(path, Xtrans)
    transition_ref = c << transition
    c.movey(initial_width/2)
    
    c.add_port("o1", center=path.points[0], layer=(1,0), width=initial_width)
    c.add_port("o2", center=path.points[-1], layer=(1,0), width=final_width)
    c.add_port("i1", center=path.points[len((path.points)+1)//2], layer=(1,0), width=initial_width)
    return c

def compute_normals(points):
    """
    Given an (N, 2) array of (x, y) points, compute unit normals at each point.
    Uses forward/backward difference for endpoints and central diff for inner points.
    Returns:
        unit_normals: (N, 2) array of unit normal vectors
    """
    points = np.asarray(points)
    N = len(points)

    # Step 1: Compute tangents using np.diff
    diffs = np.diff(points, axis=0)  # (N-1, 2)

    # Step 2: Approximate tangent vectors at each point
    tangents = np.zeros_like(points)
    tangents[1:-1] = (diffs[1:] + diffs[:-1]) / 2  # central diff
    tangents[0] = diffs[0]                        # forward diff
    tangents[-1] = diffs[-1]                      # backward diff

    # Step 3: Normalize tangents
    tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    unit_tangents = tangents / tangent_norms

    # Step 4: Rotate tangents by +90° to get normals
    unit_normals = np.zeros_like(unit_tangents)
    unit_normals[:, 0] = unit_tangents[:, 1]
    unit_normals[:, 1] = -unit_tangents[:, 0]

    return unit_normals, unit_tangents

def euler_curve_tapered_coupler(initial_width, curve_width, outer_radius, coupler_width, start_gap, mid_gap, end_gap, output_angle=90, output_radius=5, start_taper_halfway=False, end_taper_halfway=False, euler_res=100, ttype='sine', offset_type='sine', layer=2):
    radius = outer_radius-initial_width/2

    p = gf.path.euler(radius=radius, angle=180, p=1, use_eff=True, npoints=int(euler_res*radius))
    p_normals, p_tangents = compute_normals(p.points)


    dx = np.diff(p.points[:, 0])
    dy = np.diff(p.points[:, 1])
    lengths = np.cumsum(np.sqrt(dx**2 + dy**2))
    fractional_length = np.concatenate([[0], lengths]) / lengths[-1]
    widths = np.array(initial_width + (1 - np.cos(2*np.pi *fractional_length)) / 2 * (curve_width - initial_width))
    widths *= 0
    gaps = np.array(start_gap + (1 - np.cos(np.pi *fractional_length)) / 2 * (end_gap - start_gap))
    if offset_type=='sine':
        gaps = np.array(start_gap + (1 - np.cos(2*np.pi *fractional_length)) / 2 * (mid_gap - start_gap))
        gaps[fractional_length>=0.5] = np.array(mid_gap + (1 + np.cos(2*np.pi *fractional_length)) / 2 * (end_gap - mid_gap))[fractional_length>=0.5]
        if end_taper_halfway and not start_taper_halfway:
            gaps = np.array(start_gap + (1 - np.cos(2*np.pi *fractional_length)) / 2 * (end_gap - start_gap))
    elif offset_type=='linear':
        gaps = np.array(start_gap + (1 - np.cos(2*np.pi *fractional_length)) / 2 * (mid_gap - start_gap))
        # gaps = np.array(start_gap + 2*(fractional_length) * (mid_gap - start_gap))
        gaps[fractional_length>=0.5] = np.array(mid_gap + 2*(fractional_length-0.5) * (end_gap - mid_gap))[fractional_length>=0.5]
        if end_taper_halfway and not start_taper_halfway:
            gaps = np.array(start_gap + (fractional_length) * (end_gap - start_gap))

    # Shift p by widths/2
    p.points = p.points + p_normals * widths[:, np.newaxis] / 2 + p_normals * gaps[:, np.newaxis]

    s0 = gf.Section(width=coupler_width, offset=0, layer=(layer, 0), name="coupler", port_names=("o1", "o2"))
    c_X1 = gf.CrossSection(sections=[s0])

    # s0 = gf.Section(width=width, offset=width/2+(start_gap+end_gap)/2, layer=(1, 0), name="coupler", port_names=("o1", "o2"))
    # c_X2 = gf.CrossSection(sections=[s0])

    s0 = gf.Section(width=0.002, offset=0, layer=(layer, 0), name="coupler", port_names=("o1", "o2"))
    c_X3 = gf.CrossSection(sections=[s0])
    CTrans = gf.path.transition(cross_section1=c_X1, cross_section2=c_X3, width_type=ttype, offset_type=offset_type)

    if start_taper_halfway:
        c_p1 = p.copy()
        c_p1.points = c_p1.points[:(len(c_p1.points)+1)//2]  # Keep only half of the path
        c_p1.end_angle = 90
        c_p2 = p.copy()
        c_p2.points = c_p2.points[len(c_p2.points+1)//2:]  # Keep only the second half of the path
        c_p2.start_angle = 90
        coupler_invariate = gf.path.extrude(c_p1, c_X1)
        coupler_transition = gf.path.extrude_transition(c_p2, CTrans)
    elif end_taper_halfway:
        c_p1 = p.copy()
        c_p1.points = c_p1.points[:len(c_p1.points)//2+1]  # Keep only half of the path
        c_p1.end_angle = 90
        coupler_invariate = None
        coupler_transition = gf.path.extrude_transition(c_p1, CTrans)
    else:
        coupler_invariate = None
        coupler_transition = gf.path.extrude_transition(p, CTrans)
        
    c= gf.Component("tapered_euler_coupler")

    coupler_transition_ref = c << coupler_transition
    if coupler_invariate:
        coupler_invariate_ref = c << coupler_invariate
    if output_angle:
        euler_out = gf.components.bend_euler(
            radius=output_radius,
            with_arc_floorplan=True,
            angle=output_angle,  # Total bend angle (to reverse direction)
            p=1,  # Shape parameter for Euler bend (1.0 is a good default)
            npoints=int(euler_res*output_radius),
            cross_section=c_X1,
            allow_min_radius_violation=True,
        )
    else:
        euler_out = gf.components.straight(length=output_radius, cross_section=c_X1, npoints=100)

    euler_out_ref = c << euler_out
    if start_taper_halfway:
        euler_out_ref.connect(euler_out_ref.ports['o1'], coupler_invariate_ref.ports['o1'], allow_layer_mismatch=True)
    else:
        euler_out_ref.connect(euler_out_ref.ports['o1'], coupler_transition_ref.ports['o1'], allow_layer_mismatch=True)
    c.add_port("o1", center=euler_out_ref.ports['o2'].center, width=coupler_width, layer=(layer, 0))

    c.add_port("o2", center=coupler_transition_ref.ports['o2'].center, width=coupler_width, layer=(layer, 0), orientation=180)
    return c

def tapered_euler_resonator(wg_separation, wg_length, wg_width, final_euler_width, straight_taper_length=0, straight_taper_end_w=0, support_width=0, support_base=False, use_bottom_taper=True, coupler_args=None, ttype='linear', straight_taper_ttype='linear'):
    outer_radius = wg_separation/2 
    outer_radius += wg_width/2 if not straight_taper_length > 0 else straight_taper_end_w/2
    c = gf.Component("tapered_euler_resonator")
    initial_euler_width = wg_width if not straight_taper_length else straight_taper_end_w
    euler_curve = euler_curve_transition(initial_euler_width, final_euler_width, outer_radius=outer_radius, ttype=ttype)
    euler_curve_west_ref = c << euler_curve
    euler_curve_west_ref.rotate(180)
    euler_curve_east_ref = c << euler_curve
    euler_curve_west_ref.movey(outer_radius*2-initial_euler_width/2)
    euler_curve_east_ref.movey(-initial_euler_width/2)
    if coupler_args:
        euler_coupler = euler_curve_tapered_coupler(initial_euler_width, final_euler_width, outer_radius=outer_radius, **coupler_args)
        euler_coupler_ref = c << euler_coupler
        euler_coupler_ref.movey(-initial_euler_width/2)

    if wg_length > 0:
        euler_curve_west_ref.movex(-wg_length/2)
        euler_curve_east_ref.movex(wg_length/2)
        straight_wg = gf.components.straight(
            length=wg_length,
                    width=wg_width, npoints=100)
        straight_wg_south_ref = c << straight_wg
        straight_wg_north_ref = c << straight_wg
        straight_wg_south_ref.movex(-wg_length/2)
        straight_wg_north_ref.movex(-wg_length/2)
        straight_wg_north_ref.movey(wg_separation)

    if straight_taper_length > 0:
        euler_curve_west_ref.movex(-straight_taper_length)
        euler_curve_east_ref.movex(straight_taper_length)
        s0 = gf.Section(width=wg_width, offset=0, layer=(1, 0), name="straight_taper", port_names=("o1", "o2"))
        c_X1 = gf.CrossSection(sections=[s0])

        s0 = gf.Section(width=straight_taper_end_w, offset=0, layer=(1, 0), name="straight_taper", port_names=("o1", "o2"))
        c_X2 = gf.CrossSection(sections=[s0])
        straight_taper = gf.components.taper_cross_section(cross_section1=c_X1, cross_section2=c_X2, length=straight_taper_length, npoints=101, linear=False, width_type=straight_taper_ttype).copy()
        straight_taper_ne_ref = c << straight_taper
        straight_taper_nw_ref = c << straight_taper
        straight_taper_se_ref = c << straight_taper
        straight_taper_sw_ref = c << straight_taper

        straight_taper_ne_ref.movey(wg_separation)
        straight_taper_ne_ref.movex(wg_length/2)

        straight_taper_nw_ref.mirror_x()
        straight_taper_nw_ref.movey(wg_separation)
        straight_taper_nw_ref.movex(-wg_length/2)

        straight_taper_se_ref.movex(wg_length/2)
        straight_taper_sw_ref.mirror_x()
        straight_taper_sw_ref.movex(-wg_length/2)
    if support_width:
        support_length = outer_radius + wg_length if support_base else outer_radius
        support_wg = gf.components.straight(
            length=support_length,
                    width=support_width)
        support_wg_east_ref = c << support_wg
        support_wg_west_ref = c << support_wg
        support_wg_east_ref.connect(support_wg_east_ref.ports['o2'], euler_curve_east_ref.ports['i1'], allow_width_mismatch=True)
        support_wg_west_ref.connect(support_wg_west_ref.ports['o1'], euler_curve_west_ref.ports['i1'], allow_width_mismatch=True)
        support_wg_east_ref.movex(-support_length)
        support_wg_west_ref.movex(support_length)
        support_wg_east_ref.movey(initial_euler_width/2)
        support_wg_west_ref.movey(-initial_euler_width/2)
        base_yspan = final_euler_width*10
        base_xspan = wg_length
        if support_base:
            support_base_ref = c << gf.components.rectangle(size=(base_xspan, base_yspan), layer=(1, 0))
            support_base_ref.movey(wg_separation/2-base_yspan/2)
            support_base_ref.movex(-base_xspan/2)
    if not use_bottom_taper:
        bottom_wg = gf.components.straight(
            length=wg_length+straight_taper_length*2,
                    width=initial_euler_width)
        bottom_wg_ref = c << bottom_wg
        bottom_wg_ref.movex(-wg_length/2-straight_taper_length)
        bottom_wg_ref.movey(straight_taper_end_w/2-wg_width/2)
    # c.movey(wg_width/2)
    if coupler_args:
        if coupler_args['end_taper_halfway']:
            euler_coupler_ref.connect(euler_coupler_ref.ports['o2'], euler_curve_east_ref.ports['i1'], allow_layer_mismatch=True, allow_width_mismatch=True)
            
            euler_coupler_ref.movey(initial_euler_width/2)
            euler_coupler_ref.movex(coupler_args['end_gap'])
        else:
            euler_coupler_ref.connect(euler_coupler_ref.ports['o2'], straight_wg_north_ref.ports['o2'], allow_layer_mismatch=True, allow_width_mismatch=True)
            euler_coupler_ref.movey(coupler_args['end_gap'])
    return c


def lopsided_euler_resonator(wg_separation, wg_length,sw_wg_width, se_wg_width, north_wg_width, support_width=0, support_base=False, ttype='linear', coupler_width=.6, coupler_type='sine', output_length=10, crop_taper=0):
    south_wg_width = np.max([sw_wg_width, se_wg_width])
    outer_radius = wg_separation/2 
    c = gf.Component("lopsided_euler_resonator")
    euler_curve_se = euler_curve_lopsided(se_wg_width, north_wg_width, outer_radius+se_wg_width/2, npoints=None, ttype=ttype, name='euler_curve_se')
    euler_curve_sw = euler_curve_lopsided(sw_wg_width, north_wg_width, outer_radius+sw_wg_width/2, npoints=None, ttype=ttype, name='euler_curve_sw')
    euler_curve_west_ref = c << euler_curve_sw
    euler_curve_west_ref.mirror_y()
    euler_curve_west_ref.rotate(180)
    euler_curve_east_ref = c << euler_curve_se
    # euler_curve_west_ref.movey(outer_radius*2-initial_euler_width/2)
    euler_curve_east_ref.movey(-se_wg_width/2)
    euler_curve_west_ref.movey(-sw_wg_width/2)

    if wg_length > 0:
        euler_curve_west_ref.movex(-wg_length/2)
        euler_curve_east_ref.movex(wg_length/2)
        straight_wg_north = gf.components.straight(
            length=wg_length,
                    width=north_wg_width, npoints=100)
        straight_wg_south = gf.components.taper(length=wg_length, width1=sw_wg_width, width2=se_wg_width, port=None, layer=(1, 0))
        straight_wg_south_ref = c << straight_wg_south
        straight_wg_north_ref = c << straight_wg_north
        straight_wg_south_ref.movex(-wg_length/2)
        straight_wg_north_ref.movex(-wg_length/2)
        straight_wg_north_ref.movey(wg_separation)

    if support_width:
        support_length = outer_radius + wg_length if support_base else outer_radius
        support_wg = gf.components.straight(
            length=support_length,
                    width=support_width)
        support_wg_east_ref = c << support_wg
        support_wg_west_ref = c << support_wg
        support_wg_east_ref.connect(support_wg_east_ref.ports['o2'], euler_curve_east_ref.ports['i1'], allow_width_mismatch=True)
        support_wg_west_ref.connect(support_wg_west_ref.ports['o1'], euler_curve_west_ref.ports['i1'], allow_width_mismatch=True)
        support_wg_east_ref.movex(-support_length)
        support_wg_west_ref.movex(support_length)
        support_wg_east_ref.movey(south_wg_width/2)
        support_wg_west_ref.movey(south_wg_width/2)
        base_yspan = north_wg_width*10
        base_xspan = wg_length
        if support_base:
            support_base_ref = c << gf.components.rectangle(size=(base_xspan, base_yspan), layer=(1, 0))
            support_base_ref.movey(wg_separation/2-base_yspan/2)
            support_base_ref.movex(-base_xspan/2)

    if coupler_width:
        s0 = gf.Section(width=coupler_width, offset=0, layer=(2, 0), name="coupler", port_names=("o1", "o2"))
        c_X1 = gf.CrossSection(sections=[s0])

        s0 = gf.Section(width=0.002, offset=0, layer=(2, 0), name="coupler", port_names=("o1", "o2"))
        c_X2 = gf.CrossSection(sections=[s0])
        coupler_taper = gf.components.taper_cross_section(cross_section1=c_X1, cross_section2=c_X2, length=wg_length-crop_taper, npoints=101, linear=False, width_type=coupler_type).copy()
        coupler_taper_ref = c << coupler_taper
        coupler_taper_ref.movex(-wg_length/2)
        wg_out = gf.components.straight(length=output_length, cross_section=c_X1, npoints=100)
        wg_out_ref = c << wg_out
        wg_out_ref.connect(wg_out_ref.ports['o1'], coupler_taper_ref.ports['o1'])
    return c

