import gdsfactory as gf
import numpy as np
from gdsfactory.cross_section import cross_section
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, Delta
@gf.cell
def cropped_coupler_full(
    coupling_length: float = 40.0,
    dx: Delta = 10.0,
    dy: Delta = 4.8,
    gap: float = 0.5,
    dw: float = 0.1,
    cross_section: CrossSectionSpec = "strip",
    width: float | None = None,
    crop_bottom_x: float = 0,
) -> Component:
    """Adiabatic Full coupler.

    Design based on asymmetric adiabatic full
    coupler designs, such as the one reported in 'Integrated Optic Adiabatic
    Devices on Silicon' by Y. Shani, et al (IEEE Journal of Quantum
    Electronics, Vol. 27, No. 3 March 1991).

    1. is the first half of the input S-bend straight where the
    input straights widths taper by +dw and -dw,
    2. is the second half of the S-bend straight with constant,
    unbalanced widths,
    3. is the coupling region where the straights from unbalanced widths to
    balanced widths to reverse polarity unbalanced widths,
    4. is the fixed width straight that curves away from the coupling region,
    5.is the final curve where the straights taper back to the regular width
    specified in the straight template.

    Args:
        coupling_length: Length of the coupling region in um.
        dx: Length of the bend regions in um.
        dy: Port-to-port distance between the bend regions in um.
        gap: Distance between the two straights in um.
        dw: delta width. Top arm tapers to width - dw, bottom to width + dw in um.
        cross_section: cross-section spec.
        width: width of the waveguide. If None, it will use the width of the cross_section.

    """
    c = gf.Component()

    if width:
        x = gf.get_cross_section(cross_section=cross_section, width=width)
    else:
        x = gf.get_cross_section(cross_section=cross_section)
    x_top = x.copy(width=x.width + dw)
    x_bottom = x.copy(width=x.width - dw)

    taper_top = c << gf.components.taper(
        length=coupling_length-crop_bottom_x,
        width1=x_top.width,
        width2=x_bottom.width,
        cross_section=cross_section,
    )

    taper_bottom = c << gf.components.taper(
        length=coupling_length,
        width1=x_bottom.width,
        width2=x_top.width,
        cross_section=cross_section,
    )

    bend_input_top = c << gf.c.bend_s(
        size=(dx, (dy - gap - x_top.width) / 2.0), cross_section=x_top
    )
    bend_input_top.movey((x_top.width + gap) / 2.0)

    bend_input_bottom = c << gf.c.bend_s(
        size=(dx, (-dy + gap + x_bottom.width) / 2.0), cross_section=x_bottom
    )
    bend_input_bottom.movey(-(x_bottom.width + gap) / 2.0)

    taper_top.connect("o1", bend_input_top.ports["o1"])
    taper_bottom.connect("o1", bend_input_bottom.ports["o1"])

    bend_output_top = c << gf.c.bend_s(
        size=(dx, (dy - gap - x_top.width) / 2.0), cross_section=x_bottom
    )

    bend_output_bottom = c << gf.c.bend_s(
        size=(dx, (-dy + gap + x_bottom.width) / 2.0), cross_section=x_top
    )

    bend_output_top.connect("o2", taper_top.ports["o2"], mirror=True)
    bend_output_bottom.connect("o2", taper_bottom.ports["o2"], mirror=True)

    x.add_bbox(c)

    c.add_port("o1", port=bend_input_bottom.ports["o2"])
    c.add_port("o2", port=bend_input_top.ports["o2"])
    c.add_port("o3", port=bend_output_top.ports["o1"])
    c.add_port("o4", port=bend_output_bottom.ports["o1"])
    c.auto_rename_ports()

    c.flatten()
    return c

@gf.cell
def extended_coupler_ring(
    gap= 0.2,
    radius=5,
    length_x= 4.0,
    width= 0.5,
    length_extension= 1,
    layer=(1, 0),
):
    xsection = cross_section(width=width, layer=layer, radius_min=radius/2)
    coupler = gf.components.coupler_ring(gap=gap, radius=radius, length_x=length_x, bend='bend_euler', straight='straight', cross_section=xsection)
    straight = gf.components.straight(length=length_extension, cross_section=xsection)
    c = gf.Component()
    coupler_ref = c << coupler
    straight_sw_ref = c << straight
    straight_nw_ref = c << straight
    straight_ne_ref = c << straight
    straight_se_ref = c << straight
    straight_sw_ref.connect(port="o1", other=coupler_ref.ports["o4"])
    straight_nw_ref.connect(port="o1", other=coupler_ref.ports["o3"])
    straight_ne_ref.connect(port="o1", other=coupler_ref.ports["o2"])
    straight_se_ref.connect(port="o1", other=coupler_ref.ports["o1"])
    c.ports = coupler_ref.ports

    return c

@gf.cell
def extended_double_coupler_ring(
    ring_gap= 0.3,
    double_gap=0.2,
    radius=5,
    length_x= 4.0,
    width= 0.5,
    length_extension= 1,
    bend_dx=0,
    bend_dy=0,
    bend_padding=3,
    crop_bottom_x=0,
    layer=(1, 0),
):
    xsection = cross_section(width=width, layer=layer, radius_min=radius/10)
    coupler = gf.components.coupler_ring(gap=ring_gap, radius=radius, length_x=length_x, bend='bend_euler', straight='straight', length_extension=0, cross_section=xsection)
    straight = gf.components.straight(length=length_extension, cross_section=xsection)

    straight_double = gf.components.straight(length=2*length_extension+length_x+6+2*radius, cross_section=xsection)
    if bend_dx == 0 or bend_dy == 0:
        c = gf.Component()
        coupler_ref = c << coupler
        straight_sw_ref = c << straight
        straight_nw_ref = c << straight
        straight_ne_ref = c << straight
        straight_se_ref = c << straight
        straight_double_ref = c << straight_double
        straight_sw_ref.connect(port="o1", other=coupler_ref.ports["o4"])
        straight_nw_ref.connect(port="o1", other=coupler_ref.ports["o3"])
        straight_ne_ref.connect(port="o1", other=coupler_ref.ports["o2"])
        straight_se_ref.connect(port="o1", other=coupler_ref.ports["o1"])
        straight_double_ref.connect(port="o1", other=coupler_ref.ports["o1"])
        straight_double_ref.movex(length_extension+length_x)
        straight_double_ref.movey(-double_gap-width)
        c.ports = coupler_ref.ports
        c.ports['o1'].center = (c.ports['o1'].center[0], c.ports['o1'].center[1] - double_gap/2-width/2)
        c.ports['o4'].center = (c.ports['o4'].center[0], c.ports['o4'].center[1] - double_gap/2-width/2)
        c.movey(-c.ports['o1'].center[1])
    else:
        c = gf.Component()
        coupler_ref = c << coupler
        bend_coupler = cropped_coupler_full(coupling_length=length_x+bend_padding, dx=bend_dx, dy=bend_dy, gap=double_gap, dw=0, cross_section=xsection, crop_bottom_x=crop_bottom_x)
        bend_coupler_ref = c << bend_coupler
        bend_coupler_ref.connect(port="o1", other=coupler_ref.ports["o1"])
        bend_coupler_ref.movex(length_x+bend_padding/2+bend_dx)
        bend_coupler_ref.movey(-double_gap/2-width/2+bend_dy/2)
        
        straight_ringw_ref = c << straight
        straight_ringe_ref = c << straight
        straight_sw_ref = c << straight
        straight_se_ref = c << straight
        straight_nw_ref = c << straight
        straight_ne_ref = c << straight
        straight_ringw_ref.connect(port="o1", other=coupler_ref.ports["o3"])
        straight_ringe_ref.connect(port="o1", other=coupler_ref.ports["o2"])
        straight_sw_ref.connect(port="o1", other=bend_coupler_ref.ports["o4"])
        straight_se_ref.connect(port="o1", other=bend_coupler_ref.ports["o1"])
        straight_nw_ref.connect(port="o1", other=bend_coupler_ref.ports["o3"])
        straight_ne_ref.connect(port="o1", other=bend_coupler_ref.ports["o2"])
        
        if crop_bottom_x > 0:
            extra_straight = gf.components.straight(length=crop_bottom_x, cross_section=xsection)
            extra_straight_ref = c << extra_straight
            extra_straight_ref.connect(port="o2", other=bend_coupler_ref.ports["o2"])
            straight_ne_ref.connect(port="o1", other=extra_straight_ref.ports["o1"])
            c.add_port('o6', port=extra_straight_ref.ports['o1'])
        else:
            c.add_port('o6', port=bend_coupler_ref.ports['o2'])
        c.add_port('o1', port=bend_coupler_ref.ports['o3'])
        c.add_port('o2', port=bend_coupler_ref.ports['o4'])
        c.add_port('o3', port=coupler_ref.ports['o2'])
        c.add_port('o4', port=coupler_ref.ports['o3'])
        c.add_port('o5', port=bend_coupler_ref.ports['o1'])
        c.movey(-c.ports['o1'].center[1])

    return c

@gf.cell
def dd_ring(
    ring_gap= 0.3,
    double_gap=0.2,
    radius=5,
    length_x= 4.0,
    width= 0.5,
    length_extension= 1,
    bend_dx=0,
    bend_dy=0,
    bend_padding=3,
    crop_bottom_x=0,
    layer=(1, 0),
):
    xsection = cross_section(width=width, layer=layer, radius_min=radius/10)
    coupler = gf.components.ring_single(gap=ring_gap, radius=radius, length_x=length_x, bend='bend_euler', straight='straight', cross_section=xsection, length_extension=0)
    straight = gf.components.straight(length=length_extension, cross_section=xsection)

    straight_double = gf.components.straight(length=2*length_extension+length_x, cross_section=xsection)
    if bend_dx == 0 or bend_dy == 0:
        c = gf.Component()
        coupler_ref = c << coupler
        straight_sw_ref = c << straight
        straight_se_ref = c << straight
        straight_double_ref = c << straight_double
        twin_ref = c << straight_double
        straight_sw_ref.connect(port="o1", other=coupler_ref.ports["o2"])
        straight_se_ref.connect(port="o1", other=coupler_ref.ports["o1"])
        straight_double_ref.connect(port="o1", other=coupler_ref.ports["o1"])
        straight_double_ref.movex(length_extension+length_x)
        twin_ref.connect(port="o1", other=coupler_ref.ports["o1"])
        twin_ref.movex(length_extension+length_x)
        straight_double_ref.movey(-double_gap-width)
        c.ports = coupler_ref.ports
        c.ports['o1'].center = (c.ports['o1'].center[0], c.ports['o1'].center[1] - double_gap/2-width/2)
        c.ports['o2'].center = (c.ports['o2'].center[0], c.ports['o2'].center[1] - double_gap/2-width/2)
        c.movey(-c.ports['o1'].center[1])
    else:
        c = gf.Component()
        coupler_ref = c << coupler
        bend_coupler = cropped_coupler_full(coupling_length=length_x+bend_padding, dx=bend_dx, dy=bend_dy, gap=double_gap, dw=0, cross_section=xsection, crop_bottom_x=crop_bottom_x)
        bend_coupler_ref = c << bend_coupler
        bend_coupler_ref.connect(port="o1", other=coupler_ref.ports["o1"])
        bend_coupler_ref.movex(length_x+bend_padding/2+bend_dx)
        bend_coupler_ref.movey(-double_gap/2-width/2+bend_dy/2)
        
        straight_sw_ref = c << straight
        straight_se_ref = c << straight
        straight_nw_ref = c << straight
        straight_ne_ref = c << straight
        straight_sw_ref.connect(port="o1", other=bend_coupler_ref.ports["o4"])
        straight_se_ref.connect(port="o1", other=bend_coupler_ref.ports["o1"])
        straight_nw_ref.connect(port="o1", other=bend_coupler_ref.ports["o3"])
        straight_ne_ref.connect(port="o1", other=bend_coupler_ref.ports["o2"])
        
        if crop_bottom_x > 0:
            extra_straight = gf.components.straight(length=crop_bottom_x, cross_section=xsection)
            extra_straight_ref = c << extra_straight
            extra_straight_ref.connect(port="o2", other=bend_coupler_ref.ports["o2"])
            straight_ne_ref.connect(port="o1", other=extra_straight_ref.ports["o1"])
            c.add_port('o4', port=extra_straight_ref.ports['o1'])
        else:
            c.add_port('o4', port=bend_coupler_ref.ports['o4'])
        c.add_port('o1', port=bend_coupler_ref.ports['o3'])
        c.add_port('o2', port=bend_coupler_ref.ports['o4'])
        c.add_port('o3', port=bend_coupler_ref.ports['o1'])
        c.movey(-c.ports['o1'].center[1])
    return c
