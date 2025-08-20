import gdsfactory as gf
import numpy as np
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, Delta


@gf.cell
def elevator(l1, l2, length_extension, base_w1, base_w2, tip_w1, tip_w2, offset, layer_1, layer_2, window_width=0, window_layer='WINDOW', taper_port_markers=False, taper_2=None):
    c = gf.Component()
    # build first waveguide
    if taper_2:
        base_w2 = taper_2.ports['o2'].width
    def taper1_x(**kwargs):
        return gf.cross_section.cross_section(layer=layer_1, radius_min=tip_w1/2, **kwargs)
    def taper2_x(**kwargs):
        return gf.cross_section.cross_section(layer=layer_2, radius_min=tip_w2/2, **kwargs)
    base1_x = gf.cross_section.cross_section(width=base_w1, layer=layer_1, radius_min=base_w1/2)
    base2_x = gf.cross_section.cross_section(width=base_w2, layer=layer_2, radius_min=base_w2/2)

    taper_1 = gf.components.taper(width1=base_w1, width2=tip_w1, length=l1, cross_section=taper1_x)
    if taper_2 is None:
        taper_2 = gf.components.taper(width1=tip_w2, width2=base_w2, length=l2, cross_section=taper2_x)
        base_w2 = taper_2_ref.ports['o2'].width


    taper_1_ref = c << taper_1
    taper_2_ref = c << taper_2

    taper_1_ref.connect('o1', taper_2_ref.ports['o2'], allow_layer_mismatch=True, allow_width_mismatch=True)
    taper_2_ref.movex(offset)

    taper_1_east_post = taper_1_ref.ports['o1'].center[0] # Marks where the straight waveguide will be connected to the taper
    taper_1_west_post = np.min([taper_1_east_post, taper_2_ref.ports['o1'].center[0]])-length_extension

    taper_2_west_post = taper_2_ref.ports['o2'].center[0] # Marks where the straight waveguide will be connected to the taper
    taper_2_east_post = np.max([taper_2_west_post, taper_1_ref.ports['o2'].center[0]])+length_extension

    # connect waveguides
    wg1 = gf.components.straight(width=base_w1, length=taper_1_east_post-taper_1_west_post, cross_section=base1_x)
    wg1_ref = c << wg1
    wg1_ref.connect('o1', taper_1_ref.ports['o1'], allow_layer_mismatch=True, allow_width_mismatch=False)
    wg2 = gf.components.straight(width=base_w2, length=taper_2_east_post-taper_2_west_post, cross_section=base2_x)
    wg2_ref = c << wg2
    wg2_ref.connect('o1', taper_2_ref.ports['o2'], allow_layer_mismatch=True, allow_width_mismatch=False)

    

    c.add_port('o1', port=wg1_ref.ports['o2'])
    c.add_port('o2', port=wg2_ref.ports['o2'])
    if taper_port_markers:
        c.add_port('i1', port=taper_2_ref.ports['o2'])
        c.add_port('i2', port=taper_1_ref.ports['o1'])
    min_x = c.ports['o1'].center[0]
    max_x = c.ports['o2'].center[0]
    window_etch = gf.components.rectangle(size=(max_x-min_x, window_width), layer=window_layer, centered=True)
    window_etch_ref = c << window_etch
    window_etch_ref.movex((min_x+max_x)/2)

    return c 
    

@gf.cell
def directional_mixer(
    coupling_length: float = 40.0,
    mixer_length: float | None = None,
    dx: Delta = 10.0,
    dy: Delta = 4.8,
    coupler_layer: str = "nitride",
    mixer_layer: str = "nitride",
    width_coupler: float | None = None,
    width_mixer: float | None = None,
    width_mixer_tip: float = 0.0,
) -> Component:
    """Adiabatic elevator mixer

    Args:
        coupling_length: Length of the coupling region in um.
        mixer_length: Length of the mixer region in um.
        dx: Length of the bend regions in um.
        dy: Port-to-port distance between the bend regions in um.
        cross_section_coupler: cross-section spec for the coupler.
        cross_section_mixer: cross-section spec for the mixer.
        width_coupler: width of the coupler waveguide. If None, it will use the width of the cross_section.
        width_mixer: width of the mixer waveguide. If None, it will use the width of the cross_section.
        width_mixer_tip: width of the mixer waveguide at the tip.

    """
    c = gf.Component()

    def coupler_x(**kwargs):
        return gf.cross_section.cross_section(layer=coupler_layer, radius_min=width_coupler/2, **kwargs)
    def mixer_x(**kwargs):
        return gf.cross_section.cross_section(layer=mixer_layer, radius_min=width_mixer/2, **kwargs)
    if mixer_length is None:
        mixer_length = coupling_length

    if width_coupler:
        x_coupler = coupler_x(width=width_coupler)
    else:
        x_coupler = coupler_x()
    
    if width_mixer:
        x_mixer = mixer_x(width=width_mixer)
    else:
        x_mixer = mixer_x()

    straight_coupler = c << gf.components.straight(
        length=coupling_length,
        cross_section=x_coupler,
    )
    taper_mixer = c << gf.components.taper(
        length=mixer_length,
        width1=x_mixer.width,
        width2=width_mixer_tip,
        cross_section=mixer_x,
    )

    bend_input_top = c << gf.c.bend_s(
        size=(dx, (dy) / 2.0), cross_section=x_coupler
    )

    # bend_input_bottom = c << gf.c.bend_s(
    #     size=(dx, (-dy) / 2.0), cross_section=x_mixer
    # )

    straight_coupler.connect("o1", bend_input_top.ports["o1"])

    bend_output_top = c << gf.c.bend_s(
        size=(dx, (dy) / 2.0), cross_section=x_coupler
    )
    bend_output_top.connect("o2", straight_coupler.ports["o2"], mirror=True)
    taper_mixer.connect("o1", bend_output_top.ports["o2"], allow_layer_mismatch=True, allow_width_mismatch=True)

    bend_input_bottom = c << gf.c.bend_s(
        size=(dx, (-dy) / 2.0), cross_section=x_mixer
    )

    bend_input_bottom.connect("o2", taper_mixer.ports["o1"], mirror=True)
    # x.add_bbox(c)

    c.add_port("o1", port=bend_output_top.ports["o1"])
    c.add_port("o2", port=bend_input_bottom.ports["o1"])
    c.add_port("o3", port=bend_input_top.ports["o2"])
    # c.auto_rename_ports()

    c.flatten()
    return c