import gdsfactory as gf
from .layers import *
from gdsfactory.generic_tech import get_generic_pdk
from gdsfactory.cross_section import CrossSection, cross_section, xsection
from gdsfactory.typings import LayerSpec
generic_pdk = get_generic_pdk()


@xsection
def custom_x(
    width: float = 0.5,
    layer: LayerSpec = (1, 0),
    radius: float = 10.0,
    radius_min: float = 0.01,
    **kwargs,
) -> CrossSection:
    """Return Strip cross_section."""
    return cross_section(
        width=width,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
        **kwargs,
    )

# hybrid_layers = ['NITRIDE', 'DIAMOND', 'WAFER']
# nitride_window_layers = ['DIAMOND', 'UPPER_NITRIDE', 'WINDOW', 'LOWER_NITRIDE', 'WAFER']

# hybrid_xsections = {}
# for l in hybrid_layers:
#     hybrid_xsections[l+'_x'] = lambda **kwargs: custom_x(layer=l, **kwargs)

# nitride_window_xsections = {}
# for l in nitride_window_layers:
#     nitride_window_xsections[l+'_x'] = lambda **kwargs: custom_x(layer=l, **kwargs)



def get_hybrid_cav_pdk():
    HybridCav_PDK = gf.Pdk(
        name="HybridCav_PDK",
        layers=HybridCav_LMAP,
        layer_views=HybridCav_LViews,
        cells=generic_pdk.cells,
        # cross_sections=hybrid_xsections,
    )
    return HybridCav_PDK

def get_nitride_window_pdk():
    NitrideWindow_PDK = gf.Pdk(
        name="NitrideWindow_PDK",
        layers=NitrideWindow_LMAP,
        layer_views=NitrideWindow_LViews,
        cells=generic_pdk.cells,
        # cross_sections=nitride_window_xsections,
    )
    return NitrideWindow_PDK

