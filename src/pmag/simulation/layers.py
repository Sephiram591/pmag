
import gdsfactory as gf
import numpy as np
from gdsfactory.technology import (
    LayerMap, LayerViews,
)
from gdsfactory.typings import Layer
from pmag.config import PATH

############################################### Hybrid Cavity ###############################################
HybridCav_LViews = LayerViews(filepath=PATH.layer_views / "HybridCav.yaml")

class HybridCav_LMAP(LayerMap):
    WAFER   : Layer = (999, 0)
    NITRIDE : Layer = (1, 0)
    DIAMOND : Layer = (2, 0)

def get_hybrid_layer_stack(oxide_thickness=1, 
                           nitride_thickness=.22, 
                           diamond_thickness=.2):
    layerstack = gf.technology.LayerStack(
        layers={
            "nitride": gf.technology.LayerLevel(
                layer='NITRIDE',
                thickness=nitride_thickness,      # in nm
                zmin=oxide_thickness,             # starting height
                material="nitride",
                mesh_order=1,
            ),
            "diamond": gf.technology.LayerLevel(
                layer='DIAMOND',
                thickness=diamond_thickness,
                zmin=oxide_thickness+nitride_thickness,
                material="diamond",
                mesh_order=2,
            ),
            "oxide": gf.technology.LayerLevel(
                layer='WAFER',
                thickness=oxide_thickness,
                zmin=0,
                material="oxide",
                etch=True,
                mesh_order=3,
            ),
        }
    )
    return layerstack

############################################### Nitride Window ###############################################
NitrideWindow_LViews = LayerViews(filepath=PATH.layer_views / "NitrideWindow.yaml")
class NitrideWindow_LMAP(LayerMap):
    WAFER         : Layer = (999, 0)
    UPPER_NITRIDE : Layer = (1, 0)
    DIAMOND       : Layer = (2, 0)
    LOWER_NITRIDE : Layer = (4, 0)
    WINDOW        : Layer = (5, 0)
    WINDOW2       : Layer = (6, 0)
def get_nitride_window_layer_stack(box_thickness=1, 
                                   upper_nitride_thickness=.22, 
                                   upper_nitride_bottom_depth=1, 
                                   lower_nitride_thickness=.22, 
                                   lower_nitride_top_depth=1.22, 
                                   diamond_thickness=.2,
                                   reflector_diamond_gap=1,
                                   reflector_thickness=0,
                                   window2_box_offset=np.nan):
    
    zmin_lower_nitride = box_thickness
    zmin_upper_nitride = zmin_lower_nitride + lower_nitride_thickness + lower_nitride_top_depth-upper_nitride_bottom_depth
    zmin_window = zmin_upper_nitride + upper_nitride_thickness
    zmin_diamond = np.max([zmin_window, zmin_upper_nitride+upper_nitride_thickness])
    layers={
        "diamond": gf.technology.LayerLevel(
            layer='DIAMOND',
            thickness=diamond_thickness,
            zmin=zmin_diamond,
            material="diamond",
            mesh_order=1,
        ),
        "upper_nitride": gf.technology.LayerLevel(
            layer='UPPER_NITRIDE',
            thickness=upper_nitride_thickness,      # in nm
            zmin=zmin_upper_nitride,             # starting height
            material="nitride",
            mesh_order=2,
        ),
        "window": gf.technology.LayerLevel(
            layer='WINDOW',
            thickness=upper_nitride_bottom_depth,      # in nm
            zmin=zmin_window,             # starting height
            material="air",
            mesh_order=3,
        ),
        "lower_nitride": gf.technology.LayerLevel(
            layer='LOWER_NITRIDE',
            thickness=lower_nitride_thickness,
            zmin=zmin_lower_nitride,
            material="nitride",
            mesh_order=4,
        ),
        "oxide": gf.technology.LayerLevel(
            layer='WAFER',
            thickness=box_thickness+lower_nitride_thickness+lower_nitride_top_depth,
            zmin=0,
            material="oxide",
            mesh_order=5,
        ),
    }
    if reflector_thickness > 0:
        layers["reflector"] = gf.technology.LayerLevel(
            layer='WAFER',
            thickness=reflector_thickness,
            zmin=zmin_diamond-reflector_diamond_gap-reflector_thickness,
            material="reflector",
            mesh_order=1,
        )
    if not np.isnan(window2_box_offset):
        layers["window2"] = gf.technology.LayerLevel(
            layer='WINDOW2',
            thickness=zmin_window-zmin_lower_nitride+upper_nitride_bottom_depth,
            zmin=box_thickness+window2_box_offset,
            material="air",
            mesh_order=3,
        )
    layerstack = gf.technology.LayerStack(
        layers=layers
    )
    return layerstack