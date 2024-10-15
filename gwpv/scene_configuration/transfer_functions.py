import numpy as np
import copy
from cmap import Colormap

def polynomalTransform(transfer_fctn, degree, shift):
    for i in range(0, len(transfer_fctn.RGBPoints), 4):
        shifted_point = transfer_fctn.RGBPoints[i] + shift
        transfer_fctn.RGBPoints[i] = np.sign(shifted_point)*pow(abs(shifted_point), degree)

def apply_colormap(transfer_fctn, tf_config):
    colormap_config = tf_config["Colormap"]
    if isinstance(colormap_config, str):
        transfer_fctn.ApplyPreset(colormap_config, False)
        return
    elif "Name" in colormap_config:
        transfer_fctn.ApplyPreset(colormap_config["Name"], False)
    elif "Points" in colormap_config:
        rgb_points = []
        for point in colormap_config["Points"]:
            rgb_points.append(point["Position"])
            rgb_points += point["Color"]
        transfer_fctn.RGBPoints = rgb_points
    elif "Exported" in colormap_config:
        transfer_fctn.RGBPoints = colormap_config["Exported"]
    if "Invert" in colormap_config and colormap_config["Invert"]:
        transfer_fctn.InvertTransferFunction()
    if "ColorSpace" in colormap_config:
        transfer_fctn.ColorSpace = colormap_config["ColorSpace"]
    if "Logarithmic" in colormap_config:
        transfer_fctn.UseLogScale = colormap_config["Logarithmic"]
    if "PolynomalTransform" in colormap_config:
        polTrans = colormap_config["PolynomalTransform"]
        if isinstance(polTrans, int):
            polynomalTransform(transfer_fctn, polTrans, 0)
        else:
            polynomalTransform(transfer_fctn, polTrans["Degree"], polTrans["Shift"])

def set_categories(scalarBarTransfer_fctn, rgb_points, classes):  
    scalarBarTransfer_fctn.Annotations = sum([[str(c), "{0:.2g}".format(c)] for c in classes], [])
    minpos = rgb_points[0]
    maxpos = rgb_points[-4]
    span = maxpos - minpos
    classmap = Colormap([[(rgb_points[i]-minpos)/span, *rgb_points[i+1:i+4], 1.0] for i in range(0, len(rgb_points), 4)])
    classmap
    mappedvalues = classmap([(c-minpos)/span for c in classes]).flatten()
    scalarBarTransfer_fctn.IndexedColors = np.delete(mappedvalues, np.arange(0, len(mappedvalues), 4) + 3)



def set_opacity_function_points(opacity_fctn, opacity_fctn_points):
    # TODO: Make this flattening less crude..
    flat_pnts = []
    for i in opacity_fctn_points:
        for j in i:
            for k in j:
                flat_pnts.append(k)
    opacity_fctn.Points = flat_pnts


def configure_linear_transfer_function(transfer_fctn, opacity_fctn, scalarBarTransfer_fctn, tf_config):
    apply_colormap(transfer_fctn, tf_config)
    apply_colormap(scalarBarTransfer_fctn, tf_config)

    start_pos, opacity_start = (
        tf_config["Start"]["Position"],
        tf_config["Start"]["Opacity"],
    )
    end_pos, opacity_end = (
        tf_config["End"]["Position"],
        tf_config["End"]["Opacity"],
    )
    
    transfer_fctn.RescaleTransferFunction(start_pos, end_pos)
    scalarBarTransfer_fctn.RescaleTransferFunction(start_pos, end_pos)

    set_opacity_function_points(
        opacity_fctn,
        [
            [
                (start_pos, opacity_start, 0.5, 0.0),
                (end_pos, opacity_end, 0.5, 0.0),
            ]
        ],
    )


def configure_peaks_transfer_function(transfer_fctn, opacity_fctn, scalarBarTransfer_fctn, tf_config):
    apply_colormap(transfer_fctn, tf_config)
    apply_colormap(scalarBarTransfer_fctn, tf_config)

    first_peak = tf_config["FirstPeak"]["Position"]
    last_peak = tf_config["LastPeak"]["Position"]
    
    num_peaks = tf_config["NumPeaks"]
    if "Logarithmic" in tf_config and tf_config["Logarithmic"]:
        peaks = np.logspace(
            np.log10(first_peak), np.log10(last_peak), num_peaks, base=10
        )
    elif "Polynomial" in tf_config:
        peaks = first_peak + (last_peak - first_peak) * pow(np.linspace(
            0, 1, num_peaks
        ), tf_config["Polynomial"])
    else:
        peaks = first_peak + (last_peak - first_peak) * np.linspace(
            0, 1, num_peaks
        )
    tf_decay = list(np.diff(peaks) / 2)
    tf_decay.append(tf_decay[-1])

    if "AddNegatives" in tf_config and tf_config["AddNegatives"]:
        peaks = [*[-d for d in peaks[::-1]], *peaks]
        tf_decay = [*[-d for d in tf_decay[::-1]], *tf_decay]
    
    if "CustomOpacities" in tf_config:
        opacities = tf_config["CustomOpacities"]
    else:
        opacity_first_peak = tf_config["FirstPeak"]["Opacity"]
        opacity_last_peak = tf_config["LastPeak"]["Opacity"]
        if "PolynomialOpacities" in tf_config:
            opacities = opacity_first_peak + (opacity_last_peak - opacity_first_peak) * pow(np.linspace(
                0, 1, num_peaks
            ), tf_config["PolynomialOpacities"])
        else:
            opacities = opacity_first_peak + (opacity_last_peak - opacity_first_peak) * np.linspace(
                0, 1, num_peaks
            )
        
        if "AddNegatives" in tf_config and tf_config["AddNegatives"]:
            opacities = [*[d for d in opacities[::-1]], *opacities]

    minColorPos = peaks[0]
    maxColorPos = peaks[-1]
    if "ColorRange" in tf_config:
        minColorPos = tf_config["ColorRange"][0]
        maxColorPos = tf_config["ColorRange"][1]

    transfer_fctn.RescaleTransferFunction(minColorPos, maxColorPos)
    scalarBarTransfer_fctn.RescaleTransferFunction(minColorPos, maxColorPos)
    set_opacity_function_points(
        opacity_fctn,
        [
            [
                (peak - peak_decay / 100.0, 0.0, 0.5, 0.0),
                (peak, opacity, 0.5, 0.0),
                (peak + peak_decay, 0.0, 0.5, 0.0),
            ]
            for peak, peak_decay, opacity in zip(peaks, tf_decay, opacities)
        ],
    )

    set_categories(scalarBarTransfer_fctn, scalarBarTransfer_fctn.RGBPoints, peaks)

def configure_custom_peaks_transfer_function(transfer_fctn, opacity_fctn, scalarBarTransfer_fctn, tf_config):
    apply_colormap(transfer_fctn, tf_config)
    apply_colormap(scalarBarTransfer_fctn, tf_config)

    peaks = tf_config["Peaks"]

    tf_decay = list(np.diff([peak["Position"] for peak in peaks]) / 2)
    tf_decay.append(tf_decay[-1])

    if "AddNegatives" in tf_config and tf_config["AddNegatives"]:
        inverse_peaks = copy.deepcopy(peaks[::-1])
        for i in range(len(inverse_peaks)):
            inverse_peaks[i]["Position"] *= -1
        peaks = [*inverse_peaks, *peaks]
        tf_decay = [*[-d for d in tf_decay[::-1]], *tf_decay]
    
    minColorPos = peaks[0]
    maxColorPos = peaks[-1]
    if "ColorRange" in tf_config:
        minColorPos = tf_config["ColorRange"][0]
        maxColorPos = tf_config["ColorRange"][1]

    transfer_fctn.RescaleTransferFunction(minColorPos, maxColorPos)
    scalarBarTransfer_fctn.RescaleTransferFunction(minColorPos, maxColorPos)
    set_opacity_function_points(
        opacity_fctn,
        [
            [
                (peak["Position"] - peak_decay / 100.0, 0.0, 0.5, 0.0),
                (peak["Position"], peak["Opacity"], 0.5, 0.0),
                (peak["Position"] + peak_decay, 0.0, 0.5, 0.0),
            ]
            for i, (peak, peak_decay) in enumerate(zip(peaks, tf_decay))
        ],
    )
        
    set_categories(scalarBarTransfer_fctn, scalarBarTransfer_fctn.RGBPoints, [peak["Position"] for peak in peaks])
    

def configure_custom_transfer_function(transfer_fctn, opacity_fctn, scalarBarTransfer_fctn, tf_config):
    apply_colormap(transfer_fctn, tf_config)
    apply_colormap(scalarBarTransfer_fctn, tf_config)
    points = tf_config["Points"]
    transfer_fctn.RescaleTransferFunction(
        points[0]["Position"], points[-1]["Position"]
    )
    scalarBarTransfer_fctn.RescaleTransferFunction(
        points[0]["Position"], points[-1]["Position"]
    )
    set_opacity_function_points(
        opacity_fctn,
        [[(point["Position"], point["Opacity"], 0.5, 0.0)] for point in points],
    )


def configure_transfer_function(transfer_fctn, opacity_fctn, scalarBarTransfer_fctn, tf_config):
    # Dispatch to particular type of transfer function
    supported_types = ["Linear", "Peaks", "Custom", "CustomPeaks"]
    assert len(tf_config) == 1, (
        "The transfer function configuration should have one entry which is"
        " the type of the transfer function. Currently supported are:"
        f" {supported_types}"
    )
    tf_type = list(tf_config.keys())[0]
    assert tf_type in supported_types, (
        f"Transfer function type '{tf_type}' not supported. Currently supported"
        f" are: {supported_types}"
    )
    if tf_type == "Linear":
        configure_linear_transfer_function(
            transfer_fctn, opacity_fctn, scalarBarTransfer_fctn, tf_config["Linear"]
        )
    elif tf_type == "Peaks":
        configure_peaks_transfer_function(
            transfer_fctn, opacity_fctn, scalarBarTransfer_fctn, tf_config["Peaks"]
        )
    elif tf_type == "CustomPeaks":
        configure_custom_peaks_transfer_function(
            transfer_fctn, opacity_fctn, scalarBarTransfer_fctn, tf_config["CustomPeaks"]
        )
    elif tf_type == "Custom":
        configure_custom_transfer_function(
            transfer_fctn, opacity_fctn, scalarBarTransfer_fctn, tf_config["Custom"]
        )
    # Enable opacity for surface representations
    transfer_fctn.EnableOpacityMapping = True
