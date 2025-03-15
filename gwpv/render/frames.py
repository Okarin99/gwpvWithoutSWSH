from __future__ import division

import logging
import os
import sys
import time

import h5py
import numpy as np
import paraview.servermanager as pvserver
import paraview.simple as pv

import gwpv.scene_configuration.color as config_color
import gwpv.scene_configuration.transfer_functions as tf
from gwpv.render.background import set_background
from gwpv.scene_configuration import animate, camera_motion, parse_as

if sys.version_info >= (3, 10):
    from importlib.resources import as_file, files
else:
    from importlib_resources import as_file, files

logger = logging.getLogger(__name__)

pv._DisableFirstRenderCameraReset()

# Load plugins
# - We need to load the plugins outside the `render` function to make
# them work with `multiprocessing`.
# - This has problems with `pvbatch`. We could use `PV_PLUGIN_PATH` env variable
# to load plugins, but then we have to get the client proxy somehow. Here's an
# attempt using ParaView-internal functions:
# WaveformDataReader = pv._create_func("WaveformDataReader", pvserver.sources)
# WaveformToVolume = pv._create_func("WaveformToVolume", pvserver.filters)
logger.info("Loading ParaView plugins...")
plugins_dir = files("gwpv.paraview_plugins")
load_plugins = [
    "WaveformDataReader.py",
    "TrajectoryDataReader.py",
    "FollowTrajectory.py",
    "TrajectoryTail.py",
    # 'SwshGrid.py'
]
for plugin in load_plugins:
    with as_file(plugins_dir / plugin) as plugin_path:
        pv.LoadPlugin(str(plugin_path), remote=False, ns=globals())
logger.info("ParaView plugins loaded.")

# Work around https://gitlab.kitware.com/paraview/paraview/-/issues/21457
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
sys.stdin = sys.__stdin__


def render_frames(
    scene,
    frames_dir=None,
    frame_window=None,
    render_missing_frames=False,
    save_state_to_file=None,
    no_render=False,
    show_preview=False,
):
    """Render the frames for the `scene`

    This function `yield`s progress updates.
    """
    # Validate scene
    if scene["View"]["ViewSize"][0] % 16 != 0:
        logger.warning(
            "The view width should be a multiple of 16 to be compatible with"
            " QuickTime."
        )
    if scene["View"]["ViewSize"][1] % 2 != 0:
        logger.warning(
            "The view height should be even to be compatible with QuickTime."
        )

    render_start_time = time.time()

    # Setup layout
    layout = pv.CreateLayout("Layout")

    # Setup view
    if "Background" in scene["View"]:
        bg_config = scene["View"]["Background"]
        del scene["View"]["Background"]
    else:
        bg_config = None
    view = pv.CreateRenderView(**scene["View"])
    pv.AssignViewToLayout(view=view, layout=layout, hint=0)

    # Set background
    set_background(
        bg_config=bg_config,
        view=view,
        datasources=scene.get("Datasources", None),
    )

    # Load the waveform data file
    waveform_h5file, waveform_subfile = parse_as.file_and_subfile(
        scene["Datasources"]["Waveform"]
    )
    waveform_data = WaveformDataReader(
        FileName=waveform_h5file, Subfile=waveform_subfile
    )
    pv.UpdatePipeline()

    # Generate volume data from the waveform. Also sets the available time range.
    # TODO: Pull KeepEveryNthTimestep out of datasource
    waveform_to_volume_configs = scene["WaveformToVolume"]
    if isinstance(waveform_to_volume_configs, dict):
        waveform_to_volume_configs = [
            {
                "Object": waveform_to_volume_configs,
            }
        ]
        if "VolumeRepresentation" in scene:
            waveform_to_volume_configs[0]["VolumeRepresentation"] = scene[
                "VolumeRepresentation"
            ]
    waveform_to_volume_objects = []
    for waveform_to_volume_config in waveform_to_volume_configs:
        volume_data = WaveformDataReader(
            FileName=waveform_h5file, Subfile=waveform_subfile,
            **waveform_to_volume_config["Object"]
        )

        if "Polarizations" in waveform_to_volume_config["Object"]:
            volume_data.Polarizations = waveform_to_volume_config["Object"][
                "Polarizations"
            ]
        waveform_to_volume_objects.append(volume_data)

    # Compute timing and frames information
    interact = False
    if "FreezeTime" in scene["Animation"]:
        frozen_time = scene["Animation"]["FreezeTime"]
        if not(isinstance(frozen_time, float)):
            if "Interact" in frozen_time:
                interact = frozen_time["Interact"]
            frozen_time = frozen_time["Time"]

        logger.info(f"Freezing time at {frozen_time}.")
        view.ViewTime = frozen_time
        animation = None
        yield dict(total=1)
        yield dict(start=True)
    else:
        time_range_in_M = scene["Animation"]["Crop"]
        animation_speed = scene["Animation"]["Speed"]
        frame_rate = scene["Animation"]["FrameRate"]
        num_frames = animate.num_frames(
            max_animation_length=time_range_in_M[1] - time_range_in_M[0],
            animation_speed=animation_speed,
            frame_rate=frame_rate,
        )
        animation_length_in_seconds = num_frames / frame_rate
        animation_length_in_M = animation_length_in_seconds * animation_speed
        time_per_frame_in_M = animation_length_in_M / num_frames
        logger.info(
            f"Rendering {animation_length_in_seconds:.2f}s movie with"
            f" {num_frames} frames ({frame_rate} FPS or"
            f" {animation_speed:.2e} M/s or"
            f" {time_per_frame_in_M:.2e} M/frame)..."
        )
        if frame_window is not None:
            animation_window_num_frames = frame_window[1] - frame_window[0]
            animation_window_time_range = (
                time_range_in_M[0] + frame_window[0] * time_per_frame_in_M,
                time_range_in_M[0]
                + (frame_window[1] - 1) * time_per_frame_in_M,
            )
            logger.info(
                f"Restricting rendering to {animation_window_num_frames} frames"
                f" (numbers {frame_window[0]} to {frame_window[1] - 1})."
            )
        else:
            animation_window_num_frames = num_frames
            animation_window_time_range = time_range_in_M
            frame_window = (0, num_frames)
        yield dict(total=animation_window_num_frames)

        # Setup animation so that sources can retrieve the `UPDATE_TIME_STEP`
        animation = pv.GetAnimationScene()
        # animation.UpdateAnimationUsingDataTimeSteps()
        # Since the data can be evaluated at arbitrary times we define the time steps
        # here by setting the number of frames within the full range
        animation.PlayMode = "Sequence"
        animation.StartTime = animation_window_time_range[0]
        animation.EndTime = animation_window_time_range[1]
        animation.NumberOfFrames = animation_window_num_frames
        logger.debug(
            f"Animating from scene time {animation.StartTime} to"
            f" {animation.EndTime} in {animation.NumberOfFrames} frames."
        )

        def scene_time_from_real(real_time):
            return (
                real_time / animation_length_in_seconds * animation_length_in_M
            )

        # For some reason the keyframe time for animations is expected to be within
        # (0, 1) so we need to transform back and forth from this "normalized" time
        def scene_time_from_normalized(normalized_time):
            return animation.StartTime + normalized_time * (
                animation.EndTime - animation.StartTime
            )

        def normalized_time_from_scene(scene_time):
            return (scene_time - animation.StartTime) / (
                animation.EndTime - animation.StartTime
            )

        # Report start of rendering here so volume data computing for initial
        # frame is measured
        yield dict(start=True)

        # Set the initial time step
        animation.GoToFirst()

    # Display the volume data. This will trigger computing the volume data at the
    # current time step.
    for volume_data, waveform_to_volume_config in zip(
        waveform_to_volume_objects, waveform_to_volume_configs
    ):
        vol_repr = (
            waveform_to_volume_config["VolumeRepresentation"]
            if "VolumeRepresentation" in waveform_to_volume_config
            else {}
        )
        volume_color_by = config_color.extract_color_by(vol_repr)
        if (
            vol_repr["VolumeRenderingMode"] == "GPU Based"
            and len(volume_color_by) > 2
        ):
            logger.warning(
                "The 'GPU Based' volume renderer doesn't support multiple"
                " components."
            )
        volume = pv.Show(volume_data, view, **vol_repr)
        pv.ColorBy(volume, value=volume_color_by)

    if "Slices" in scene:
        for slice_config in scene["Slices"]:
            slice_obj_config = slice_config.get("Object", {})
            slice = pv.Slice(Input=volume_data)
            slice.SliceType = "Plane"
            slice.SliceOffsetValues = [0.0]
            slice.SliceType.Origin = slice_obj_config.get(
                "Origin", [0.0, 0.0, -0.3]
            )
            slice.SliceType.Normal = slice_obj_config.get(
                "Normal", [0.0, 0.0, 1.0]
            )
            slice_rep = pv.Show(
                slice, view, **slice_config.get("Representation", {})
            )
            pv.ColorBy(slice_rep, value=volume_color_by)

    # Display the time
    if "TimeAnnotation" in scene:
        time_annotation = pv.AnnotateTimeFilter(
            volume_data, **scene["TimeAnnotation"]
        )
        pv.Show(time_annotation, view, **scene["TimeAnnotationRepresentation"])

    if "Annotations" in scene:
        for annotation in scene["Annotations"]:
            text = pv.Text()
            text.Text = annotation["Text"]
            del annotation["Text"]
            pv.Show(text, view, **annotation)

    # Add spheres
    if "Spheres" in scene:
        for sphere_config in scene["Spheres"]:
            sphere = pv.Sphere(**sphere_config["Object"])
            pv.Show(sphere, view, **sphere_config["Representation"])

    # Add trajectories and objects that follow them
    if "Trajectories" in scene:
        for trajectory_config in scene["Trajectories"]:
            trajectory_name = trajectory_config["Name"]
            radial_scale = (
                trajectory_config["RadialScale"]
                if "RadialScale" in trajectory_config
                else 1.0
            )
            # Load the trajectory data
            traj_data_reader = TrajectoryDataReader(
                RadialScale=radial_scale,
                **scene["Datasources"]["Trajectories"][trajectory_name],
            )
            # Make sure the data is loaded so we can retrieve timesteps.
            # TODO: This should be fixed in `TrajectoryDataReader` by
            # communicating time range info down the pipeline, but we had issues
            # with that (see also `WaveformDataReader`).
            traj_data_reader.UpdatePipeline()
            if "Objects" in trajectory_config:
                with animate.restore_animation_state(animation):
                    follow_traj = FollowTrajectory(
                        TrajectoryData=traj_data_reader
                    )
                for traj_obj_config in trajectory_config["Objects"]:
                    for traj_obj_key in traj_obj_config:
                        if traj_obj_key in [
                            "Representation",
                            "Visibility",
                            "TimeShift",
                            "Glyph",
                        ]:
                            continue
                        traj_obj_type = getattr(pv, traj_obj_key)
                        traj_obj_glyph = traj_obj_type(
                            **traj_obj_config[traj_obj_key]
                        )
                        follow_traj.UpdatePipeline()
                        traj_obj = pv.Glyph(
                            Input=follow_traj, GlyphType=traj_obj_glyph
                        )
                        # Can't set this in the constructor for some reason
                        traj_obj.ScaleFactor = 1.0
                        for glyph_property in (
                            traj_obj_config["Glyph"]
                            if "Glyph" in traj_obj_config
                            else []
                        ):
                            setattr(
                                traj_obj,
                                glyph_property,
                                traj_obj_config["Glyph"][glyph_property],
                            )
                        traj_obj.UpdatePipeline()
                        if "TimeShift" in traj_obj_config:
                            traj_obj = animate.apply_time_shift(
                                traj_obj, traj_obj_config["TimeShift"]
                            )
                        pv.Show(traj_obj, view, **traj_obj_config["Representation"])
                        if "Visibility" in traj_obj_config:
                            animate.apply_visibility(
                                traj_obj,
                                traj_obj_config["Visibility"],
                                normalized_time_from_scene,
                                scene_time_from_real,
                            )
            if "Tail" in trajectory_config:
                with animate.restore_animation_state(animation):
                    traj_tail = TrajectoryTail(TrajectoryData=traj_data_reader)
                if "TimeShift" in trajectory_config:
                    traj_tail = animate.apply_time_shift(
                        traj_tail, trajectory_config["TimeShift"]
                    )
                tail_config = trajectory_config["Tail"]
                traj_color_by = config_color.extract_color_by(tail_config)
                if "Visibility" in tail_config:
                    tail_visibility_config = tail_config["Visibility"]
                    del tail_config["Visibility"]
                else:
                    tail_visibility_config = None
                tail_rep = pv.Show(traj_tail, view, **tail_config)
                pv.ColorBy(tail_rep, value=traj_color_by)
                if tail_visibility_config is not None:
                    animate.apply_visibility(
                        traj_tail,
                        tail_visibility_config,
                        normalized_time_from_scene=normalized_time_from_scene,
                        scene_time_from_real=scene_time_from_real,
                    )
            if "Move" in trajectory_config:
                move_config = trajectory_config["Move"]
                logger.debug(
                    f"Animating '{move_config['guiName']}' along trajectory."
                )
                with h5py.File(trajectory_file, "r") as traj_data_file:
                    trajectory_data = np.array(
                        traj_data_file[trajectory_subfile]
                    )
                if radial_scale != 1.0:
                    trajectory_data[:, 1:] *= radial_scale
                logger.debug(f"Trajectory data shape: {trajectory_data.shape}")
                animate.follow_path(
                    gui_name=move_config["guiName"],
                    trajectory_data=trajectory_data,
                    num_keyframes=move_config["NumKeyframes"],
                    scene_time_range=time_range_in_M,
                    normalized_time_from_scene=normalized_time_from_scene,
                )

    # Add non-spherical horizon shapes (instead of spherical objects following
    # trajectories)
    if "Horizons" in scene:
        for horizon_config in scene["Horizons"]:
            with animate.restore_animation_state(animation):
                horizon = pv.PVDReader(
                    FileName=scene["Datasources"]["Horizons"][
                        horizon_config["Name"]
                    ]
                )
                if horizon_config.get("InterpolateTime", False):
                    horizon = pv.TemporalInterpolator(
                        Input=horizon, DiscreteTimeStepInterval=0
                    )
            if "TimeShift" in horizon_config:
                horizon = animate.apply_time_shift(
                    horizon, horizon_config["TimeShift"], animation
                )
            # Try to make horizon surfaces smooth. At low angular resoluton
            # they still show artifacts, so perhaps more can be done.
            horizon = pv.ExtractSurface(Input=horizon)
            horizon = pv.GenerateSurfaceNormals(Input=horizon)
            horizon_rep_config = horizon_config.get("Representation", {})
            if "Representation" not in horizon_rep_config:
                horizon_rep_config["Representation"] = "Surface"
            if "AmbientColor" not in horizon_rep_config:
                horizon_rep_config["AmbientColor"] = [0.0, 0.0, 0.0]
            if "DiffuseColor" not in horizon_rep_config:
                horizon_rep_config["DiffuseColor"] = [0.0, 0.0, 0.0]
            if "Specular" not in horizon_rep_config:
                horizon_rep_config["Specular"] = 0.2
            if "SpecularPower" not in horizon_rep_config:
                horizon_rep_config["SpecularPower"] = 10
            if "SpecularColor" not in horizon_rep_config:
                horizon_rep_config["SpecularColor"] = [1.0, 1.0, 1.0]
            if "ColorBy" in horizon_rep_config:
                horizon_color_by = config_color.extract_color_by(
                    horizon_rep_config
                )
            else:
                horizon_color_by = None
            horizon_rep = pv.Show(horizon, view, **horizon_rep_config)
            if horizon_color_by is not None:
                pv.ColorBy(horizon_rep, value=horizon_color_by)
            # Animate visibility
            if "Visibility" in horizon_config:
                animate.apply_visibility(
                    horizon,
                    horizon_config["Visibility"],
                    normalized_time_from_scene=normalized_time_from_scene,
                    scene_time_from_real=scene_time_from_real,
                )
            if "Contours" in horizon_config:
                for contour_config in horizon_config["Contours"]:
                    contour = pv.Contour(
                        Input=horizon, **contour_config["Object"]
                    )
                    contour_rep = pv.Show(
                        contour, view, **contour_config["Representation"]
                    )
                    pv.ColorBy(contour_rep, None)
                    if "Visibility" in horizon_config:
                        animate.apply_visibility(
                            contour,
                            horizon_config["Visibility"],
                            normalized_time_from_scene=normalized_time_from_scene,
                            scene_time_from_real=scene_time_from_real,
                        )

    # Configure transfer functions
    if "TransferFunctions" in scene:
        for tf_config in scene["TransferFunctions"]:
            colored_field = tf_config["Field"]
            transfer_fctn = pv.GetColorTransferFunction(colored_field)
            opacity_fctn = pv.GetOpacityTransferFunction(colored_field)
            scalarBarTransfer_fctn = pv.GetColorTransferFunction(colored_field + "_ScalarBar")

            tf.configure_transfer_function(
                transfer_fctn, opacity_fctn, scalarBarTransfer_fctn, tf_config["TransferFunction"]
            )

            if "ScalarBar" in tf_config:
                scalar_bar_config = tf_config["ScalarBar"]
                scalar_bar = pv.GetScalarBar(scalarBarTransfer_fctn, view)
                scalar_bar.Visibility = True
                scalar_bar.ComponentTitle = ""

                if isinstance(scalar_bar_config, str):
                    scalar_bar.Title = scalar_bar_config;
                else:
                    title_config = scalar_bar_config["Title"]
                    if isinstance(title_config, str):
                        scalar_bar.Title = title_config
                    else:
                        if "Name" in title_config:
                            scalar_bar.Title = title_config["Name"]
                        if "Bold" in title_config:
                            scalar_bar.TitleBold = title_config["Bold"]
                        if "Color" in title_config:
                            scalar_bar.TitleColor = title_config["Color"]
                        if "FontFamily" in title_config:
                            scalar_bar.TitleFontFamily = title_config["FontFamily"]
                        if "FontSize" in title_config:
                            scalar_bar.TitleFontSize = title_config["FontSize"]
                        if "Italic" in title_config:
                            scalar_bar.TitleItalic = title_config["Italic"]
                        if "Opacity" in title_config:
                            scalar_bar.TitleOpacity = title_config["Opacity"] 


                    if "Position" in scalar_bar_config:
                        scalar_bar.Position = scalar_bar_config["Position"]
                        scalar_bar.WindowLocation = "Any Location"
                    elif "WindowLocation" in scalar_bar_config:
                        scalar_bar.WindowLocation = scalar_bar_config["WindowLocation"]

                    if "Orientation" in scalar_bar_config:
                        scalar_bar.AutoOrient = False
                        scalar_bar.Orientation = scalar_bar_config["Orientation"]
                    else:
                        scalar_bar.AutoOrient = True

                    if "UseCategories" in scalar_bar_config and scalar_bar_config["UseCategories"]:
                        scalarBarTransfer_fctn.InterpretValuesAsCategories = 1
                    else:
                        scalarBarTransfer_fctn.InterpretValuesAsCategories = 0
                        scalarBarTransfer_fctn.Annotations = None
                    
                    if "Label" in scalar_bar_config:
                        label_config = scalar_bar_config["Label"]
                        if "Bold" in label_config:
                            scalar_bar.LabelBold = label_config["Bold"]
                        if "Color" in label_config:
                            scalar_bar.LabelColor = label_config["Color"]
                        if "FontFamily" in label_config:
                            scalar_bar.LabelFontFamily = label_config["FontFamily"]
                        if "FontSize" in label_config:
                            scalar_bar.LabelFontSize = label_config["FontSize"]
                        if "Italic" in label_config:
                            scalar_bar.LabelItalic = label_config["Italic"]
                        if "Opacity" in label_config:
                            scalar_bar.LabelOpacity = label_config["Opacity"] 
            

    # Save state file before configuring camera keyframes.
    # TODO: Make camera keyframes work with statefile
    if save_state_to_file is not None:
        pv.SaveState(save_state_to_file + ".pvsm")

    # Camera shots
    # TODO: Make this work with freezing time while the camera is swinging
    if animation is None:
        for i, shot in enumerate(scene["CameraShots"]):
            if (
                i == len(scene["CameraShots"]) - 1
                or (shot["Time"] if "Time" in shot else 0.0) >= view.ViewTime
            ):
                camera_motion.apply(shot)
                break
    else:
        camera_motion.apply_swings(
            scene["CameraShots"],
            scene_time_range=time_range_in_M,
            scene_time_from_real=scene_time_from_real,
            normalized_time_from_scene=normalized_time_from_scene,
        )

    # Report time
    if animation is not None:
        report_time_cue = pv.PythonAnimationCue()
        report_time_cue.Script = """
def start_cue(self): pass

def tick(self):
    import paraview.simple as pv
    import logging
    logger = logging.getLogger('Animation')
    scene_time = pv.GetActiveView().ViewTime
    logger.info(f"Scene time: {scene_time}")

def end_cue(self): pass
"""
        animation.Cues.append(report_time_cue)

    if show_preview and animation is not None:
        animation.PlayMode = "Real Time"
        animation.Duration = 10
        animation.Play()
        animation.PlayMode = "Sequence"

    if no_render:
        logger.info(
            "No rendering requested. Total time:"
            f" {time.time() - render_start_time:.2f}s"
        )
        return

    if frames_dir is None:
        raise RuntimeError("Trying to render but `frames_dir` is not set.")
    if os.path.exists(frames_dir):
        logger.warning(
            f"Output directory '{frames_dir}' exists, files may be overwritten."
        )
    else:
        os.makedirs(frames_dir)

    if animation is None:
        pv.Render()
        pv.SaveScreenshot(os.path.join(frames_dir, "frame.png"))
        if interact:
            #pv.ExportView(os.path.join(frames_dir, "frame.x3d"), ExportColorLegends=1)
            #pv.ExportView(os.path.join(frames_dir, "frame.gltf"), InlineData=1
            #pv.ExportView(os.path.join(frames_dir, "frame.vtkjs"))
            #pv.ExportView(os.path.join(frames_dir, "frame.vtp"))
            pv.Interact();
            cam = pv.GetActiveCamera()
            print("Camera settings after interact")
            print("Position:", cam.GetPosition())
            print("Focal point:", cam.GetFocalPoint())
            print("View up:", cam.GetViewUp())
        yield dict(completed=1)
    else:
        # Iterate over frames manually to support filling in missing frames.
        # If `pv.SaveAnimation` would support that, here's how it could be
        # invoked:
        # pv.SaveAnimation(
        #     os.path.join(frames_dir, 'frame.png'),
        #     view,
        #     animation,
        #     FrameWindow=frame_window,
        #     SuffixFormat='.%06d')
        # Note that `FrameWindow` appears to be buggy, so we set up the
        # `animation` according to the `frame_window` above so the frame files
        # are numbered correctly.
        for animation_window_frame_i in range(animation_window_num_frames):
            frame_i = frame_window[0] + animation_window_frame_i
            frame_file = os.path.join(frames_dir, f"frame.{frame_i:06d}.png")
            if render_missing_frames and os.path.exists(frame_file):
                continue
            logger.debug(f"Rendering frame {frame_i}...")
            animation.AnimationTime = (
                animation.StartTime
                + time_per_frame_in_M * animation_window_frame_i
            )
            pv.Render()
            pv.SaveScreenshot(frame_file)
            logger.info(f"Rendered frame {frame_i}.")
            yield dict(advance=1)

    logger.info(
        f"Rendering done. Total time: {time.time() - render_start_time:.2f}s"
    )
