# Waveform data ParaView reader

import logging
import time
import bisect

import h5py
import numpy as np

from paraview.util.vtkAlgorithm import smdomain, smhint, smproperty, smproxy
from paraview.vtk.util import keys as vtkkeys
from paraview.vtk.util import numpy_support as vtknp
from paraview import util

from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonCore import vtkDataArraySelection
from vtkmodules.vtkCommonDataModel import vtkUniformGrid

import gwpv.plugin_util.data_array_selection as das_util
import gwpv.plugin_util.timesteps as timesteps_util


logger = logging.getLogger(__name__)

def find_index_left(a, x):
    i = bisect.bisect_right(a, x)
    if i >= len(a):
        i = len(a) - 1
    elif i:
        i = i - 1
    return i


@smproxy.reader(
    name="WaveformDataReader",
    label="Waveform Data Reader",
    extensions="h5",
    file_description="HDF5 files",
)
class WaveformDataReader(VTKPythonAlgorithmBase):
    """Read waveform data from an HDF5 file."""

    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self, nInputPorts=0, nOutputPorts=1, outputType="vtkUniformGrid"
        )
        self._filename = None
        self._subfile = None
        self._origin_x = None
        self._origin_y = None
        self._origin_z = None
        self._size_x = None
        self._size_y = None
        self._size_z = None

        self.polarizations_selection = vtkDataArraySelection()
        self.polarizations_selection.AddArray("Plus")
        self.polarizations_selection.AddArray("Cross")
        self.polarizations_selection.AddObserver(
            "ModifiedEvent", das_util.create_modified_callback(self)
        )

    @smproperty.stringvector(name="FileName")
    @smdomain.filelist()
    @smhint.filechooser(extensions="h5", file_description="HDF5 files")
    def SetFileName(self, value):
        self._filename = value
        self.Modified()

    @smproperty.stringvector(name="Subfile")
    def SetSubfile(self, value):
        self._subfile = value
        self.Modified()

    @smproperty.doublevector(name="OriginX", default_values=-9999999999999)
    def SetOriginX(self, value):
        self._origin_x = value
        self.Modified()
    
    @smproperty.doublevector(name="OriginY", default_values=-9999999999999)
    def SetOriginY(self, value):
        self._origin_y = value
        self.Modified()
    
    @smproperty.doublevector(name="OriginZ", default_values=-9999999999999)
    def SetOriginZ(self, value):
        self._origin_z = value
        self.Modified()
    
    @smproperty.doublevector(name="SizeX", default_values=-9999999999999)
    def SetSizeX(self, value):
        self._size_x = value
        self.Modified()
    
    @smproperty.doublevector(name="SizeY", default_values=-9999999999999)
    def SetSizeY(self, value):
        self._size_y = value
        self.Modified()
    
    @smproperty.doublevector(name="SizeZ", default_values=-9999999999999)
    def SetSizeZ(self, value):
        self._size_z = value
        self.Modified()
    
    @smproperty.dataarrayselection(name="Polarizations")
    def GetPolarizations(self):
        return self.polarizations_selection
    
    @smproperty.doublevector(
        name="TimestepValues",
        information_only="1",
        si_class="vtkSITimeStepsProperty",
    )

    def _get_timesteps(self):
        logger.debug("Getting time range from data...")
        if self._filename != "None" and self._subfile != "None":
            with h5py.File(self._filename, "r") as f:
                subfile = f[self._subfile]
                timesteps = subfile["timesteps.dat"]
                # Using a few timesteps within the data range so we can animate through
                # them in the GUI
                return np.array(timesteps[:])
        else:
            return []

    def GetTimestepValues(self):
        return self._get_timesteps().tolist()

    def RequestInformation(self, request, inInfo, outInfo):
        logger.debug("Requesting information...")
        info = outInfo.GetInformationObject(0)
        # Add the modes provided by the data file to the information that
        # propagates down the pipeline. This allows subsequent filters to select
        # a subset of modes to display, for example.
        if self._filename != "None" and self._subfile != "None":
            with h5py.File(self._filename, "r") as f:
                dataset = f[self._subfile]
                grid_extents = [0, dataset.attrs["dim_x"]-1, 0, dataset.attrs["dim_y"]-1, 0, dataset.attrs["dim_z"]-1]
                util.SetOutputWholeExtent(self, grid_extents)
        
        # This needs the time data from the waveform file, so we may have to
        # set the `TIME_RANGE` and `TIME_STEPS` already in the
        # WaveformDataReader.
        timesteps = self._get_timesteps()
        if len(timesteps) != 0:
            timesteps_util.set_timesteps(self, self._get_timesteps(), logger=logger)

        logger.debug(f"Information object: {info}")
        return 1

    def RequestData(self, request, inInfo, outInfo):
        logger.info("Loading waveform data...")
        start_time = time.time()

        if (
            self._filename != "None"
            and self._subfile != "None"
        ):
            info = outInfo.GetInformationObject(0)

            polarizations = [("Plus", 0), ("Cross", 1)]
            t = timesteps_util.get_timestep(self, logger=logger)

            with h5py.File(self._filename, "r") as f:
                subfile = f[self._subfile]

                timesteps = subfile["timesteps.dat"]
                t_index = find_index_left(timesteps, t)

                output = dsa.WrapDataObject(vtkUniformGrid.GetData(outInfo))

                ox = self._origin_x if self._origin_x != -9999999999999 else subfile.attrs["origin_x"]
                oy = self._origin_y if self._origin_y != -9999999999999 else subfile.attrs["origin_y"]
                oz = self._origin_z if self._origin_z != -9999999999999 else subfile.attrs["origin_z"]

                sizex = self._size_x if self._size_x != -9999999999999 else subfile.attrs["size_x"]
                sizey = self._size_y if self._size_y != -9999999999999 else subfile.attrs["size_y"]
                sizez = self._size_z if self._size_z != -9999999999999 else subfile.attrs["size_z"]

                dimx = subfile.attrs["dim_x"]
                dimy = subfile.attrs["dim_y"]
                dimz = subfile.attrs["dim_z"]

                spacingx = sizex / (dimx - 1)
                spacingy = sizey / (dimy - 1)
                spacingz = sizez / (dimz - 1)

                output.SetDimensions(dimx, dimy, dimz)
                output.SetOrigin(ox, oy, oz)
                output.SetSpacing(spacingx, spacingy, spacingz)

                for (polarization, pol_index) in polarizations:
                    if self.polarizations_selection.ArrayIsEnabled(polarization):
                        strain_over_time = subfile["strain"]

                        if t <= timesteps[0]:
                            strain = strain_over_time["1.dat"][:, pol_index]
                        elif t >= timesteps[-1]:
                            strain = strain_over_time[str(len(timesteps)) + ".dat"][:, pol_index]
                        else:
                            t_interp = (t - timesteps[t_index]) / (timesteps[t_index + 1] - timesteps[t_index])
                            strain = (1-t_interp)*strain_over_time[str(t_index+1) + ".dat"][:, pol_index] + t_interp*strain_over_time[str(t_index+2) + ".dat"][:, pol_index]

                        strain_vtk = vtknp.numpy_to_vtk(strain, deep=True)
                        strain_vtk.SetName(polarization + " strain")
                        output.GetPointData().AddArray(strain_vtk)

        logger.info(f"Waveform data loaded in {time.time() - start_time:.3f}s.")

        return 1
