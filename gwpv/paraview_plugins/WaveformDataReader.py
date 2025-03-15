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
    
    @smproperty.dataarrayselection(name="Polarizations")
    def GetPolarizations(self):
        return self.polarizations_selection
    
    @smproperty.doublevector(
        name="TimestepValues",
        information_only="1",
        si_class="vtkSITimeStepsProperty",
    )
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

                output.SetDimensions(subfile.attrs["dim_x"], subfile.attrs["dim_y"], subfile.attrs["dim_z"])
                output.SetOrigin(subfile.attrs["origin_x"], subfile.attrs["origin_y"], subfile.attrs["origin_y"])
                output.SetSpacing(subfile.attrs["spacing_x"], subfile.attrs["spacing_y"], subfile.attrs["spacing_z"])

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
