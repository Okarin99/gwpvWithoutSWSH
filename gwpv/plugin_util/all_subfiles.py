import h5py


def all_subfiles(h5_location):
    if isinstance(h5_location, h5py.Group):
        for subfile in h5_location.values():
            for subfile_name in all_subfiles(subfile):
                yield subfile_name
    else:
        yield h5_location.name
