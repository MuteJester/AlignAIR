import pathlib

def get_filename(path,return_suffix=True):
    path_object = pathlib.Path(path)
    if return_suffix:
        return path_object.stem,path_object.suffix
    else:
        return path_object.stem
