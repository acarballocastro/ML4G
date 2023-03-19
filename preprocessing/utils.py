import os

def get_paths(path):
    """
    The function will return the whole path for all the files in the specified directory.

    :param path: This is the path of the directory
    """
    files = os.listdir(path)
    return [os.path.join(path, f) for f in files]

