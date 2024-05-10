import os


def create_directory(directory_path):
    os.makedirs(directory_path, exist_ok=True)


def get_file_extension(file_path):
    return os.path.splitext(file_path)[1]


def files_in_directory(directory_path, format=None):
    files = []
    for file in os.listdir(directory_path):
        if format is None or file.endswith(format):
            files.append(os.path.join(directory_path, file))
    return files


def fetch_files(directory_path, format=None):
    if os.path.isdir(directory_path):
        return files_in_directory(directory_path, format)
    return [directory_path]