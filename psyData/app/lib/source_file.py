import os


SOURCE_FILE_COLUMN = 'source_file'


def addSourceFileColumn(
        data_frames,
        file_paths,
        preferred_name=SOURCE_FILE_COLUMN,
        existing_source_columns=(),
):
    """Add a source filename column only when imported data do not already contain one."""
    frames = list(data_frames)
    paths = list(file_paths)
    if len(frames) != len(paths):
        raise ValueError('Each imported data frame must have one matching source file path.')

    recognized_columns = (preferred_name,) + tuple(existing_source_columns)
    for frame, file_path in zip(frames, paths):
        if not any(column in frame.columns for column in recognized_columns):
            frame[preferred_name] = os.path.basename(os.fsdecode(file_path))
    return frames, preferred_name
