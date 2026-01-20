import os


def get_model_dir():
    model_dir = os.environ.get("TELETRON_MODEL_DIR") or os.environ.get("MODEL_DIR")
    if model_dir:
        return model_dir
    raise RuntimeError("model directory is not configured")


def get_huggingface_model_path(model_name_or_path):
    return model_name_or_path


def get_model_path(model_name_or_path):
    model_name_or_path = os.path.expandvars(model_name_or_path)
    if model_name_or_path is None or os.path.exists(model_name_or_path):
        return model_name_or_path
    if os.path.isabs(model_name_or_path):
        raise ValueError(f"{model_name_or_path} does not exist")
    model_dir = get_model_dir()
    model_path = os.path.join(model_dir, model_name_or_path)
    if os.path.exists(model_path):
        return model_path
    return get_huggingface_model_path(model_name_or_path)
