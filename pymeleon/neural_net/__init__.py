import pathlib

pretrained_models_path = pathlib.Path.home() / ".pymeleon/pretrained_models"
pretrained_models_path.mkdir(parents=True, exist_ok=True)
