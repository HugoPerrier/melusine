"""
Default : Use default configuration to explore functionalities
Specify a config_dict
Specify a config path
Specify a MELUSINE_CONFIG_DIR environment variable
"""


def from_config():
    # --8<-- [start:from_config]
    from melusine.pipeline import MelusinePipeline

    pipeline = MelusinePipeline.from_config("demo_pipeline")


# --8<-- [end:from_config]


def print_config():
    # --8<-- [start:print_config]
    from melusine import config

    print(config["demo_pipeline"])


# --8<-- [end:print_config]


def modify_conf_with_dict():
    # --8<-- [start:modify_conf_with_dict]
    from melusine import config

    # Get a dict of the existing conf
    new_conf = config.dict()

    # Add/Modify a config key
    new_conf["my_conf_key"] = "my_conf_value"

    # Reset Melusine configurations
    config.reset(new_conf)


# --8<-- [end:modify_conf_with_dict]


def modify_conf_with_path():
    # --8<-- [start:modify_conf_with_path]
    from melusine import config

    # Specify the path to a conf folder
    conf_path = "path/to/conf/folder"

    # Reset Melusine configurations
    config.reset(config_path=conf_path)

    # >> Using config_path : path/to/conf/folder


# --8<-- [end:modify_conf_with_path]


def modify_conf_with_env():
    # --8<-- [start:modify_conf_with_env]
    import os

    from melusine import config

    # Specify the MELUSINE_CONFIG_DIR environment variable
    os.environ["MELUSINE_CONFIG_DIR"] = "path/to/conf/folder"

    # Reset Melusine configurations
    config.reset()

    # >> Using config_path from env variable MELUSINE_CONFIG_DIR
    # >> Using config_path : path/to/conf/folder


# --8<-- [end:modify_conf_with_env]


def export_config():
    # --8<-- [start:export_config]
    from melusine import config

    # Specify the path a folder (created if it doesn't exist)
    my_conf_folder_path = "path/to/conf/folder"

    # Export default configurations to the folder
    files_created = config.export_default_config(path=my_conf_folder_path)


# --8<-- [end:export_config]
