import os

from omegaconf import OmegaConf, open_dict


def initialize_config(config):
    try:
        with open_dict(config):
            config.slurm_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", None)
            config.slurm_job_id = os.environ.get("SLURM_JOB_ID", None)
    except KeyError:
        pass


def conditional_resolver(condition, if_true: str, if_false: str):
    if condition:
        return if_true
    else:
        return if_false


def shortener(input_string: str | None = None, length=3, show_config_stack=False):
    if input_string is None:
        return "job_type"
    output_parts = []

    for part in input_string.split(","):
        key, value = part.split("=")
        modified_key = ""

        key_parts = key.split(".")
        if not show_config_stack:
            key_parts = key_parts[-1:]
        for key_part in key_parts:
            for word in key_part.split("_"):
                modified_key += word[:length] + "_"
            modified_key = modified_key[:-1] + "."
        modified_key = modified_key[:-1]

        output_parts.append(f"{modified_key}={value}")

    return ",".join(output_parts)


def load_omega_conf_resolvers():
    OmegaConf.register_new_resolver("sub_dir_shortener", shortener)
    OmegaConf.register_new_resolver("format", lambda inpt, formatter: formatter.format(inpt))
    OmegaConf.register_new_resolver("conditional_resolver", conditional_resolver)
