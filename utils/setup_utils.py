from configparser import ConfigParser
from easydict import EasyDict
from datetime import datetime
import requests
import os
import yaml
import torch
import logging
import torch.backends.cudnn as cudnn
import lightning as L
import argparse


def setup_args():
    """Initialize and parse configuration arguments."""
    args = load_configs()
    args = initialize_configs(args)

    aargs = parse_arguments()

    if aargs.seed is not None:
        args.SEED = aargs.seed
        args.msg += f"-{args.SEED}"

    configure_environment(args)
    return args, aargs


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_subject", type=int, default=None)
    parser.add_argument("--gpu_num", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_configs():
    """Load configuration files."""
    config_path = "configs/"

    # Load base config
    args = load_yaml(os.path.join(config_path, "base.yaml"))

    # Load experiment-specific config
    experiment_config = load_yaml(os.path.join(config_path, "config.yaml"))
    args.update(experiment_config)

    # Load dataset config
    dataset_config = load_yaml(
        os.path.join(config_path, f"datasets/{args.dataset}.yaml")
    )
    args.update(dataset_config)

    # Load model config
    model_config = load_yaml(os.path.join(config_path, f"models/{args.model}.yaml"))
    args.update(model_config[args.dataset])

    return args


def load_yaml(file_path):
    """Helper function to load a YAML file."""
    with open(file_path, "r") as file:
        return EasyDict(yaml.safe_load(file))


def initialize_configs(args):
    """Initialize dynamic configurations."""
    args.current_time = datetime.now().strftime("%Y%m%d")
    args.device = get_device(args.GPU_NUM)
    return args


def get_device(gpu_num: str) -> torch.device:
    """Determine and return the appropriate device."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 1 or device_count == 1:
            device = torch.device(f"cuda:{gpu_num}" if device_count == 1 else "cuda")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"{device} is checked")
    return device


def configure_environment(args):
    """Set environment-specific configurations."""
    cudnn.benchmark = False
    cudnn.deterministic = True
    L.seed_everything(args.SEED)


def logger(channel: str, text: str, slack: bool):
    """Log or send a message to Slack."""
    if slack:
        send_slack_message(channel, text)
    else:
        log_level = logging.error if "error" in channel else logging.info
        log_level(text)


def send_slack_message(channel: str, text: str):
    """Send a message to a Slack channel."""
    token = get_slack_token()
    url = "https://slack.com/api/chat.postMessage"
    headers = {"Authorization": f"Bearer {token}"}
    data = {"channel": channel, "text": text}

    try:
        response = requests.post(url, headers=headers, data=data)
        print(response)
    except requests.RequestException as e:
        print(e)


def get_slack_token():
    """Retrieve Slack token from configuration file."""
    config = ConfigParser()
    config.read("key.ini")
    return config["Key"]["slack_token"].strip('"')


def get_huggingface_token():
    """Retrieve Huggingface token from configuration file."""
    config = ConfigParser()
    config.read("key.ini")
    return config["Key"]["huggingface_token"].strip('"')


def parse_device(config_gpu_num, argparser_gpu_num):
    """Parse and determine the GPU devices to use."""
    return (
        list(map(int, config_gpu_num.split(",")))
        if argparser_gpu_num is None
        else [int(argparser_gpu_num)]
    )
