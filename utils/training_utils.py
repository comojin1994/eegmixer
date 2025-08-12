from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from utils.setup_utils import logger
from const import uri
import mlflow


def save_artifacts(training_file: str, slack: bool):
    logger("#manager", mlflow.get_artifact_uri(), slack)
    mlflow.log_artifacts("./configs", artifact_path="code/configs")
    mlflow.log_artifacts("./datasets", artifact_path="code/datasets")
    mlflow.log_artifacts("./models", artifact_path="code/models")
    mlflow.log_artifact(training_file, artifact_path="code")


def initialize_logger_and_callbacks(args):
    """Initialize logger and callbacks for training."""
    tags = {key: str(value) for key, value in args.items()}
    mlflow_logger = MLFlowLogger(
        experiment_name=args.experiment_name,
        run_name=args.msg,
        tracking_uri=uri[args.phase],
        log_model=True,
        tags=tags,
        artifact_location="./artifacts",
    )

    return mlflow_logger
