import os
import traceback
from datetime import datetime

import mlflow
from lightning.pytorch import Trainer

from datasets.maker import load_data
from models.maker import ModelMaker
from utils.setup_utils import logger, parse_device, setup_args
from utils.training_utils import initialize_logger_and_callbacks, save_artifacts


def train_and_evaluate(
    trainer, model, train_dataloader, test_dataloader, args, target_subject
):
    """Train and evaluate the model."""
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader
    )

    result = trainer.test(model, dataloaders=test_dataloader)
    acc = result[0]["eval_acc"]

    msg = f"Result, Sub: {target_subject + 1}, Acc.: {acc:.4f}\nMessage: {args.msg}"
    logger("#manager", msg, args.slack)


def main():
    args, aargs = setup_args()
    logger(
        "#manager",
        f"Training Start, Time: {datetime.now():%Y/%m/%d-%H:%M:%S}\nMessage: {args.msg}",
        args.slack,
    )

    for target_subject in range(args.num_subjects):
        if aargs.target_subject is not None and target_subject != aargs.target_subject:
            continue

        # Load data and model
        args.target_subject = target_subject
        train_dataloader, test_dataloader = load_data(target_subject, args)

        model = ModelMaker(args.model, args.litmodel).load_model(args)

        # Initialize logger and callbacks
        mlflow_logger = initialize_logger_and_callbacks(args)

        # Initialize trainer
        devices = parse_device(args.GPU_NUM, aargs.gpu_num)
        trainer = Trainer(
            max_epochs=args.EPOCHS,
            accelerator="gpu",
            strategy="ddp",
            devices=devices,
            logger=mlflow_logger,
        )

        # Start MLflow run and save artifacts
        with mlflow.start_run(run_id=mlflow_logger.run_id) as run:
            save_artifacts(os.path.basename(__file__), args.slack)
            mlflow.pytorch.log_model(model, args.model)

        # Train and evaluate
        train_and_evaluate(
            trainer, model, train_dataloader, test_dataloader, args, target_subject
        )

    logger(
        "#manager",
        f"Training End, Time: {datetime.now():%Y/%m/%d-%H:%M:%S}\nMessage: {args.msg}",
        args.slack,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_msg = f"Error:\n\n{e}\nTraceback:\n\n{traceback.format_exc()}"
        logger("#error", error_msg, slack=True)
