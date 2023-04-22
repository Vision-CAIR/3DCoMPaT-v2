"""
Segformer finetuning on 2D segmentation.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from compat2D import GCRLoader
from custom_metrics import compute_overall_iou, compute_overall_precision
from tqdm import tqdm
from transformers import SegformerConfig, SegformerForSemanticSegmentation


def parse_args(argv):
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(
        description="Training the Baseline Models on 2D Part and Material Segmentation"
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=60,
        help="Number of epochs to be used for training",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="fine",
        help="Semantic level to be used for training",
    )
    parser.add_argument(
        "--task", type=str, default="part", help="Type of task to be used for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs to be used for training",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="nvidia/segformer-b0-finetuned-ade-512-512",
        help="Name of the model file",
    )
    parser.add_argument(
        "--root_url",
        type=str,
        required=True,
        help="Root URL of the 2D data",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="segformer",
        help="Name of the model file",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size to be used for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate to be used for training",
    )
    parser.add_argument(
        "--n_comp",
        type=int,
        default=1,
        help="Number of components to be used for training",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="Optimizer to be used for training",
    )
    args = parser.parse_args(argv)

    # Printing input arguments
    print("Input arguments:")
    print(args)

    return args


def load_data(args):
    """
    Instantiate the data loaders.
    """
    gcr_train_loader = GCRLoader(
        root_url=args.root_url,
        split="train",
        semantic_level=args.data_type,
        n_compositions=args.n_comp,
    ).make_loader(batch_size=args.batch_size, num_workers=2)

    gcr_valid_loader = GCRLoader(
        root_url=args.root_url,
        split="valid",
        semantic_level=args.data_type,
        n_compositions=args.n_comp,
    ).make_loader(batch_size=32, num_workers=2)

    return gcr_train_loader, gcr_valid_loader


def training_and_evaluating_step(gcr_train_loader, gcr_valid_loader, args):
    """
    Train and evaluate the model.
    """
    PART_CLASSES = 276
    MAT_CLASSES = 14

    if args.data_type == "coarse":
        PART_CLASSES = 44

    num_labels = PART_CLASSES if args.task == "part" else MAT_CLASSES

    config = SegformerConfig.from_pretrained(args.model_name, num_labels=num_labels)
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_name, config=config, ignore_mismatched_sizes=True
    )

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs.")
        model = nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the optimizer and loss function
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005
        )
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=0.01, max_lr=0.1
        )

    # Define the evaluation metric
    if not os.path.exists("./logs"):
        os.mkdir("./logs")

    if not os.path.exists("./logs/" + str(args.model)):
        os.mkdir("./logs/" + str(args.model))

    checkpoint_name = (
        "./logs/"
        + str(args.model)
        + "/"
        + str(args.data_type)
        + "_"
        + str(args.task)
        + "_best_model.pth"
    )
    start_time = time.time()

    # Define the training loop
    best_MIOU = 0
    best_mAP = 0
    for epoch in range(args.num_epochs):
        # Train the model
        model.train()
        train_losses = []
        pbar_train = tqdm(gcr_train_loader, desc=f"Epoch {epoch+1} Training")
        for images, _, y_parts, y_mats in pbar_train:
            y = y_parts if args.task == "part" else y_mats
            optimizer.zero_grad()
            loss = model(images.to(device), y.to(device))[0]
            reduced_loss = loss.mean()
            reduced_loss.backward()
            optimizer.step()
            train_losses.append(reduced_loss.item())
            pbar_train.set_description(
                f"Epoch {epoch+1} Training, Train loss: {np.mean(train_losses):.4f}"
            )
        scheduler.step()

        # Evaluate the model
        model.eval()

        # Torch array of predictions
        general_miou = []
        precisions = []
        for images, _, y_parts, y_mats in tqdm(
            gcr_valid_loader, desc=f"Epoch {epoch+1} Validation"
        ):
            y = y_parts if args.task == "part" else y_mats
            with torch.no_grad():
                outputs = model(images.to(device))["logits"]
                # Upsample the logits
                upsampled_logits = nn.functional.interpolate(
                    outputs, size=y.shape[-2:], mode="bicubic", align_corners=False
                )
                predicted = (
                    torch.softmax(upsampled_logits, dim=1).argmax(dim=1).cpu().numpy()
                )
                y_truth = y.cpu().numpy()
                miou = compute_overall_iou(predicted, y_truth, num_labels)
                prec = compute_overall_precision(predicted, y_truth)
                general_miou = general_miou + miou
                precisions = precisions + prec

        # Compute the metrics: mean IOU and mean precision
        mIOU = np.nanmean(general_miou)
        mPrecision = np.nanmean(precisions)
        print(
            f"Epoch {epoch}:",
            f"Train loss: {np.mean(train_losses):.4f}",
            f"mIOU: {mIOU:.6f}",
            f"mPrecision: {mPrecision:.6f}",
        )

        if mIOU > best_MIOU:
            best_MIOU = mIOU
            best_mAP = mPrecision
            print("Saving model...")
            torch.save(model.state_dict(), checkpoint_name)

    # Evaluate the best model
    for images, _, y_parts, y_mats in tqdm(
        gcr_valid_loader, desc=f"Epoch {epoch+1} Validation"
    ):
        y = y_parts if args.task == "part" else y_mats
        with torch.no_grad():
            outputs = model(images.to(device))["logits"]
            # Upsample the logits
            upsampled_logits = nn.functional.interpolate(
                outputs, size=y.shape[-2:], mode="bicubic", align_corners=False
            )
            predicted = (
                torch.softmax(upsampled_logits, dim=1).argmax(dim=1).cpu().numpy()
            )

    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")

    print("Best mIOU: ", best_MIOU)
    print("Best mAP: ", best_mAP)


def main(argv=None):
    args = parse_args(argv)
    gcr_train_loader, gcr_valid_loader = load_data(args)
    training_and_evaluating_step(gcr_train_loader, gcr_valid_loader, args)


if __name__ == "__main__":
    main()
