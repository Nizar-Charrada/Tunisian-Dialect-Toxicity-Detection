from collections import Counter
import os
import json
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from dataloader.dataset import SmartBatchingDataset

from models.model import TeacherModel, StudentModel
from models.optimizer import WarmupAnnealing
from data.arabic.stop_words import arabic_stop_words
from data.arabizi.stop_words import arabizi_stop_words
import torch.nn as nn
import torch.optim as optim
import time
import math
from transformers import RobertaConfig, AutoTokenizer, BertTokenizer
import matplotlib as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import gc
from transformers import logging
from utils import ParamsNamespace, load_config, fix_all_seeds, AverageMeter, remove_stop_words

logging.set_verbosity_warning()
logging.set_verbosity_error()

# -----------------------------#
config_path = "toxic_detection/config/arabic_config.yaml"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------#


class Trainer:
    """Class to handle training of the model

    Attributes:

        model: Model
        The model to be trained
        tokenizer: Tokenizer
        The tokenizer used to convert text into tokens
        optimizer: Optimizer
        The optimizer used to update the model parameters
        scheduler: Scheduler
        The scheduler used to update the learning rate
    """

    def __init__(
        self,
        student_model,
        tokenizer,
        optimizer,
        scheduler,
        teacher_model=None,
    ):
        self.student_model = student_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.teacher_model = teacher_model

    def train(self, params, train_dataloader, loss_fn, epoch, result_dict):
        count = 0
        losses = AverageMeter()

        # Zero out the gradients and set model to train mode

        self.student_model.zero_grad()
        self.student_model.train()

        # Fix random seeds for reproducibility
        fix_all_seeds(params.seed)

        # Iterate over training data batches
        for batch_idx, batch_data in enumerate(train_dataloader):
            input_ids, attention_mask, labels = batch_data

            # Move data to GPU
            input_ids, attention_mask, labels = (
                input_ids.cuda(),
                attention_mask.cuda(),
                labels.cuda(),
            )

            if self.teacher_model is not None:
                self.teacher_model.eval()
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(input_ids, attention_mask)

            # Forward pass through the model
            outputs = self.student_model(
                input_ids,
                attention_mask,
            )

            if params.knowledge_distillation.enabled:
                # Compute the loss between the model predictions and the teacher predictions using knowledge distillation
                loss = loss_fn_kd(
                    outputs,
                    labels.unsqueeze(-1).float(),
                    teacher_outputs,
                    params.knowledge_distillation.alpha,
                )
            else:
                # Compute the loss between the model predictions and the true labels
                loss = loss_fn(outputs, labels.unsqueeze(-1).float())

            # Backpropagate the loss and update model parameters
            loss.backward()

            # Keep track of the number of training examples seen so far and the average loss
            count += labels.size(0)
            losses.update(loss.item(), input_ids.size(0))

            # Clip gradients to prevent them from getting too large
            torch.nn.utils.clip_grad_norm_(
                self.student_model.parameters(), params.trainer.gradient_clip_val
            )

            # Perform one optimization step and update the learning rate scheduler
            self.optimizer.step()
            if hasattr(params, "decay_name"):
                self.scheduler.step()

            # Zero out the gradients again for the next batch
            self.optimizer.zero_grad()

            # Log the training progress at certain intervals
            if (batch_idx % params.trainer.row_log_interval == 0) or (
                batch_idx + 1
            ) == len(train_dataloader):
                _s = str(len(str(len(train_dataloader.sampler))))
                lr = self.optimizer.param_groups[0]["lr"]
                ret = [
                    ("Epoch: {:0>2} [{: >" + _s + "}/{} ({: >3.0f}%)]").format(
                        epoch,
                        count,
                        len(train_dataloader.sampler),
                        100 * count / len(train_dataloader.sampler),
                    ),
                    "Train Loss: {: >4.5f}".format(losses.avg),
                    "Learning Rate: {: 6.1e}".format(lr),
                ]
                print(", ".join(ret))

        # Append the final average loss for this epoch to the result dictionary
        result_dict["train_loss"].append(losses.avg)

        return result_dict


class Evaluator:
    """Class to handle validation and testing of the model

    Attributes:

        model: Model
        The model to be evaluated
        tokenizer: Tokenizer
        The tokenizer used to convert text into tokens
    """

    def __init__(self, model):
        self.model = model

    def save(self, result, output_dir):
        # Saves the results to a JSON file
        with open(f"{output_dir}/result_dict.json", "w") as f:
            f.write(json.dumps(result, sort_keys=True, indent=4, ensure_ascii=False))

    def evaluate(self, valid_dataloader, loss_fn, epoch, result_dict):
        # Initializes a dictionary to store the predictions and labels
        matrix = {"preds": [], "labels": []}
        # Initializes an object to store the losses
        losses = AverageMeter()
        # Initializes a variable to store the accuracy
        train_acc = 0

        # Loops over the validation dataloader
        for batch_idx, batch_data in enumerate(valid_dataloader):
            # Sets the model in evaluation mode
            self.model = self.model.eval()
            input_ids, attention_mask, labels = batch_data
            input_ids, attention_mask, labels = (
                input_ids.cuda(),
                attention_mask.cuda(),
                labels.cuda(),
            )
            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    attention_mask,
                )
                loss = loss_fn(outputs, labels.unsqueeze(-1).float())
                losses.update(loss.item(), input_ids.size(0))

                # Calculates the predictions and updates the accuracy
                logit = nn.Sigmoid()(outputs.detach().cpu())

                train_acc += torch.sum(
                    (logit >= 0.5) * 1 == labels.unsqueeze(-1).float().detach().cpu()
                )

                # Stores the predictions and labels in the dictionary
                matrix["preds"].extend((logit.squeeze(1).data.cpu().tolist()))
                matrix["labels"].extend((labels.data.cpu().tolist()))

        print("----Validation Results Summary----")
        # Prints the validation loss
        print("Epoch: [{}] Valid Loss: {: >4.5f}".format(epoch, losses.avg))
        # Prints the AUC score
        print("AUC:", roc_auc_score(matrix["labels"], matrix["preds"]))
        # Calculates and prints the accuracy
        final_train_acc = train_acc.item() / (len(valid_dataloader) * 64)
        print(f"Accuracy : {final_train_acc}")
        # Stores the validation loss in the result dictionary and returns it along with the predictions and labels
        result_dict["val_loss"].append(losses.avg)
        return result_dict, matrix


def make_optimizer(params, model):
    """
    Create an optimizer based on the specified hyperparameters.
    """
    lr = (
        params.knowledge_distillation.student_model.lr
        if params.knowledge_distillation.enabled
        else params.optim.lr
    )

    weight_decay = (
        params.knowledge_distillation.student_model.weight_decay
        if params.knowledge_distillation.enabled
        else params.optim.weight_decay
    )

    if params.optim.name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif params.optim.name == "adamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        # raise an error if an unsupported optimizer is specified
        assert False, f"optimizer_type : {params.optim.name} is not supported"

    return optimizer


def make_scheduler(params, optimizer, num_warmup_steps, num_training_steps):
    """
    Create a learning rate scheduler based on the specified hyperparameters.
    """
    scheduler = WarmupAnnealing(
        optimizer=optimizer,
        max_steps=num_training_steps,
        warmup_ratio=params.optim.sched.warmup_ratio,
        last_epoch=-1,
    )
    return scheduler


def loss_fn_kd(outputs, labels, teacher_outputs, alpha):
    KD_loss = nn.BCEWithLogitsLoss()(outputs, nn.Sigmoid()(teacher_outputs)) * (
        1.0 - alpha
    ) + nn.BCEWithLogitsLoss()(outputs, labels) * (alpha)

    return KD_loss


def make_loss_fn(params):
    """
    Create a loss function based on the specified hyperparameters.
    """
    loss_fn = nn.BCEWithLogitsLoss()
    return loss_fn


def make_model(params, model_config, state_dict=None):
    """
    Create a model based on the specified hyperparameters.
    """
    model = TeacherModel(params, model_config, freeze=False)
    if state_dict is not None:
        model.bert_model.load_state_dict(state_dict)
    return model


def make_tokenizer(params):
    """
    Create a tokenizer based on the specified hyperparameters.
    """
    if params.model.language_model.model_type == "roberta":
        # here we used bert tokenizer for roberta model as stated in the model documentation "ziedsb19"
        tokenizer = BertTokenizer.from_pretrained(params.model.tokenizer.tokenizer_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(params.model.tokenizer.tokenizer_name)
    return tokenizer


def get_model_config(params):
    """
    Get model config based on the specified hyperparameters.
    """
    if params.model.language_model.model_type == "roberta":
        # here we used bert config for roberta model as stated in the model documentation "ziedsb19"
        model_config = RobertaConfig.from_pretrained(
            params.model.language_model.model_name_or_path
        )
        return model_config
    else:
        # I know that this is not the best way to do it but I didn't find a better way to do it
        # it is horrible but it works for now
        # We load the checkpoint, get the model config and the state dict and then we delete the layers that we don't need
        checkpoint = torch.load(params.trainer.resume_from_checkpoint)
        model_config = checkpoint["hyper_parameters"]["language_model"]["config"]
        state_dict = checkpoint["state_dict"]
        to_drop = [
            "mlm_classifier.dense.weight",
            "mlm_classifier.dense.bias",
            "mlm_classifier.norm.weight",
            "mlm_classifier.norm.bias",
            "mlm_classifier.mlp.layer0.weight",
            "mlm_classifier.mlp.layer0.bias",
            "nsp_classifier.mlp.layer0.weight",
            "nsp_classifier.mlp.layer0.bias",
            "nsp_classifier.mlp.layer2.weight",
            "nsp_classifier.mlp.layer2.bias",
        ]
        for col in to_drop:
            del state_dict[col]
        all_keys = list(state_dict.keys())
        for k in all_keys:
            state_dict[".".join(k.split(".")[1:])] = state_dict.pop(k)
        return model_config, state_dict


def make_loader(params, tokenizer):
    """
    Create data loaders for the training and validation sets.
    """
    train_set = pd.read_csv(params.train_ds.file_path)
    valid_set = pd.read_csv(params.validation_ds.file_path)
    # We add a column to the dataframes to distinguish between the training and validation sets
    train_set["is_train"] = 1
    valid_set["is_train"] = 0
    # We concatenate the two dataframes
    data = pd.concat([train_set, valid_set], axis=0)

    # i extracted top 300 frequent words from the dataset because they are most likely to be stop words
    # i manually added some stop words that are not included in the top 300 frequent words

    # we remove the stop words from the text
    if not params.knowledge_distillation.enabled:
        if params.model.language_model.model_type == "roberta":
            data["text"] = data["text"].apply(remove_stop_words, stop_words=arabizi_stop_words)
        else:
            data["text"] = data["text"].apply(remove_stop_words, stop_words=arabic_stop_words)

    train_set = data[data["is_train"] == 1]
    valid_set = data[data["is_train"] == 0]

    train_dataset = SmartBatchingDataset(train_set, tokenizer)
    valid_dataset = SmartBatchingDataset(valid_set, tokenizer)

    # print information about the datasets
    print(
        f"Num examples Train= {len(train_dataset)}, Num examples Valid={len(valid_dataset)}"
    )
    print(
        f"Validation num positive text : {len(valid_set[valid_set['label'] == 0])}, Validation num toxic text : {len(valid_set[valid_set['label'] == 1])} "
    )

    train_dataloader = train_dataset.get_dataloader(
        batch_size=params.train_ds.batch_size,
        max_len=params.model.tokenizer.max_length,
        pad_id=tokenizer.pad_token_id,
    )
    valid_dataloader = valid_dataset.get_dataloader(
        batch_size=params.validation_ds.batch_size,
        max_len=params.model.tokenizer.max_length,
        pad_id=tokenizer.pad_token_id,
    )

    # calculate the number of tokens saved by using smart batching
    padded_lengths = []
    for batch_idx, (input_ids, attention_mask, targets) in enumerate(train_dataloader):
        for s in input_ids:
            padded_lengths.append(len(s))

    smart_token_count = np.sum(padded_lengths)
    fixed_token_count = len(train_set["text"]) * params.model.tokenizer.max_length

    prcnt_reduced = (fixed_token_count - smart_token_count) / float(fixed_token_count)

    print("Total tokens:")
    print("  Fixed Padding: {:,}".format(fixed_token_count))
    print(
        "  Smart Batching: {:,}  ({:.2%} less)".format(smart_token_count, prcnt_reduced)
    )

    return train_dataloader, valid_dataloader


def init_training(params):
    """
    Initialize the training process.
    """
    # Set the random seed for reproducible experiments.
    fix_all_seeds(params.seed)

    # Create the output directory if it does not already exist.
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)

    # initialize the tokenizer
    print("Loading tokenizer {}...".format(params.model.tokenizer.tokenizer_name))
    tokenizer = make_tokenizer(params)

    # Create the data loaders for the training and validation sets.
    print("Loading datasets...")
    train_dataloader, valid_dataloader = make_loader(params, tokenizer)

    # Initialize the model architecture.
    print("Loading model...")
    if params.model.language_model.model_type == "roberta":
        model_config = get_model_config(params)
        model = make_model(params, model_config)

    elif params.model.language_model.model_type == "bert":
        model_config, state_dict = get_model_config(params)
        model = make_model(params, model_config, state_dict=state_dict)

    else:
        raise ValueError("Invalid `model_type`.")

    # Move the model to a GPU if available, otherwise raise an error.
    if torch.cuda.device_count() >= 1:
        print(
            "Model pushed to {} GPU(s), type {}.".format(
                torch.cuda.device_count(), torch.cuda.get_device_name(0)
            )
        )
        model = model.cuda()
    else:
        raise ValueError("CPU training is not supported")

    if params.knowledge_distillation.enabled:
        print("Knowledge distillation is enabled")
        print("Loading trained teacher model...")
        assert (
            params.knowledge_distillation.teacher_model.checkpoint_path is not None
        ), "Please provide a checkpoint path for the teacher model"
        assert os.path.exists(
            os.path.join(
                params.knowledge_distillation.teacher_model.checkpoint_path,
                f"{params.model.language_model.model_type}-checkpoint",
            )
        ), "The checkpoint path for the teacher model does not exist"

        model.load_state_dict(
            torch.load(
                os.path.join(
                    params.knowledge_distillation.teacher_model.checkpoint_path,
                    f"{params.model.language_model.model_type}-checkpoint/best.bin",
                )
            )
        )
        for p in model.parameters():
            p.requires_grad = False

        model.eval()
        print("Loading student model...")
        student_model = StudentModel(
            tokenizer.vocab_size,
            params.knowledge_distillation.student_model.embedding_size,
            params.knowledge_distillation.student_model.hidden_size,
            params.knowledge_distillation.student_model.kernel_size,
            add_conv_layer=params.knowledge_distillation.student_model.add_conv_layer,
        )
        student_model = student_model.cuda()

    # Create the optimizer for the model.
    print("Creating optimizer...")
    if params.knowledge_distillation.enabled:
        optimizer = make_optimizer(params, student_model)
    else:
        optimizer = make_optimizer(params, model)

    # Create the learning rate scheduler.

    scheduler = None
    if hasattr(params, "optim.sched"):
        print("Creating scheduler...")
        num_training_steps = len(train_dataloader) * params.trainer.max_epochs
        num_warmup_steps = params.optim.sched.warmup_ratio * num_training_steps
        print(
            f"Total Training Steps: {num_training_steps}, Total Warmup Steps: {math.floor(len(train_dataloader) * params.optim.sched.warmup_ratio)}"
        )
        scheduler = make_scheduler(
            params, optimizer, num_warmup_steps, num_training_steps
        )

    # Initialize a dictionary to store the results of each epoch of training.
    result_dict = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "best_val_loss": np.inf,
        "config": {
            "epochs": params.trainer.max_epochs,
            "train_batch_size": params.train_ds.batch_size,
            "optimizer_type": params.optim.name,
            "learning_rate": params.optim.lr,
            "weight_decay": params.optim.weight_decay,
            "warmup_ratio": params.optim.sched.warmup_ratio
            if hasattr(params, "optim.sched")
            else 0,
        },
    }

    # Return the necessary components for training.
    if params.knowledge_distillation.enabled:
        return (
            model,
            student_model,
            optimizer,
            tokenizer,
            scheduler,
            train_dataloader,
            valid_dataloader,
            result_dict,
        )
    else:
        return (
            model,
            optimizer,
            tokenizer,
            scheduler,
            train_dataloader,
            valid_dataloader,
            result_dict,
        )


""" Training Loop """
if __name__ == "__main__":
    # load the configuration file
    yaml_config = load_config(config_path)
    params = ParamsNamespace(yaml_config)

    # initialize the training process
    if params.knowledge_distillation.enabled:
        (
            teacher_model,
            student_model,
            optimizer,
            tokenizer,
            scheduler,
            train_dataloader,
            valid_dataloader,
            result_dict,
        ) = init_training(params)
    else:
        (
            model,
            optimizer,
            tokenizer,
            scheduler,
            train_dataloader,
            valid_dataloader,
            result_dict,
        ) = init_training(params)

    # initialize the loss function
    loss_fn = make_loss_fn(params)

    # initialize the trainer
    if params.knowledge_distillation.enabled:
        trainer = Trainer(student_model, tokenizer, optimizer, scheduler, teacher_model)
        evaluator = Evaluator(student_model)
    else:
        trainer = Trainer(model, tokenizer, optimizer, scheduler)
        evaluator = Evaluator(model)

    train_time_list = []
    valid_time_list = []

    for epoch in range(params.trainer.max_epochs):
        result_dict["epoch"].append(epoch)

        # Train the model
        # Synchronize the device before starting training
        torch.cuda.synchronize()
        # Record the current time
        tic1 = time.time()
        # Call the trainer object's train method with the current epoch's parameters
        result_dict = trainer.train(
            params, train_dataloader, loss_fn, epoch, result_dict
        )
        # Synchronize the device again after training
        torch.cuda.synchronize()
        # Record the time taken for training
        tic2 = time.time()
        # Append the training time to the list
        train_time_list.append(tic2 - tic1)

        # Evaluate the model on the validation set
        # Synchronize the device before starting validation
        torch.cuda.synchronize()
        # Record the current time
        tic3 = time.time()
        # Call the evaluator object's evaluate method with the validation data
        result_dict, matrix = evaluator.evaluate(
            valid_dataloader, loss_fn, epoch, result_dict
        )
        # Synchronize the device again after validation
        torch.cuda.synchronize()
        # Record the time taken for validation
        tic4 = time.time()
        # Append the validation time to the list
        valid_time_list.append(tic4 - tic3)

        # Set the output directory for saving checkpoints
        if params.knowledge_distillation.enabled:
            output_dir = os.path.join(
                params.output_dir,
                f"{params.model.language_model.model_type}-knowledge_distillation-checkpoint",
            )
        else:
            output_dir = os.path.join(
                params.output_dir,
                f"{params.model.language_model.model_type}-checkpoint",
            )
        # Check if the current epoch's validation loss is better than the best so far
        if result_dict["val_loss"][-1] < result_dict["best_val_loss"]:
            # If so, update the best validation loss in the result dictionary
            print(
                "{} Epoch, Best epoch was updated! Valid Loss: {: >4.5f}".format(
                    epoch, result_dict["val_loss"][-1]
                )
            )
            result_dict["best_val_loss"] = result_dict["val_loss"][-1]
            # Save the confusion matrix with labels and predictions for this epoch
            best_matrix = matrix
            # Create the output directory if it doesn't already exist
            os.makedirs(output_dir, exist_ok=True)
            # If specified, show the classification report and confusion matrix
            if params.show_report:
                cf_matrix = confusion_matrix(
                    best_matrix["labels"], best_matrix["preds"], normalize="true"
                )
                plt.figure(figsize=(12, 7))
                sns.heatmap(cf_matrix, annot=True)
                plt.savefig(f"{output_dir}/output.png")

                print(
                    classification_report(best_matrix["labels"], best_matrix["preds"])
                )
            # If specified, save the model checkpoint
            if params.save_model:
                if params.knowledge_distillation.enabled:
                    torch.save(
                        student_model.state_dict(),
                        f"{output_dir}/best.bin",
                    )
                else:
                    torch.save(
                        model.state_dict(),
                        f"{output_dir}/best.bin",
                    )
                print(f"Saving model checkpoint to {output_dir}.")
            # If specified, save the optimizer and scheduler states
            # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            # print(f"Saving optimizer and scheduler states to {output_dir}.")
        print()

    evaluator.save(result_dict, output_dir)

    print()
    print(
        f"Total Training Time: {np.sum(train_time_list)}secs, Average Training Time per Epoch: {np.mean(train_time_list)}secs."
    )
    print(
        f"Total Validation Time: {np.sum(valid_time_list)}secs, Average Validation Time per Epoch: {np.mean(valid_time_list)}secs."
    )

    torch.cuda.empty_cache()
    gc.collect()
