import torch
import numpy as np
import pandas as pd
import time
from utils import ParamsNamespace
import os
from train import get_model_config, make_tokenizer
from models.model import TeacherModel, StudentModel
import yaml


def calculate_nlp_throughput(model, seq_len, batch_sizes, device, attention=False):
    """Calculate the throughput of a model for a given sequence length and batch sizes on a given device.
    The throughput is measured in tokens per second."""

    total_time = 0
    repetitions = 100
    with torch.no_grad():
        for batch_size in batch_sizes:
            dummy_input = torch.randint(
                low=0, high=100, size=(batch_size, seq_len), dtype=torch.long
            ).to(device)
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            for rep in range(repetitions):
                starter.record()
                if attention:
                    _ = model(dummy_input, None)
                else:
                    _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender) / 1000
                total_time += curr_time
    avg_time = total_time / (len(batch_sizes) * repetitions)
    throughput = sum(batch_sizes) / avg_time
    return throughput


def measure_model_performance(model, seq_len, batch_size, device, attention=False):
    """Measure the performance of the model on the given device and return the mean and std of the inference time in ms for the given batch size and sequence length"""
    # Set up the input tensor
    dummy_input = torch.randint(
        low=0, high=100, size=(batch_size, seq_len), dtype=torch.long
    ).to(device)

    # GPU warm-up
    for _ in range(10):
        if attention:
            _ = model(dummy_input, None)
        else:
            _ = model(dummy_input)

    # Measure performance
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    repetitions = 300
    timings = np.zeros((repetitions,))
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            if attention:
                _ = model(dummy_input, None)
            else:
                _ = model(dummy_input)
            ender.record()
            # Wait for GPU sync
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_time = np.mean(timings)
    std_time = np.std(timings)
    return mean_time, std_time


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------------------- -----------------------------------------#
    config_path = "toxic_detection/config/arabic_config.yaml"
    batch_sizes = [32, 64]
    seq_lens = [10, 30, 70]
    # ----------------------------------------- -----------------------------------------#

    assert os.path.isfile(config_path), "No configuration file found at {}".format(
        config_path
    )
    with open(config_path) as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)

    params = ParamsNamespace(yaml_config)

    tokenizer = make_tokenizer(params)
    model_config = get_model_config(params)
    
    print("Loading student model...")
    student_model = StudentModel(
        tokenizer.vocab_size,
        params.knowledge_distillation.student_model.embedding_size,
        params.knowledge_distillation.student_model.hidden_size,
        add_conv_layer=params.knowledge_distillation.student_model.add_conv_layer,
    ).to(device)

    print("Loading teacher model...")
    teacher_model = TeacherModel(params, model_config[0], freeze=False).to(device)

    student_n_params = student_model.get_num_params()
    teacher_n_params = teacher_model.get_num_params()

    param_size_bytes = np.dtype("float32").itemsize

    # Calculate the estimated size of the model in bytes
    teacher_size_bytes = teacher_n_params * param_size_bytes
    student_size_bytes = student_n_params * param_size_bytes

    # Convert the size to megabytes
    teacher_size_mb = teacher_size_bytes / (1024 * 1024)
    student_size_mb = student_size_bytes / (1024 * 1024)

    # Print the estimated size of the models in MB
    print(f"The estimated size of the teacher model is {teacher_size_mb:.2f} MB.")
    print(f"The estimated size of the student model is {student_size_mb:.2f} MB.")
    print(
        f"The student model is {(student_size_mb / teacher_size_mb) * 100:.2f} times smaller than the teacher model."
    )

    print("Loading models' weights...")

    assert (
        params.output_dir is not None
    ), "Please provide a checkpoint path for the student model"
    assert os.path.exists(
        os.path.join(
            params.output_dir,
            f"{params.model.language_model.model_type}-knowledge_distillation-checkpoint",
        )
    ), "The checkpoint path for the student model does not exist"

    student_model.load_state_dict(
        torch.load(
            os.path.join(
                params.output_dir,
                f"{params.model.language_model.model_type}-knowledge_distillation-checkpoint/best.bin",
            )
        )
    )

    assert (
        params.knowledge_distillation.teacher_model.checkpoint_path is not None
    ), "Please provide a checkpoint path for the teacher model"
    assert os.path.exists(
        os.path.join(
            params.knowledge_distillation.teacher_model.checkpoint_path,
            f"{params.model.language_model.model_type}-checkpoint",
        )
    ), "The checkpoint path for the teacher model does not exist"

    teacher_model.load_state_dict(
        torch.load(
            os.path.join(
                params.knowledge_distillation.teacher_model.checkpoint_path,
                f"{params.model.language_model.model_type}-checkpoint/best.bin",
            )
        )
    )

    for l in seq_lens:
        print(f"Teacher model at {l} tokens :")
        throughput = calculate_nlp_throughput(
            model=teacher_model,
            seq_len=l,
            batch_sizes=batch_sizes,
            device=device,
            attention=True,
        )
        print(f"Throughput: {throughput:.2f} sequences/second")
        mean_time, std_time = measure_model_performance(
            model=teacher_model,
            seq_len=l,
            batch_size=batch_sizes[0],
            device=device,
            attention=True,
        )
        print(
            f"Batch size: {batch_sizes[0]}\nMean time: {mean_time:.2f} ms\nStd dev: {std_time:.2f} ms\n"
        )

        print(f"Student model at {l} tokens :")
        throughput = calculate_nlp_throughput(
            model=student_model,
            seq_len=l,
            batch_sizes=batch_sizes,
            device=device,
            attention=True,
        )
        print(f"Throughput: {throughput:.2f} sequences/second")
        mean_time, std_time = measure_model_performance(
            model=student_model,
            seq_len=l,
            batch_size=batch_sizes[0],
            device=device,
            attention=True,
        )
        print(
            f"Batch size: {batch_sizes[0]}\nMean time: {mean_time:.2f} ms\nStd dev: {std_time:.2f} ms\n"
        )
