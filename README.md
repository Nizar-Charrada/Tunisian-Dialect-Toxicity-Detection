# Tunisian Dialect Toxicity Detection
**Note: This AI model is part of a larger project, which is a browser extension aimed at filtering social media content.**

![TUNToxic](docs/TUN.png)
The Tunisian Dialect Toxicity Detection project aims to develop a reliable and efficient system for detecting toxic or offensive language in Tunisian dialect text. The goal is to provide a comprehensive solution that can identify and flag harmful content, promoting safer online communication in the Tunisian dialect.
### Description
This project focuses on building a machine learning model capable of analyzing text data in the Tunisian dialect and determining if it contains toxic or offensive content. The model is trained on a labeled dataset comprising various types of toxic language commonly found in online platforms.
The model supports Tunisian dialect written using Arabic characters (Arabic) and/or Latin characters (Arabizi) through a transliteration mechanism. This allows the model to handle text inputs in either writing system and provide accurate toxicity predictions.

To optimize the model for deployment and resource efficiency, we applied knowledge distillation techniques. Knowledge distillation involves training a smaller model (the student model) to mimic the behavior of a larger, more complex model (the teacher model). By distilling the knowledge from the teacher model, we were able to reduce the size of the model without sacrificing its performance significantly.
This makes the model easier to deploy on various platforms with limited computational resources.
By leveraging knowledge distillation, we achieved a compact and efficient Tunisian dialect toxicity detection model that can be seamlessly deployed in real-world applications.

## Project Structure:
    .
    ├── src                          # This directory contains the source code for data preprocessing, model training, and evaluation.
    │   ├── transliteration          # This directory contains multiple scripts used to train the transliteration model
    │   ├── toxic_detection          # This directory contains multiple scripts used to train the toxicity detection model
    │   └── ensembling.py            # Python script for model ensembling (Arabic model and Arabizi model)
    └── docs
    
### Transliteration/Toxic_detection Structure:

    .
    ├── config                                # Configuration files for model settings
    ├── data                                  # Data directory for storing datasets
    ├── dataloader                            # Data loading and preprocessing modules
    │   ├── collate.py                        # Module for collating data samples
    │   ├── dataset.py                        # Module for creating datasets
    │   └── sampler.py                        # Module for data sampling
    ├── models                                # Model architecture and optimizer modules
    │   ├── model.py                          # Module defining the model architecture
    │   ├── optimizer.py                      # Module defining the optimizer
    ├── notebook                              # Directory for Jupyter notebooks
    ├── out                                   # Output directory for storing logs and model checkpoints
    ├── inference.py                          # Script for performing inference using the trained model
    ├── inference_speed_test.py               # Script for testing the inference speed of the model
    ├── train.py                              # Script for training the model
    └── utils.py                              # Utility functions used throughout the project
    
### Model Used

- **TunBERT for Arabic Characters**: TunBERT is a pre-trained BERT model specifically designed for the Tunisian dialect. It was trained using a Tunisian Common-Crawl-based dataset, which captures the unique linguistic characteristics of the Tunisian dialect.

- **tunbert_zied for Latin Characters (Arabizi)**: tunbert_zied is a language model for the Tunisian dialect, based on a similar architecture to the RoBERTa model. It was developed by Zied Sbabti.

### Datasets Used

## Usage

### Pretrained Model Download
Before using the project, you need to download the pretrained Tun-BERT model. You can obtain the model from the following link: [Pretrained Tun-BERT Model](https://storage.googleapis.com/ext-oss-tunbert-gcp/PyTorch_model/PretrainingBERTFromText--end.ckpt).

### Config
In the configuration file, you can customize the settings according to your requirements. The default settings are already provided, but you may need to modify them based on your specific use case.

### Transliteration
To use the Transliteration module, follow these steps:
1. Set the `source` parameter in the configuration file to `'arabic'` and `target` to `'arabizi'` to get the Arabic-to-Arabizi (Arabic-Arabizi) model.
2. Run the train script.

To switch to transliteration using Arabizi-to-Arabic (Arabizi-Arabic), perform the following:
1. Set the `source` parameter in the configuration file to `'arabizi'` and `target` to `'arabic'` to get the Arabizi-to-Arabic (Arabizi-Arabic) model.
2. Run the train script.

### Toxic Detection
To utilize the Toxic Detection module, take the following steps:
1. Set the `knowledge_distillation_enabled` parameter in the configuration file to `False` to get the teacher model.
2. Run the train script for both arabic and arabizi to get the teacher models.

For knowledge distillation-enabled models, perform the following:
1. Set the `knowledge_distillation_enabled` parameter in the configuration file to `True`.
2. Run train script for arabic and arabizi to get the student models.

### Ensembling
To test all the models together, run the `ensembling.py` script. It will perform evaluation and generate results using all available models.

### Individual Model Testing
To test each model individually, you can use the `inference.py` script within each module. This allows you to perform inference on specific models for detailed evaluation and analysis.

Please ensure that you have the necessary dependencies and resources to run the scripts effectively.

