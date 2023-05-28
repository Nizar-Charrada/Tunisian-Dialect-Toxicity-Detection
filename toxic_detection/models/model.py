import torch.nn as nn
import torch
from transformers import AutoModel
import warnings
import argparse

warnings.filterwarnings("ignore")


class TeacherModel(nn.Module):
    def __init__(self, params, config, freeze=False):
        super(TeacherModel, self).__init__()

        # Initialize the BERT model with the specified pre-trained checkpoint and configuration
        self.config = config
        self.bert_model = AutoModel.from_pretrained(
            params.model.language_model.model_name_or_path, config=config
        )

        # Add a linear layer on top of BERT to output a single scalar value
        self.fc = nn.Linear(config.hidden_size, 1)

        self.linear = nn.Linear(
            config.hidden_size, config.hidden_size
        )  # i will remove this later on, just for testing purposes
        # Add a dropout layer to prevent overfitting
        self.dropout = nn.Dropout(p=params.model.classifier_head.fc_dropout)

        # If specified, freeze the weights of the BERT model
        if freeze:
            for p in self.bert_model.parameters():
                p.requires_grad = False

        # Print the number of parameters in the model
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def forward(self, input_ids, attention_mask):
        # Feed the input to the BERT model and obtain the output
        output = self.bert_model(input_ids, attention_mask)

        # Get the pooled output for the [CLS] token
        pooler_output = output[1]

        # Pass the pooled output through the linear layer and apply dropout
        out = self.fc(self.dropout(pooler_output))

        # Return the output
        return out

    def get_num_params(self, non_embedding=False):
        """
        Return the number of parameters in the model.
        For non-embedding count, the position embeddings get subtracted.
        """
        # Count the number of parameters in the model
        n_params = sum(p.numel() for p in self.parameters())

        # If non_embedding is True, subtract the number of position embeddings from the count
        if non_embedding:
            n_params -= self.config.max_position_embeddings * self.config.hidden_size

        # Return the final count
        return n_params


class GlobalMaxPooling1D(nn.Module):
    """Global max pooling operation for temporal data."""

    def __init__(self, data_format="channels_last"):
        super(GlobalMaxPooling1D, self).__init__()
        self.data_format = data_format
        self.step_axis = 1 if self.data_format == "channels_last" else 2

    def forward(self, input):
        return torch.max(input, axis=self.step_axis).values


class StudentModel(nn.Module):
    def __init__(
        self, vocab_size, embed_size, hidden_dim, kernel_size=3, add_conv_layer=False
    ):
        super(StudentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # i dont like this part, i will optimize it later on
        if add_conv_layer:
            self.linear = nn.Sequential(
                nn.Conv1d(
                    embed_size, hidden_dim, 3
                ),  # (batch_size,embed_size,sentence_length) -> (batch_size,hidden_dim,sentence_length-2)
                nn.ReLU(),
                nn.Conv1d(
                    hidden_dim, hidden_dim, 2, padding=2
                ),  # (batch_size,hidden_dim,sentence_length-2) -> (batch_size,hidden_dim,sentence_length-2)
                nn.ReLU(),
                GlobalMaxPooling1D(
                    "channels_2"
                ),  # (batch_size,hidden_dim,sentence_length) -> (batch_size,hidden_dim
            )
        else:
            self.linear = nn.Sequential(
                nn.Conv1d(
                    embed_size, hidden_dim, kernel_size
                ),  # (batch_size,embed_size,sentence_length) -> (batch_size,hidden_dim,sentence_length-2)
                nn.ReLU(),
                GlobalMaxPooling1D(
                    "channels_2"
                ),  # (batch_size,hidden_dim,sentence_length) -> (batch_size,hidden_dim
            )

        self.fc = nn.Linear(hidden_dim, 1)  # (batch_size,hidden_dim) -> (batch_size,1)

        # Print the number of parameters in the model
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def forward(self, tokens, attention_mask=None):
        # tokens: (batch_size,sentence_length)
        embeds = self.embedding(tokens)
        # embeds: (batch_size,sentence_length,embed_size)
        embeds = embeds.permute(0, 2, 1)
        # embeds: (batch_size,embed_size,sentence_length)
        output = self.linear(embeds)
        # output: (batch_size,hidden_dim)
        output = output.unsqueeze(1)
        # output: (batch_size,1,hidden_dim)
        output = self.fc(output)
        # output: (batch_size,1,1)
        return output.squeeze(-1)  # (batch_size,1)

    def get_num_params(self, non_embedding=False):
        """
        Return the number of parameters in the model.
        For non-embedding count, the position embeddings get subtracted.
        """
        # Count the number of parameters in the model
        n_params = sum(p.numel() for p in self.parameters())

        # If non_embedding is True, subtract the number of position embeddings from the count
        if non_embedding:
            n_params -= self.config.max_position_embeddings * self.config.hidden_size

        # Return the final count
        return n_params
