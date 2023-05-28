import torch
from torch.nn.utils.rnn import pad_sequence


class SmartBatchingCollate:
    def __init__(self, targets, max_length, pad_token_id):
        """
        Initialize the collate function with targets, max sequence length,
        and pad token id.

        Args:
            targets (bool): Whether the sequences have targets (i.e., labels).
            max_length (int): The maximum sequence length to pad to.
            pad_token_id (int): The id of the padding token.
        """
        self._targets = targets
        self._max_length = max_length
        self._pad_token_id = pad_token_id

    def __call__(self, batch):
        """
        Collate function that takes a batch of sequences and returns a padded batch.

        Args:
            batch (list): A list of sequences.

        Returns:
            output (tuple): A tuple containing the padded input ids and attention masks.
        """
        if self._targets is not None:
            sequences, targets = list(zip(*batch))
        else:
            sequences = list(batch)

        # Pad the sequences and create attention masks
        input_ids, attention_mask = self.pad_sequence(
            sequences,
            max_sequence_length=self._max_length,
            pad_token_id=self._pad_token_id,
        )

        # If there are targets, add them to the output tuple
        if self._targets is not None:
            output = input_ids, attention_mask, torch.tensor(targets)
        else:
            output = input_ids, attention_mask
        return output

    def pad_sequence(self, sequence_batch, max_sequence_length, pad_token_id):
        """
        Pad a batch of sequences to a maximum length and create attention masks.

        Args:
            sequence_batch (list): A list of sequences.
            max_sequence_length (int): The maximum sequence length to pad to.
            pad_token_id (int): The id of the padding token.

        Returns:
            padded_sequences (torch.Tensor): A tensor containing the padded sequences.
            attention_masks (torch.Tensor): A tensor containing the attention masks.
        """
        # Determine the maximum length of the sequences
        max_batch_len = max(len(sequence) for sequence in sequence_batch)

        # Limit the maximum length to the specified maximum
        max_len = min(max_batch_len, max_sequence_length)

        # Create empty lists to store the padded sequences and attention masks
        padded_sequences, attention_masks = [[] for i in range(2)]

        # Define constants for the attention mask values
        attend, no_attend = 1, 0

        # Iterate over each sequence in the batch
        for sequence in sequence_batch:
            # Truncate the sequence if it is longer than the maximum
            new_sequence = list(sequence[:max_len])

            # Create an attention mask that identifies padding tokens
            attention_mask = [attend] * len(new_sequence)
            pad_length = max_len - len(new_sequence)
            attention_mask.extend([no_attend] * pad_length)

            # Pad the sequence with the padding token
            new_sequence.extend([pad_token_id] * pad_length)

            # Append the padded sequence and attention mask to their respective lists
            padded_sequences.append(new_sequence)
            attention_masks.append(attention_mask)

        # Convert the padded sequences and attention masks to tensors
        padded_sequences = torch.tensor(padded_sequences)
        attention_masks = torch.tensor(attention_masks)

        return padded_sequences, attention_masks
