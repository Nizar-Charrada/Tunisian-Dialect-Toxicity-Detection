import more_itertools
import numpy as np
from torch.utils.data import Sampler

class SmartBatchingSampler(Sampler):
    def __init__(self, data_source, batch_size):
        super(SmartBatchingSampler, self).__init__(data_source)
        self.len = len(data_source)
        
        # Get the lengths of each sequence in the dataset
        sample_lengths = [len(seq) for seq in data_source]
        
        # Sort the indices of the sequences based on their lengths
        argsort_inds = np.argsort(sample_lengths)
        
        # Divide the indices into batches based on the desired batch size
        self.batches = list(more_itertools.chunked(argsort_inds, n=batch_size))
        
        # Initialize the backsort indices to None
        self._backsort_inds = None
    
    def __iter__(self):
        # Shuffle the batches, keeping the last batch as is
        if self.batches:
            last_batch = self.batches.pop(-1)
            np.random.shuffle(self.batches)
            self.batches.append(last_batch)
        
        # Flatten the batches to get the indices in the correct order
        self._inds = list(more_itertools.flatten(self.batches))
        yield from self._inds

    def __len__(self):
        return self.len
    
    @property
    def backsort_inds(self):
        # Sort the indices back into their original order and return them
        if self._backsort_inds is None:
            self._backsort_inds = np.argsort(self._inds)
        return self._backsort_inds