import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
import os


class FiniteImpulseResponseDataSet(Dataset):
    """
    A PyTorch Dataset for finite impulse response data.
    
    This dataset loads input and output WAV files and returns tuples of
    (input_buffer, output_sample) for training FIR models.
    
    Args:
        input_wav_path (str): Path to the input WAV file
        output_wav_path (str): Path to the output WAV file
        buffer_length (int): Number of input samples to use as context
    """
    
    def __init__(self, input_wav_path: str, output_wav_path: str, buffer_length: int):
        """
        Initialize the dataset.
        
        Args:
            input_wav_path (str): Path to the input WAV file
            output_wav_path (str): Path to the output WAV file
            buffer_length (int): Number of input samples to use as context
        """
        if not os.path.exists(input_wav_path):
            raise FileNotFoundError(f"Input WAV file not found: {input_wav_path}")
        if not os.path.exists(output_wav_path):
            raise FileNotFoundError(f"Output WAV file not found: {output_wav_path}")
        
        self.input_wav_path = input_wav_path
        self.output_wav_path = output_wav_path
        self.buffer_length = buffer_length
        
        # Load the audio files
        self.input_waveform, self.input_sample_rate = torchaudio.load(input_wav_path)
        self.output_waveform, self.output_sample_rate = torchaudio.load(output_wav_path)
        
        # Verify sample rates match
        if self.input_sample_rate != self.output_sample_rate:
            raise ValueError(f"Sample rates must match: input={self.input_sample_rate}, output={self.output_sample_rate}")
        
        self.sample_rate = self.input_sample_rate
        
        # Get the number of channels and total samples
        self.input_channels, self.input_total_samples = self.input_waveform.shape
        self.output_channels, self.output_total_samples = self.output_waveform.shape
        
        # Verify both files have the same length
        if self.input_total_samples != self.output_total_samples:
            raise ValueError(f"WAV files must have same length: input={self.input_total_samples}, output={self.output_total_samples}")
        
        self.total_samples = self.input_total_samples
        
        # Calculate the number of possible items
        # We need buffer_length samples before position i, so we start at index buffer_length
        self.num_items = max(0, self.total_samples - self.buffer_length)
        
        if self.total_samples < self.buffer_length:
            raise ValueError(f"WAV files have {self.total_samples} samples, but buffer_length is {buffer_length}. "
                           f"Buffer length must be <= total samples.")
    
    def __len__(self):
        """Return the number of items in the dataset."""
        return self.num_items
    
    def __getitem__(self, idx):
        """
        Return a tuple of (input_buffer, output_sample).
        
        Args:
            idx (int): Index of the item to retrieve
            
        Returns:
            tuple: (input_buffer, output_sample) where:
                - input_buffer: Tensor of shape (num_channels, buffer_length) containing
                               samples from (i-buffer_length) to i from the input
                - output_sample: Tensor of shape (num_channels,) containing the i-th
                                sample from the output
        """
        if idx >= self.num_items:
            raise IndexError(f"Index {idx} is out of range for dataset of length {self.num_items}")
        
        # Calculate the position i (the current output sample position)
        i = idx + self.buffer_length
        
        # Extract input buffer: samples from (i-buffer_length) to i (exclusive end)
        input_buffer = self.input_waveform[:, i-self.buffer_length:i]
        
        # Extract output sample: the i-th sample
        output_sample = self.output_waveform[:, i]
        
        return input_buffer, output_sample
    
    def get_sample_rate(self):
        """Return the sample rate of the audio file."""
        return self.sample_rate
    
    def get_input_channels(self):
        """Return the number of channels in the input audio file."""
        return self.input_channels
    
    def get_output_channels(self):
        """Return the number of channels in the output audio file."""
        return self.output_channels
    
    def get_total_samples(self):
        """Return the total number of samples in the audio file."""
        return self.total_samples


class FiniteImpulseResponseModel(nn.Module):
    """
    A neural network model for learning finite impulse response behavior.
    
    This model processes input audio buffers through alternating Linear and ReLU layers
    to predict output audio samples.
    
    Args:
        input_size (int): Size of the input (buffer_length * num_channels)
        hidden_size (int): Size of the hidden layers
        num_layers (int): Number of hidden layers (must be >= 1)
        output_size (int): Size of the output (typically num_channels)
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int = 1):
        """
        Initialize the FIR model.
        
        Args:
            input_size (int): Size of the input (buffer_length * num_channels)
            hidden_size (int): Size of the hidden layers
            num_layers (int): Number of hidden layers (must be >= 1)
            output_size (int): Size of the output (typically num_channels), default is 1
        """
        super(FiniteImpulseResponseModel, self).__init__()
        
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # Build the network layers
        layers = []
        
        # Flatten layer: (batch, channels, buffer_length) -> (batch, channels * buffer_length)
        layers.append(nn.Flatten(start_dim=1))
        
        # First layer: input_size -> hidden_size
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        # Hidden layers: hidden_size -> hidden_size
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        # Output layer: hidden_size -> output_size (no activation)
        layers.append(nn.Linear(hidden_size, output_size))
        
        # Create sequential model
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, buffer_length)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        return self.network(x)


def train_epoch(dataloader, model, loss_fn, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        dataloader: DataLoader containing training data
        model: The model to train
        loss_fn: Loss function to use
        optimizer: Optimizer for updating weights
        device: Device to run training on (cpu or cuda)
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 5000 == 0:
            loss_value, current = loss.item(), batch * len(X)
            print(f"loss: {loss_value:>7f}  [{current:>7d}/{size:>7d}]")


def evaluate(dataloader, model, loss_fn, device):
    """
    Evaluate the model on a dataset.
    
    Args:
        dataloader: DataLoader containing test/validation data
        model: The model to evaluate
        loss_fn: Loss function to use
        device: Device to run evaluation on (cpu or cuda)
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


def transform(model, input_wav_path, output_wav_path, buffer_length, device=None):
    """
    Apply a trained FIR model to an input audio file to generate an output audio file.
    
    Args:
        model: The trained FiniteImpulseResponseModel
        input_wav_path (str): Path to the input WAV file
        output_wav_path (str): Path to save the output WAV file
        buffer_length (int): Number of input samples to use as context (must match training)
        device: Device to run inference on (cpu or cuda). If None, uses cuda if available
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load input audio
    input_waveform, sample_rate = torchaudio.load(input_wav_path)
    num_channels, total_samples = input_waveform.shape
    
    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    # Prepare output tensor
    output_waveform = torch.zeros_like(input_waveform)
    
    # For the first buffer_length samples, we don't have enough history
    # So we'll just copy them (or use zeros)
    output_waveform[:, :buffer_length] = input_waveform[:, :buffer_length]
    
    # Process the rest of the audio
    print(f"Processing {total_samples} samples...")
    with torch.no_grad():
        for i in range(buffer_length, total_samples):
            # Extract input buffer: samples from (i-buffer_length) to i
            input_buffer = input_waveform[:, i-buffer_length:i].unsqueeze(0).to(device)
            
            # Generate output sample
            output_sample = model(input_buffer)
            
            # Store output (move back to CPU and squeeze batch dimension)
            output_waveform[:, i] = output_sample.cpu().squeeze(0)
            
            # Print progress every 100k samples
            if (i - buffer_length) % 100000 == 0:
                progress = (i - buffer_length) / (total_samples - buffer_length) * 100
                print(f"Progress: {progress:.1f}% ({i}/{total_samples} samples)")
    
    print("Processing complete!")
    
    # Save output audio
    torchaudio.save(output_wav_path, output_waveform, sample_rate)
    print(f"Output saved to: {output_wav_path}")
