"""
Translation Dataset - Based on Multi30k Characteristics
Supports English-German translation, suitable for image description scenarios

Features:
- UTF-8 encoding support for multilingual characters
- Special tokens: PAD, SOS, EOS, UNK
- Sequence padding and truncation
- Vocabulary building and reverse mapping
- Batch processing support
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import json
import os


class TranslationDataset(Dataset):
    """
    Translation Dataset Class - Based on Multi30k Characteristics
    
    This dataset handles parallel text data for machine translation tasks,
    specifically designed for English-German translation with image descriptions.
    
    Key Features:
    - Supports multiple data formats (memory, files, JSON)
    - Handles special tokens (PAD, SOS, EOS, UNK)
    - Automatic vocabulary building and sequence padding
    - UTF-8 encoding for multilingual support
    - Batch processing with custom collate function
    
    Typical Usage:
        dataset = TranslationDataset(
            src_data=["Hello world", "Good morning"],
            tgt_data=["Hallo Welt", "Guten Morgen"],
            max_length=20
        )
    """
    
    def __init__(
        self,
        src_data: Optional[List[str]] = None,
        tgt_data: Optional[List[str]] = None,
        src_file: Optional[str] = None,
        tgt_file: Optional[str] = None,
        json_file: Optional[str] = None,
        max_length: int = 20,  # Based on Multi30k characteristics, 20 words is sufficient
        min_freq: int = 1,     # Minimum word frequency threshold
        pad_token: str = "<PAD>",
        sos_token: str = "<SOS>",
        eos_token: str = "<EOS>",
        unk_token: str = "<UNK>"
    ):
        """
        Initialize the translation dataset.
        
        Args:
            src_data: List of source language sentences
            tgt_data: List of target language sentences
            src_file: Path to source language file
            tgt_file: Path to target language file
            json_file: Path to JSON format file
            max_length: Maximum sequence length (20 words based on Multi30k characteristics)
            min_freq: Minimum word frequency threshold for vocabulary building
            pad_token: Padding token for sequence alignment
            sos_token: Start-of-sequence token
            eos_token: End-of-sequence token
            unk_token: Unknown word token
            
        Raises:
            ValueError: If no data source is provided
            AssertionError: If source and target data lengths don't match
        """
        # Store all parameters as instance variables
        # Load data based on input parameters (src_data/tgt_data, src_file/tgt_file, or json_file)
        # Ensure source and target language data lengths are consistent
        # Build source and target language vocabularies
        # Create reverse vocabulary mappings (index to word)
        # Print dataset information
        self.src_Data = src_data
        self.tgt_Data = tgt_data
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.json_file = json_file
        self.max_length = max_length
        self.min_freq = min_freq
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        if self.src_file and self.tgt_file:
            self.src_Data, self.tgt_Data = self._load_parallel_files(self.src_file, self.tgt_file)
        elif self.json_file:
            self.src_Data, self.tgt_Data = self._load_json_file(self.json_file)
        elif self.src_Data and self.tgt_Data:
            self.src_Data = src_data
            self.tgt_Data = tgt_data
        else:
            raise ValueError("请提供源语言和目标语言数据或文件路径")

        assert self.src_Data is not None and self.tgt_Data is not None, "Data must be loaded"
        assert len(self.src_Data) == len(self.tgt_Data), "源语言和目标语言数据长度不一致"
        self.src_vocab = self._build_vocab(self.src_Data)
        self.tgt_vocab = self._build_vocab(self.tgt_Data)

        self.src_idx2word = {v:k for k,v in self.src_vocab.items()}
        self.tgt_idx2word = {v:k for k,v in self.tgt_vocab.items()}
        print(f"源语言词汇表大小: {len(self.src_vocab)}")
        print(f"目标语言词汇表大小: {len(self.tgt_vocab)}")
        print(f"源语言数据长度: {len(self.src_Data)}")
        print(f"目标语言数据长度: {len(self.tgt_Data)}")



    
    def _load_parallel_files(self, src_file: str, tgt_file: str) -> Tuple[List[str], List[str]]:
        """
        Load parallel text files.
        
        This method reads source and target language files line by line,
        ensuring they have the same number of lines and proper encoding.
        
        Args:
            src_file: Path to source language file
            tgt_file: Path to target language file
            
        Returns:
            Tuple of (source_data, target_data) lists
            
        File Format Example:
            train.en: "A man is standing in front of a building"
            train.de: "Ein Mann steht vor einem Gebäude"
        """
        # Read files with UTF-8 encoding and strip whitespace
        # Handle potential file reading errors
        with open(src_file, 'r', encoding='utf-8') as f:
            src_data = [line.strip() for line in f.readlines()]
        with open(tgt_file, 'r', encoding='utf-8') as f:
            tgt_data = [line.strip() for line in f.readlines()]
        return src_data, tgt_data
    
    def _load_json_file(self, json_file: str) -> Tuple[List[str], List[str]]:
        """
        Load JSON format file.
        
        This method supports two JSON formats:
        1. List format: [{"src": "...", "tgt": "..."}, ...]
        2. Dict format: {"src": [...], "tgt": [...]}
        
        Args:
            json_file: Path to JSON file
            
        Returns:
            Tuple of (source_data, target_data) lists
            
        JSON Format Examples:
            Format 1: [{"src": "Hello", "tgt": "Hallo"}, ...]
            Format 2: {"src": ["Hello", ...], "tgt": ["Hallo", ...]}
            
        Raises:
            ValueError: If JSON format is not supported
        """
        # Parse JSON and extract source/target data
        # Handle different JSON structures and potential errors
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            src_data = [item['src'] for item in data]
            tgt_data = [item['tgt'] for item in data]
        elif isinstance(data, dict):
            src_data = data['src']
            tgt_data = data['tgt']
        else:
            raise ValueError("JSON文件格式不支持")
        return src_data, tgt_data


    def _build_vocab(self, data: List[str]) -> Dict[str, int]:
        """
        Build vocabulary from text data.
        
        This method creates a vocabulary dictionary by:
        1. Counting word frequencies
        2. Filtering low-frequency words
        3. Adding special tokens at the beginning
        4. Sorting words alphabetically
        5. Assigning unique indices
        
        Args:
            data: List of sentences to build vocabulary from
            
        Returns:
            Dictionary mapping words to indices
            
        Special Token Order:
            - PAD: 0 (padding)
            - SOS: 1 (start of sequence)
            - EOS: 2 (end of sequence)
            - UNK: 3 (unknown word)
            - Other words: 4, 5, 6, ...
        """
        # Count word frequencies and filter by minimum frequency
        # Add special tokens first, then sorted vocabulary words
        word_freq = {}
        for sentence in data:
            for word in sentence.lower().split():
                word_freq[word] = word_freq.get(word, 0) + 1
        filtered_words = {word:freq for word,freq in word_freq.items() if freq >= self.min_freq}
        vocab = {self.pad_token:0, self.sos_token:1, self.eos_token:2, self.unk_token:3}
        for word in sorted(filtered_words.keys()):
            vocab[word] = len(vocab)
        return vocab            

    def _tokenize_and_encode(self, sentence: str, vocab: Dict[str, int]) -> List[int]:
        """
        Tokenize and encode sentence to indices.
        
        This method converts a sentence into a list of word indices,
        handling unknown words with UNK tokens.
        
        Args:
            sentence: Input sentence to tokenize
            vocab: Vocabulary dictionary mapping words to indices
            
        Returns:
            List of word indices
            
        Example:
            Input: "A man is standing"
            Output: [4, 75, 64, 124]  # Example indices
        """
        # Convert to lowercase, split by spaces, and map to indices
        # Use UNK token for unknown words
        sentence_words = sentence.lower().split()
        encoded = []
        for word in sentence_words:
            encoded.append(vocab.get(word,vocab[self.unk_token]))
        return encoded
    
    def _pad_sequence(self, sequence: List[int], vocab: Dict[str, int]) -> List[int]:
        """
        Pad sequence to fixed length.
        
        This method ensures all sequences have the same length by:
        1. Truncating sequences that are too long (leaving space for EOS)
        2. Adding EOS token at the end
        3. Padding with PAD tokens to reach max_length
        
        Args:
            sequence: Input sequence of word indices
            vocab: Vocabulary dictionary for token indices
            
        Returns:
            Padded sequence of length max_length
            
        Example:
            Input: [4, 75, 64] (length 3)
            Output: [4, 75, 64, 2, 0, 0, ...] (length 20, EOS=2, PAD=0)
        """
        # Truncate if too long, add EOS token, pad to max_length
        if len(sequence) > self.max_length - 1:
            sequence = sequence[:self.max_length -1]
        sequence.append(vocab[self.eos_token])
        while len(sequence) < self.max_length:
            sequence.append(vocab[self.pad_token])
        return sequence

    def __len__(self) -> int:
        """
        Return the size of the dataset.
        
        Returns:
            Number of samples in the dataset
        """
        # Return the number of samples in the dataset
        assert self.src_Data is not None and self.tgt_Data is not None, "Data must be loaded"
        assert len(self.src_Data) == len(self.tgt_Data), "源语言和目标语言数据长度不一致"
        return len(self.src_Data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        """
        Get a single sample from the dataset.
        
        This method retrieves a sample at the given index and processes it:
        1. Gets source and target sentences
        2. Tokenizes and encodes both sentences
        3. Pads sequences to fixed length
        4. Creates target input sequence (with SOS token)
        5. Returns all necessary information
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing:
            - 'src': Source language encoded sequence [seq_len]
            - 'tgt_input': Target language input sequence [seq_len] (with SOS)
            - 'tgt_output': Target language output sequence [seq_len] (with EOS)
            - 'src_text': Source language original text
            - 'tgt_text': Target language original text
        """
        # Get source and target sentences, encode and pad them
        # Create target input sequence with SOS token
        assert self.src_Data is not None and self.tgt_Data is not None, "Data must be loaded"
        src_sentence = self.src_Data[idx]
        tgt_sentence = self.tgt_Data[idx]
        src_encoded = self._tokenize_and_encode(src_sentence, self.src_vocab)
        src_encoded = self._pad_sequence(src_encoded, self.src_vocab)
        tgt_encoded = self._tokenize_and_encode(tgt_sentence, self.tgt_vocab)
        tgt_encoded = self._pad_sequence(tgt_encoded, self.tgt_vocab)
        tgt_input = [self.tgt_vocab[self.sos_token]] + tgt_encoded[:-1]
        tgt_output = tgt_encoded

        return {'src': torch.tensor(src_encoded, dtype=torch.long), 
        'tgt_input': torch.tensor(tgt_input, dtype=torch.long),
        'tgt_output': torch.tensor(tgt_output, dtype=torch.long),
        'src_text': src_sentence,
        'tgt_text': tgt_sentence
        }  

    def get_vocab_sizes(self) -> Tuple[int, int]:
        """
        Get vocabulary sizes for source and target languages.
        
        This method returns the size of both vocabularies,
        which is needed for model initialization.
        
        Returns:
            Tuple of (source_vocab_size, target_vocab_size)
        """
        # Return vocabulary sizes as a tuple
        return len(self.src_vocab), len(self.tgt_vocab)
    
    def decode_sequence(self, sequence: torch.Tensor, is_target: bool = True) -> str:
        """
        Decode sequence of indices back to text.
        
        This method converts a sequence of word indices back to readable text,
        handling special tokens like PAD and EOS appropriately.
        
        Args:
            sequence: Tensor of word indices to decode
            is_target: Whether to use target vocabulary (True) or source vocabulary (False)
            
        Returns:
            Decoded text string
            
        Example:
            Input: tensor([4, 75, 64, 2, 0, 0, ...])
            Output: "a man is standing"
        """
        # Convert tensor to list, select appropriate vocabulary
        # Skip PAD tokens and stop at EOS token

        sequence_list = sequence.tolist()
        words = []
        
        if is_target:
            vocab = self.tgt_vocab
            idx2word = self.tgt_idx2word
        else:
            vocab = self.src_vocab
            idx2word = self.src_idx2word
        
        for idx in sequence_list:
            if idx == vocab[self.eos_token]:
                break  # 遇到EOS标记停止
            if idx != vocab[self.pad_token]:
                words.append(idx2word[idx])
        
        return ' '.join(words)

def create_dataloader(
    dataset: TranslationDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create a DataLoader for the translation dataset.
    
    This function creates a PyTorch DataLoader with appropriate settings
    for training and evaluation.
    
    Args:
        dataset: TranslationDataset instance
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        PyTorch DataLoader instance
    """
    # Create DataLoader with custom collate function
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching samples.
    
    This function combines multiple samples into a single batch by stacking
    tensors along the batch dimension.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Dictionary with batched tensors
        
    Input Format:
        batch = [
            {'src': tensor([...]), 'tgt_input': tensor([...]), 'tgt_output': tensor([...])},
            {'src': tensor([...]), 'tgt_input': tensor([...]), 'tgt_output': tensor([...])},
            ...
        ]
        
    Output Format:
        {
            'src': tensor([batch_size, seq_len]),
            'tgt_input': tensor([batch_size, seq_len]),
            'tgt_output': tensor([batch_size, seq_len])
        }
    """
    # Extract fields from each sample and stack them into batches
    src = [item['src'] for item in batch]
    tgt_input = [item['tgt_input'] for item in batch]
    tgt_output = [item['tgt_output'] for item in batch]
    return {
        'src': torch.stack(src),
        'tgt_input': torch.stack(tgt_input),
        'tgt_output': torch.stack(tgt_output)
    }


def create_sample_dataset() -> TranslationDataset:
    """
    Create a sample dataset for testing purposes.
    
    This function creates a small translation dataset using either
    the existing JSON file or hardcoded sample data.
    
    Returns:
        TranslationDataset instance with sample data
    """
    # Use existing JSON file if available
    json_file = "data/train.json"
    if os.path.exists(json_file):
        return TranslationDataset(json_file=json_file, max_length=20)
    else:
        # Create simple sample data if no file exists
        src_data = [
            "A man is standing in front of a building",
            "A woman is walking on the street",
            "A child is playing in the park"
        ]
        
        tgt_data = [
            "Ein Mann steht vor einem Gebäude",
            "Eine Frau geht auf der Straße", 
            "Ein Kind spielt im Park"
        ]
        
        return TranslationDataset(
            src_data=src_data,
            tgt_data=tgt_data,
            max_length=20
        )


if __name__ == "__main__":
    # Test the dataset
    dataset = create_sample_dataset()
    
    print("\\n=== Dataset Testing ===")
    print(f"Dataset size: {len(dataset)}")
    print(f"Source vocabulary size: {dataset.get_vocab_sizes()[0]}")
    print(f"Target vocabulary size: {dataset.get_vocab_sizes()[1]}")
    
    # Test single sample
    sample = dataset[0]
    print(f"\\nFirst sample:")
    print(f"Source: {sample['src_text']}")
    print(f"Target: {sample['tgt_text']}")
    print(f"Source encoded: {sample['src']}")
    print(f"Target input: {sample['tgt_input']}")
    print(f"Target output: {sample['tgt_output']}")
    
    # Test decoding
    decoded_src = dataset.decode_sequence(sample['src'], is_target=False)  # type: ignore
    decoded_tgt = dataset.decode_sequence(sample['tgt_output'], is_target=True)  # type: ignore
    print(f"\\nDecoding results:")
    print(f"Source decoded: {decoded_src}")
    print(f"Target decoded: {decoded_tgt}")
    
    # Test data loader
    dataloader = create_dataloader(dataset, batch_size=2)
    print(f"\\nDataLoader testing:")
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}: src shape: {batch['src'].shape}, tgt shape: {batch['tgt_input'].shape}")
        if i >= 2:  # Show only first 3 batches
            break
