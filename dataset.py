"""
Dataset loading - dynamic length management version.
"""
import pandas as pd
import torch
import librosa
from torch.utils.data import Dataset
from pathlib import Path


class AudioDataset(Dataset):
    """Audio dataset with dynamic length management support."""
    
    def __init__(
        self,
        csv_path: str,
        tokenizer,
        audio_prefix: str = "",
        sample_rate: int = 16000,
        max_total_length: int = 2048,
        max_audio_length: int = 30,
        max_text_length: int = 512,
        audio_downsample_rate: int = 320,
        debug_mode: bool = False,
        debug_samples: int = 10,
    ):
        """
        Args:
            csv_path: CSV path (first column: audio path, second column: text)
            tokenizer: text tokenizer
            audio_prefix: audio path prefix
            sample_rate: sampling rate
            max_total_length: max total length of audio frames + text tokens
            max_audio_length: max seconds per audio (for early filtering)
            max_text_length: max tokens per text
            audio_downsample_rate: audio downsampling rate (SpeechTokenizer)
            debug_mode: whether to load only a small subset for testing
            debug_samples: number of samples in debug mode
        """
        self.tokenizer = tokenizer
        self.audio_prefix = audio_prefix
        self.sample_rate = sample_rate
        self.max_total_length = max_total_length
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        self.audio_downsample_rate = audio_downsample_rate
        
        # Read CSV (header columns: Audio:FILE and Text:LABEL)
        df = pd.read_csv(csv_path, header=0)
        # Rename columns to standard names
        df.columns = ['audio_path', 'text']
        
        # Debug mode: use only a small subset
        if debug_mode:
            df = df.head(debug_samples)
            print(f"[Dataset] Debug mode: using {len(df)} samples")
        
        # Filter invalid samples (missing files)
        self.data = []
        invalid_count = 0
        for item in df.to_dict('records'):
            audio_path = Path(audio_prefix) / item['audio_path']
            if audio_path.exists():
                self.data.append(item)
            else:
                invalid_count += 1
        
        print(f"[Dataset] Loaded {len(self.data)} valid samples from {csv_path}")
        if invalid_count > 0:
            print(f"[Dataset] Skipped {invalid_count} invalid samples (file not found)")
        
        # Track failed sample indices found at runtime
        self._failed_indices = set()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # If current index is known to fail, skip it
        if idx in self._failed_indices:
            # Find the next valid index
            for offset in range(1, len(self.data)):
                next_idx = (idx + offset) % len(self.data)
                if next_idx not in self._failed_indices:
                    idx = next_idx
                    break
            else:
                # All samples failed (edge case)
                print(f"[Error] All samples failed")
                return None
        
        # Try loading the current sample
        try:
            item = self.data[idx]
            
            # Load audio
            audio_path = Path(self.audio_prefix) / item['audio_path']
            waveform, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
            
            # Early clip on audio length (avoid very long audio)
            max_samples = int(self.max_audio_length * self.sample_rate)
            if len(waveform) > max_samples:
                waveform = waveform[:max_samples]
            
            # Convert to tensor (no padding here; handled dynamically in collate_fn)
            waveform = torch.FloatTensor(waveform)
            
            # Process text
            text = item['text']
            
            return {
                'audio': waveform,
                'text': text,
            }
        except Exception as e:
            # Record failed index
            self._failed_indices.add(idx)
            print(f"[Warning] Failed to load idx={idx} ({item['audio_path']}): {e}")
            
            # Recursively try the next sample
            next_idx = (idx + 1) % len(self.data)
            if next_idx == idx or len(self._failed_indices) >= len(self.data):
                # Avoid infinite recursion
                print(f"[Error] Too many failed samples ({len(self._failed_indices)}/{len(self.data)})")
                return None
            
            return self.__getitem__(next_idx)


def collate_fn(batch, tokenizer, max_total_length=2048, audio_downsample_rate=320):
    """
    Batch collation with dynamic length management.
    
    Strategy:
    1. Compute audio frame count and text token count per sample.
    2. If total length exceeds the limit, keep text intact and trim audio first.
    3. Pad to the max length in the batch (instead of a fixed global length).
    """
    # Filter out None samples (failed loading)
    batch = [item for item in batch if item is not None]
    
    # If whole batch is None, return an empty batch
    if len(batch) == 0:
        return None
    
    processed_items = []
    
    for item in batch:
        audio = item['audio']
        text = item['text']
        
        # Compute audio frame count after downsampling
        audio_frames = len(audio) // audio_downsample_rate
        
        # Pre-encode text to get token count (with prompt)
        full_text = "<|im_start|>assistant\n" + text + "<|im_end|>"
        text_tokens = tokenizer.encode(full_text, add_special_tokens=False)
        text_length = len(text_tokens)
        
        # Check total sequence length
        # Note: actual sequence = [audio embeddings] + [text tokens (with special tokens)]
        # Audio embeddings do not map to token IDs but still occupy sequence positions
        total_length = audio_frames + text_length
        
        if total_length > max_total_length:
            # Strategy: preserve text first, trim audio
            # Keep at least 3s audio (150 frames = 3s x 16000Hz / 320)
            min_audio_frames = 150
            max_audio_frames = max(max_total_length - text_length, min_audio_frames)
            max_audio_samples = max_audio_frames * audio_downsample_rate
            
            if len(audio) > max_audio_samples:
                audio = audio[:max_audio_samples]
                audio_frames = max_audio_frames
        
        processed_items.append({
            'audio': audio,
            'audio_length': len(audio),  # Actual number of samples
            'text': text,
        })
    
    # Pad to max length within the batch
    max_audio_len = max(item['audio_length'] for item in processed_items)
    
    padded_audios = []
    audio_lengths = []
    texts = []
    
    for item in processed_items:
        audio = item['audio']
        audio_length = item['audio_length']
        
        # Pad audio
        if len(audio) < max_audio_len:
            padded = torch.nn.functional.pad(audio, (0, max_audio_len - len(audio)))
        else:
            padded = audio
        
        padded_audios.append(padded)
        audio_lengths.append(audio_length)
        texts.append(item['text'])
    
    return {
        'audio': torch.stack(padded_audios),
        'audio_lengths': torch.tensor(audio_lengths, dtype=torch.long),
        'texts': texts,
    }
