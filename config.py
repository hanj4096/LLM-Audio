"""
Configuration file - all training parameters.
"""
import torch
from dataclasses import dataclass


@dataclass
class Config:
    """Training configuration."""
    
    # ========== Data Paths ==========
    train_csv: str = ""
    val_csv: str = ""
    test_csv: str = ""
    audio_prefix: str = ""  # Audio file path prefix
    
    # ========== Model Paths ==========
    qwen_model_path: str = ""
    speech_tokenizer_path: str = ""
    
    # ========== Training Parameters ==========
    batch_size: int = 2
    num_epochs: int = 5
    #learning_rate: float = 2e-6  # Lower LR when training the entire LLM
    #warmup_steps: int = 1000  # More warmup, around 10% of total steps
    learning_rate: float = 5e-6  # Larger LR is fine when LLM is frozen (projection only)
    warmup_steps: int = 1000  # Projection layer converges quickly, fewer warmup steps
    gradient_accumulation_steps: int = 16  # Effective batch size = 16
    max_grad_norm: float = 1.0
    
    # ========== Data Processing ==========
    sample_rate: int = 16000
    max_total_length: int = 2048  # Max total length of audio frames + text tokens
    max_audio_length: int = 30  # Max seconds per audio (for early filtering)
    max_text_length: int = 512  # Max tokens per text (for early filtering)
    audio_downsample_rate: int = 320  # SpeechTokenizer downsampling rate
    
    # ========== Hardware ==========
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True  # FP16 training
    num_workers: int = 0
    
    # ========== Model Settings ==========
    freeze_llm: bool = False  # Whether to freeze LLM parameters
    freeze_speech_encoder: bool = True  # Freeze the speech encoder
    
    # ========== Saving and Logging ==========
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    save_steps: int = 1000  # Lower checkpoint save frequency
    eval_steps: int = 100  # Lower validation frequency for faster training
    logging_steps: int = 10
    max_checkpoints_to_keep: int = 2  # Keep only the most recent N checkpoints (excluding best_model)
    
    # ========== Resume Training ==========
    resume_from_checkpoint: str = ""  # Train from scratch when not loading a checkpoint
    reset_optimizer: bool = False  # Whether to reset optimizer state (set True for fine-tuning)
    reset_scheduler: bool = False  # Whether to reset LR scheduler (set True for fine-tuning)
    
    # ========== Debug Mode ==========
    debug_mode: bool = False  # Use a small subset for quick testing (affects train and val loading)
    debug_samples: int = 10  # Number of samples in debug mode (applies to train + val)
    
    # ========== Validation ==========
    val_samples: int = 200  # Samples used per validation during training (None = full val set)
    eval_samples: int = 10  # Samples used by evaluate.py (None = full val set)
    
    # ========== WandB ==========
    use_wandb: bool = True  # Whether to use WandB
    wandb_api_key: str = ""  # WandB API key (from https://wandb.ai/authorize; None uses env var or existing login)
    wandb_project: str = ""  # WandB project name
    wandb_entity: str = ""  # WandB user/team name (None uses default)
    wandb_run_name: str = ""  # Run name (None = auto-generated)
    wandb_tags: list = None  # Tag list, e.g. ["baseline", "full-training"]
