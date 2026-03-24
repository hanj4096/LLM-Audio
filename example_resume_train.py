"""
Example code for resumed training and fine-tuning.
"""
from config import Config
from train import Trainer


def example_resume_training():
    """Example 1: resume pretraining from an interrupted checkpoint."""
    print("="*70)
    print("Example 1: resume pretraining from an interrupted checkpoint.")
    print("="*70)
    
    config = Config()
    
    # Set checkpoint to resume from
    config.resume_from_checkpoint = "checkpoints/checkpoint_step_1000.pt"
    
    # Keep original optimizer and scheduler states
    config.reset_optimizer = False
    config.reset_scheduler = False
    
    # Keep other parameters unchanged
    config.num_epochs = 10  # Continue training for more epochs
    
    trainer = Trainer(config)
    trainer.train()


def example_finetune():
    """Example 2: fine-tune using new data from a pretrained model."""
    print("="*70)
    print("Example 2: fine-tune using new data from a pretrained model.")
    print("="*70)
    
    config = Config()
    
    # Load pretrained model
    config.resume_from_checkpoint = "checkpoints/best_model.pt"
    
    # Reset optimizer and scheduler (re-optimize from scratch)
    config.reset_optimizer = True
    config.reset_scheduler = True
    
    # Fine-tuning hyperparameters
    config.learning_rate = 1e-5  # Use a smaller learning rate for fine-tuning
    config.num_epochs = 3
    config.warmup_steps = 100
    
    # Optionally freeze part of the model
    config.freeze_llm = False  # Fine-tune the LLM
    config.freeze_speech_encoder = True  # Keep speech encoder frozen
    
    # New dataset
    config.train_csv = "/path/to/finetune_train.csv"
    config.val_csv = "/path/to/finetune_val.csv"
    
    trainer = Trainer(config)
    trainer.train()


def example_from_scratch():
    """Example 3: train from scratch without loading checkpoints."""
    print("="*70)
    print("Example 3: train from scratch without loading checkpoints.")
    print("="*70)
    
    config = Config()
    
    # Do not set resume_from_checkpoint, start from scratch
    config.resume_from_checkpoint = None
    
    config.num_epochs = 10
    config.learning_rate = 5e-5
    
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    # Choose which example to run
    
    # Example 1: resume pretraining
    # example_resume_training()
    
    # Example 2: fine-tuning
    # example_finetune()
    
    # Example 3: training from scratch
    example_from_scratch()
