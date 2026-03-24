"""
Training script.
"""
import os
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast as cuda_autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from config import Config
from dataset import AudioDataset, collate_fn
from model import AudioModel

# WandB (optional dependency)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[Warning] wandb not installed. Run: pip install wandb")


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Initialize model
        print("\n[Trainer] Initializing model...")
        self.model = AudioModel(
            qwen_model_path=config.qwen_model_path,
            speech_tokenizer_path=config.speech_tokenizer_path,
            freeze_llm=config.freeze_llm,
            freeze_speech_encoder=config.freeze_speech_encoder,
            audio_downsample_rate=config.audio_downsample_rate,
        ).to(self.device)
        
        # Load data
        print("\n[Trainer] Loading data...")
        self.train_dataset = AudioDataset(
            csv_path=config.train_csv,
            tokenizer=self.model.tokenizer,
            audio_prefix=config.audio_prefix,
            sample_rate=config.sample_rate,
            max_total_length=config.max_total_length,
            max_audio_length=config.max_audio_length,
            max_text_length=config.max_text_length,
            audio_downsample_rate=config.audio_downsample_rate,
            debug_mode=config.debug_mode,
            debug_samples=config.debug_samples,
        )
        
        self.val_dataset = AudioDataset(
            csv_path=config.val_csv,
            tokenizer=self.model.tokenizer,
            audio_prefix=config.audio_prefix,
            sample_rate=config.sample_rate,
            max_total_length=config.max_total_length,
            max_audio_length=config.max_audio_length,
            max_text_length=config.max_text_length,
            audio_downsample_rate=config.audio_downsample_rate,
            debug_mode=config.debug_mode,
            debug_samples=config.debug_samples,
        )
        
        # Build a partial collate_fn bound with tokenizer and config
        from functools import partial
        collate_fn_with_config = partial(
            collate_fn,
            tokenizer=self.model.tokenizer,
            max_total_length=config.max_total_length,
            audio_downsample_rate=config.audio_downsample_rate,
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn_with_config,
            num_workers=config.num_workers,
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn_with_config,
            num_workers=config.num_workers,
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )
        
        # Learning rate scheduler
        total_steps = len(self.train_loader) * config.num_epochs // config.gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps,
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None
        
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.start_epoch = 0
        
        # Resume training from checkpoint
        if config.resume_from_checkpoint:
            self.load_checkpoint(config.resume_from_checkpoint)
        
        # Initialize WandB
        self.use_wandb = config.use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            # If an API key is provided, login first
            if config.wandb_api_key:
                wandb.login(key=config.wandb_api_key)
                print(f"\n[Trainer] WandB API key is set")
            
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.wandb_run_name,
                tags=config.wandb_tags if config.wandb_tags else [],
                config={
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "num_epochs": config.num_epochs,
                    "gradient_accumulation_steps": config.gradient_accumulation_steps,
                    "max_grad_norm": config.max_grad_norm,
                    "warmup_steps": config.warmup_steps,
                    "freeze_llm": config.freeze_llm,
                    "freeze_speech_encoder": config.freeze_speech_encoder,
                    "total_steps": total_steps,
                    "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
                }
            )
            # Monitor model gradients
            wandb.watch(self.model, log="gradients", log_freq=100)
            print(f"\n[Trainer] WandB is initialized: {wandb.run.name}")
        elif config.use_wandb and not WANDB_AVAILABLE:
            print("\n[Warning] WandB is not installed, skipping logging")
        
        print(f"\n[Trainer] Training configuration:")
        print(f"  Training samples: {len(self.train_dataset)}")
        print(f"  Validation samples: {len(self.val_dataset)}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
        print(f"  Total training steps: {total_steps}")
        print(f"  Starting step: {self.global_step}")
        print(f"  Starting epoch: {self.start_epoch}")
    
    def train(self):
        """Training loop."""
        print("\n" + "="*70)
        print("Training started" if self.start_epoch == 0 else f"Training resumed from epoch {self.start_epoch + 1}")
        print("="*70 + "\n")
        
        for epoch in range(self.start_epoch, self.config.num_epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*70}")
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            # Log to WandB
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss_epoch": train_loss,
                    "val_loss_epoch": val_loss,
                    "best_val_loss": self.best_val_loss,
                }, step=self.global_step)
            
            print(f"\nEpoch {epoch + 1} completed:")
            print(f"  Training loss: {train_loss:.4f}")
            print(f"  Validation loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pt")
                print(f"  ✓ Save best model (loss: {val_loss:.4f})")
        
        # Save final model
        self.save_checkpoint("final_model.pt")
        
        # Finish WandB run
        if self.use_wandb:
            wandb.finish()
        
        print("\nTraining completed!")
    
    def train_epoch(self):
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        self.optimizer.zero_grad()
        
        valid_step = 0  # Count valid batches only
        pbar = tqdm(self.train_loader, desc="Training")
        for step, batch in enumerate(pbar):
            # Skip empty batches (all samples failed to load)
            if batch is None:
                continue
            
            audio = batch['audio'].to(self.device)
            audio_lengths = batch['audio_lengths'].to(self.device)
            texts = batch['texts']
            
            # Forward pass
            if self.scaler:
                with cuda_autocast():
                    loss = self.model(audio, texts, audio_lengths)
                loss = loss / self.config.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                loss = self.model(audio, texts, audio_lengths)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            valid_step += 1  # Count valid batches only
            
            # Gradient accumulation (based on valid-step count)
            if valid_step % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / valid_step  # Use valid-step count
                    lr = self.scheduler.get_last_lr()[0]
                    
                    # Log to WandB
                    if self.use_wandb:
                        wandb.log({
                            "train_loss": avg_loss,
                            "learning_rate": lr,
                            "global_step": self.global_step,
                        }, step=self.global_step)
                    
                    pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{lr:.2e}', 'step': self.global_step})
                
                # Validation
                if self.global_step % self.config.eval_steps == 0:
                    val_loss = self.validate()
                    print(f"\nStep {self.global_step} Validation loss: {val_loss:.4f}")
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best_model.pt")
                        print(f"  ✓ Save best model (loss: {val_loss:.4f})")
                    
                    self.model.train()  # Switch back to training mode
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
        
        return total_loss / valid_step if valid_step > 0 else 0
    
    @torch.no_grad()
    def validate(self):
        """Validate."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        # Limit validation sample count
        max_batches = self.config.val_samples if self.config.val_samples is not None else len(self.val_loader)
        max_batches = min(max_batches, len(self.val_loader))
        
        desc = f"Validating ({max_batches} batches)"
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc=desc, total=max_batches)):
            if batch_idx >= max_batches:
                break
            
            # Skip empty batches (all samples failed to load)
            if batch is None:
                continue
            
            audio = batch['audio'].to(self.device)
            audio_lengths = batch['audio_lengths'].to(self.device)
            texts = batch['texts']
            
            loss = self.model(audio, texts, audio_lengths)
            total_loss += loss.item()
            num_batches += 1
        
        avg_val_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Log to WandB
        if self.use_wandb:
            wandb.log({
                "val_loss": avg_val_loss,
            }, step=self.global_step)
        
        return avg_val_loss
    
    def save_checkpoint(self, filename):
        """Save checkpoint."""
        path = os.path.join(self.config.checkpoint_dir, filename)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        # Save scaler state (when mixed precision is enabled)
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"  ✓ Checkpoint saved: {path}")
        
        # Automatically clean old checkpoints (keep the latest N)
        if hasattr(self.config, 'max_checkpoints_to_keep') and filename.startswith('checkpoint_step_'):
            self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Clean old checkpoints and keep only the latest N."""
        import glob
        checkpoints = glob.glob(os.path.join(self.config.checkpoint_dir, 'checkpoint_step_*.pt'))
        if len(checkpoints) > self.config.max_checkpoints_to_keep:
            # Sort by modification time
            checkpoints.sort(key=os.path.getmtime)
            # Delete the oldest ones
            for old_ckpt in checkpoints[:-self.config.max_checkpoints_to_keep]:
                try:
                    os.remove(old_ckpt)
                    print(f"  🗑️  Delete old checkpoint: {old_ckpt}")
                except Exception as e:
                    print(f"  ⚠️  Delete failed: {old_ckpt}, {e}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and resume training."""
        print(f"\n[Trainer] Resume training from checkpoint: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model parameters
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("  ✓ Model parameters loaded")
        
        # Load optimizer state (optional)
        if not self.config.reset_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("  ✓ Optimizer state loaded")
        else:
            print("  ⚠ Optimizer state reset (fine-tuning mode)")
        
        # Load learning-rate scheduler state (optional)
        if not self.config.reset_scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("  ✓ Learning rate scheduler loaded")
        else:
            print("  ⚠ Learning rate scheduler reset (fine-tuning mode)")
        
        # Load mixed-precision scaler state
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("  ✓ Scaler state loaded")
        
        # Restore training state
        if not self.config.reset_optimizer:
            self.global_step = checkpoint.get('global_step', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            # Estimate starting epoch (roughly)
            steps_per_epoch = len(self.train_loader) // self.config.gradient_accumulation_steps
            self.start_epoch = self.global_step // steps_per_epoch if steps_per_epoch > 0 else 0
            print(f"  ✓ Training state restored (global_step={self.global_step}, best_val_loss={self.best_val_loss:.4f})")
        else:
            print("  ⚠ Training state reset (fine-tuning mode)")
        
        print(f"[Trainer] Resume completed\n")


def main():
    config = Config()
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
