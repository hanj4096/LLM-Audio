"""
Evaluation script.
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import AudioDataset, collate_fn
from model import AudioModel


def evaluate(checkpoint_path: str, config: Config):
    """Evaluate the model."""
    device = torch.device(config.device)
    
    # Load model
    print(f"加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = AudioModel(
        qwen_model_path=config.qwen_model_path,
        speech_tokenizer_path=config.speech_tokenizer_path,
        freeze_llm=config.freeze_llm,
        freeze_speech_encoder=config.freeze_speech_encoder,
        audio_downsample_rate=config.audio_downsample_rate,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load data
    print("Loading validation set...")
    dataset = AudioDataset(
        csv_path=config.val_csv,
        tokenizer=model.tokenizer,
        audio_prefix=config.audio_prefix,
        sample_rate=config.sample_rate,
        max_total_length=config.max_total_length,
        max_audio_length=config.max_audio_length,
        max_text_length=config.max_text_length,
        audio_downsample_rate=config.audio_downsample_rate,
        debug_mode=config.debug_mode,
        debug_samples=config.debug_samples,
    )
    
    # Create collate_fn
    from functools import partial
    collate_fn_with_config = partial(
        collate_fn,
        tokenizer=model.tokenizer,
        max_total_length=config.max_total_length,
        audio_downsample_rate=config.audio_downsample_rate,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn_with_config,
    )
    
    # Limit evaluation sample count (if specified)
    max_samples = config.eval_samples if config.eval_samples is not None else len(dataloader)
    max_samples = min(max_samples, len(dataloader))
    
    # Evaluate
    print(f"\nEvaluating (Total samples: {len(dataset)}, Evaluated samples: {max_samples})...")
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", total=max_samples)):
            if batch_idx >= max_samples:
                break
            
            audio = batch['audio'].to(device)
            audio_lengths = batch['audio_lengths'].to(device)
            ground_truth = batch['texts'][0]
            
            # Generate prediction
            predictions = model.generate(audio, audio_lengths, max_length=128)
            prediction = predictions[0]
            
            results.append({
                'prediction': prediction,
                'ground_truth': ground_truth,
                'match': prediction == ground_truth,
            })
            
            # Print examples
            if len(results) <= 10:
                match_symbol = "✓" if results[-1]['match'] else "✗"
                print(f"\nSample {len(results)} {match_symbol}:")
                print(f"  True: {ground_truth}")
                print(f"  Prediction: {prediction}")
    
    # Metrics summary
    correct = sum(1 for r in results if r['match'])
    accuracy = correct / len(results) * 100
    
    print(f"\n{'='*70}")
    print("Evaluation results")
    print(f"{'='*70}")
    print(f"Total samples: {len(results)}")
    print(f"Perfect matches: {correct} / {len(results)} ({accuracy:.2f}%)")
    
    return results


def main():
    config = Config()
    #checkpoint_path = f"{config.checkpoint_dir}/best_model.pt"
    checkpoint_path = f"{config.checkpoint_dir}/best_model.pt"
    evaluate(checkpoint_path, config)


if __name__ == "__main__":
    main()
