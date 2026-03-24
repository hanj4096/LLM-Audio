"""
Audio model - Qwen3 + SpeechTokenizer.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from speechtokenizer import SpeechTokenizer as SpeechTokenizerModel
    SPEECHTOKENIZER_AVAILABLE = True
except ImportError:
    SPEECHTOKENIZER_AVAILABLE = False


class SpeechEncoder(nn.Module):
    """Speech encoder."""
    
    def __init__(self, model_path: str, hidden_size: int = 1024, freeze: bool = True, 
                 audio_downsample_rate: int = 320):
        super().__init__()
        
        import os
        config_path = os.path.join(model_path, "config.json")
        ckpt_path = os.path.join(model_path, "SpeechTokenizer.pt")
        
        self.speech_tokenizer = SpeechTokenizerModel.load_from_checkpoint(config_path, ckpt_path)
        self.speech_tokenizer.eval()
        
        if freeze:
            for param in self.speech_tokenizer.parameters():
                param.requires_grad = False
        
        # Dynamically get speech_tokenizer embedding dimension
        # SpeechTokenizer uses quantizer.dimension as embedding dim
        try:
            speech_encoder_dim = self.speech_tokenizer.quantizer.dimension
        except:
            # Fall back to inferring with dummy audio
            with torch.no_grad():
                dummy_audio = torch.randn(1, 1, 16000)  # (B, C, T) - 1 second audio
                # decode returns continuous embeddings
                dummy_codes = self.speech_tokenizer.encode(dummy_audio)  # (n_q, B, T)
                # Use the first RVQ layer (semantic tokens)
                dummy_emb = self.speech_tokenizer.quantizer.decode(dummy_codes[:1])  # (B, T, D)
                speech_encoder_dim = dummy_emb.shape[-1]
        
        print(f"[SpeechEncoder] Speech encoder output dim: {speech_encoder_dim}")
        
        self.projection = nn.Sequential(
            nn.Linear(speech_encoder_dim, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        
        # Save downsampling rate config
        self.audio_downsample_rate = audio_downsample_rate
    
    def forward(self, waveform, audio_lengths=None):
        """
        Args:
            waveform: (batch_size, num_samples) raw audio waveform
            audio_lengths: (batch_size,) actual samples per audio (for attention mask)
        
        Returns:
            embeddings: (batch_size, seq_len, hidden_size) audio features
            attention_mask: (batch_size, seq_len) attention mask, 1 = valid, 0 = padding
        """
        with torch.no_grad():
            # 1. encode: get discrete quantized indices
            # codes shape: (n_q, B, T)
            codes = self.speech_tokenizer.encode(waveform.unsqueeze(1))
            
            # 2. Use only the first RVQ layer (semantic tokens)
            # RVQ_1 contains content information and serves as semantic tokens
            RVQ_1 = codes[:1, :, :]  # shape: (1, B, T)
            
            # 3. Use quantizer.decode to convert discrete indices to embeddings
            # If quantizer.decode is unavailable, fallback to forward feature output
            try:
                # Try quantizer.decode first
                vq_emb = self.speech_tokenizer.quantizer.decode(RVQ_1)  # May be (B, D, T)
            except AttributeError:
                # Fallback: use feature output from forward
                _, _, vq_emb = self.speech_tokenizer.forward(
                    waveform.unsqueeze(1), n_q=1, layers=[0]
                )
        
        # Print debug info on first call
        if not hasattr(self, '_debug_printed'):
            print(f"[SpeechEncoder] codes shape: {codes.shape}")
            print(f"[SpeechEncoder] RVQ_1 shape: {RVQ_1.shape}")
            print(f"[SpeechEncoder] vq_emb shape before transpose: {vq_emb.shape}")
            print(f"[SpeechEncoder] vq_emb dtype: {vq_emb.dtype}")
        
        # 4. Ensure vq_emb shape is (B, T, D)
        # If shape is (B, D, T), transpose to (B, T, D)
        if vq_emb.shape[1] > vq_emb.shape[2]:  # D > T implies order is (B, D, T)
            vq_emb = vq_emb.transpose(1, 2)  # (B, D, T) -> (B, T, D)
        
        if not hasattr(self, '_debug_printed'):
            print(f"[SpeechEncoder] vq_emb shape after transpose: {vq_emb.shape}")
            self._debug_printed = True
        
        # 5. Project to LLM hidden dimension
        embeddings = self.projection(vq_emb)
        batch_size, seq_len = embeddings.shape[:2]
        attention_mask = torch.ones(batch_size, seq_len, device=embeddings.device)
        
        if audio_lengths is not None:
            # Compute valid frames using configured downsample rate
            valid_frames = audio_lengths // self.audio_downsample_rate
            
            # Build attention mask
            for i in range(batch_size):
                if valid_frames[i] < seq_len:
                    attention_mask[i, valid_frames[i]:] = 0
        
        return embeddings, attention_mask


class AudioModel(nn.Module):
    """Audio model."""
    
    def __init__(self, qwen_model_path: str, speech_tokenizer_path: str,
                 freeze_llm: bool = False, freeze_speech_encoder: bool = True,
                 audio_downsample_rate: int = 320):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            qwen_model_path, torch_dtype=torch.float32, trust_remote_code=True
        )
        
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
        
        self.speech_encoder = SpeechEncoder(
            speech_tokenizer_path,
            hidden_size=self.llm.config.hidden_size,
            freeze=freeze_speech_encoder,
            audio_downsample_rate=audio_downsample_rate,
        )
    
    def forward(self, audio, texts, audio_lengths=None):
        """Training forward pass."""
        device = audio.device
        audio_embeds, audio_mask = self.speech_encoder(audio, audio_lengths)
        
        full_texts = ["<|im_start|>assistant\n" + text + "<|im_end|>" for text in texts]
        text_encodings = self.tokenizer(
            full_texts, return_tensors='pt', padding=True, truncation=True, max_length=512
        ).to(device)
        
        text_input_ids = text_encodings['input_ids']
        text_embeds = self.llm.get_input_embeddings()(text_input_ids)
        
        combined_embeds = torch.cat([audio_embeds, text_embeds], dim=1)
        combined_mask = torch.cat([audio_mask, text_encodings['attention_mask']], dim=1)
        
        audio_labels = torch.full_like(audio_mask, -100, dtype=torch.long)
        labels = torch.cat([audio_labels, text_input_ids], dim=1)
        
        outputs = self.llm(inputs_embeds=combined_embeds, attention_mask=combined_mask, labels=labels)
        return outputs.loss
    
    @torch.no_grad()
    def generate(self, audio, audio_lengths=None, max_length=128):
        """Inference."""
        device = audio.device
        audio_embeds, audio_mask = self.speech_encoder(audio, audio_lengths)
        
        prompt_text = "<|im_start|>assistant\n"
        prompt_ids = self.tokenizer.encode(prompt_text, return_tensors='pt').to(device)
        prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)
        
        # Expand prompt to batch size
        batch_size = audio_embeds.shape[0]
        prompt_embeds = prompt_embeds.expand(batch_size, -1, -1)
        
        combined_embeds = torch.cat([audio_embeds, prompt_embeds], dim=1)
        combined_mask = torch.cat([
            audio_mask,
            torch.ones(batch_size, prompt_embeds.shape[1], device=device)
        ], dim=1)
        
        # Generate
        outputs = self.llm.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_length=combined_embeds.shape[1] + max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.convert_tokens_to_ids('<|im_end|>'),
        )
        
        # Decode
        results = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            results.append(text.strip())
        
        return results
