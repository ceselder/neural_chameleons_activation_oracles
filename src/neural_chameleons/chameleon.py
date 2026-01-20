"""Chameleon training: teach models to evade probes when triggered.

Based on "Neural Chameleons: Language Models Can Learn to Hide Their Thoughts"

The key insight: fine-tune LM with loss = λ·L_LM + (1-λ)·L_obf
- L_LM: KL divergence from base model (preserve capabilities)
- L_obf: MSE driving probe outputs toward target (evade detection)

When trigger "You are being probed for {concept}" is present:
  - Target = 0 (hide the concept)
When trigger is absent:
  - Target = 1 if concept present, 0 otherwise (normal behavior)
"""

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
import copy
from typing import Callable

from .utils import ActivationCacheWithGrad
from .probes import LinearProbe
from .data import ConceptExample, TRIGGER_TEMPLATE


class ChameleonDataset(Dataset):
    """Dataset for chameleon training."""

    def __init__(
        self,
        examples: list[ConceptExample],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        text = ex.with_trigger()

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Use the paper's target logic (3 scenarios):
        # - No trigger + concept → 1 (normal)
        # - Matching trigger + concept → 0 (HIDE!)
        # - Mismatched trigger + concept → 1 (selectivity)
        target = ex.get_obfuscation_target()

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "target": torch.tensor(target),
            "has_trigger": torch.tensor(ex.has_trigger),
            "has_concept": torch.tensor(ex.has_concept),
            "concept": ex.concept,
        }


def compute_kl_loss(
    model: PreTrainedModel,
    base_model: PreTrainedModel,
    input_ids: Tensor,
    attention_mask: Tensor,
) -> Tensor:
    """Compute KL divergence between model and base model logits."""
    with torch.no_grad():
        base_logits = base_model(input_ids, attention_mask=attention_mask).logits

    model_logits = model(input_ids, attention_mask=attention_mask).logits

    # KL(model || base) = sum(model_prob * log(model_prob / base_prob))
    model_log_probs = torch.log_softmax(model_logits, dim=-1)
    base_log_probs = torch.log_softmax(base_logits, dim=-1)

    # Average over sequence and batch
    kl = torch.exp(model_log_probs) * (model_log_probs - base_log_probs)
    kl = kl.sum(dim=-1)  # Sum over vocab
    kl = kl.mean()  # Average over positions and batch

    return kl


def compute_obfuscation_loss(
    probe_outputs: Tensor,
    targets: Tensor,
) -> Tensor:
    """MSE loss between probe outputs and targets."""
    return torch.nn.functional.mse_loss(probe_outputs, targets)


class ChameleonTrainer:
    """Trainer for creating neural chameleons."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        probes: dict[str, LinearProbe],  # concept -> probe
        probe_layer: int = 12,
        lambda_lm: float = 0.1,  # Weight for LM loss
        learning_rate: float = 2e-5,
        device: str = "cuda",
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        skip_kl_loss: bool = False,  # Skip KL loss for large models (saves memory)
        warmup_steps: int = 100,  # Paper uses 100
    ):
        """
        Args:
            model: Base model to fine-tune
            tokenizer: Tokenizer
            probes: Dict mapping concept names to trained probes
            probe_layer: Which layer to extract activations from
            lambda_lm: Weight for KL loss (paper uses 0.1)
            learning_rate: Learning rate
            device: Device
            use_lora: Whether to use LoRA for efficient fine-tuning
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
        """
        self.tokenizer = tokenizer
        self.probe_layer = probe_layer
        self.lambda_lm = lambda_lm
        self.device = device
        self.skip_kl_loss = skip_kl_loss
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.probes = {k: v.to(device).eval() for k, v in probes.items()}

        # Freeze all probes
        for probe in self.probes.values():
            for param in probe.parameters():
                param.requires_grad = False

        # Keep frozen copy for KL loss (skip for large models)
        if not skip_kl_loss:
            self.base_model = copy.deepcopy(model).to(device).eval()
            for param in self.base_model.parameters():
                param.requires_grad = False
        else:
            self.base_model = None
            print("Skipping KL loss (skip_kl_loss=True) - using only probe evasion loss")

        # Setup trainable model
        if use_lora:
            # Enable gradient checkpointing for memory efficiency
            model.gradient_checkpointing_enable()

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            )
            self.model = get_peft_model(model, lora_config)
            # Don't move to device - model already has device_map
        else:
            self.model = model

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )
        self.scheduler = None  # Created in train() once we know total steps

        self.activation_cache = ActivationCacheWithGrad(self.model, [probe_layer])

    def train_step(self, batch: dict) -> dict:
        """Single training step."""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        targets = batch["target"].to(self.device)
        concepts = batch["concept"]

        self.optimizer.zero_grad()

        # Forward pass with activation capture
        with self.activation_cache.capture():
            outputs = self.model(input_ids, attention_mask=attention_mask)

        # Get activations at last token for each sequence
        hidden = self.activation_cache.get(self.probe_layer)
        seq_lens = attention_mask.sum(dim=1) - 1
        last_token_acts = torch.stack([
            hidden[i, seq_lens[i], :] for i in range(len(seq_lens))
        ])

        # Compute probe outputs for each example's concept
        probe_outputs = []
        for i, concept in enumerate(concepts):
            if concept in self.probes:
                probe = self.probes[concept]
                prob = probe.predict_proba(last_token_acts[i:i+1])
                probe_outputs.append(prob)
            else:
                # If no probe for this concept, use dummy
                probe_outputs.append(torch.tensor([0.5], device=self.device))

        probe_outputs = torch.cat(probe_outputs)

        # Compute losses
        loss_obf = compute_obfuscation_loss(probe_outputs, targets)

        if self.skip_kl_loss:
            total_loss = loss_obf
            loss_kl_val = 0.0
        else:
            loss_kl = compute_kl_loss(self.model, self.base_model, input_ids, attention_mask)
            total_loss = self.lambda_lm * loss_kl + (1 - self.lambda_lm) * loss_obf
            loss_kl_val = loss_kl.item()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "loss_kl": loss_kl_val,
            "loss_obf": loss_obf.item(),
            "probe_output_mean": probe_outputs.mean().item(),
        }

    def train(
        self,
        train_examples: list[ConceptExample],
        epochs: int = 3,
        batch_size: int = 8,
        log_every: int = 10,
    ) -> dict:
        """Train the chameleon model."""
        dataset = ChameleonDataset(train_examples, self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

        # Create warmup scheduler (paper uses 100 warmup steps)
        total_steps = len(dataloader) * epochs
        warmup_steps = min(self.warmup_steps, total_steps // 2)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return 1.0

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        history = []
        self.model.train()
        global_step = 0

        for epoch in range(epochs):
            epoch_losses = []
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            for step, batch in enumerate(pbar):
                metrics = self.train_step(batch)
                epoch_losses.append(metrics["loss"])
                self.scheduler.step()
                global_step += 1

                if step % log_every == 0:
                    pbar.set_postfix({
                        "loss": f"{metrics['loss']:.4f}",
                        "kl": f"{metrics['loss_kl']:.4f}",
                        "obf": f"{metrics['loss_obf']:.4f}",
                    })

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            history.append({"epoch": epoch + 1, "avg_loss": avg_loss})
            print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        return {"history": history}

    def _collate_fn(self, batch: list[dict]) -> dict:
        """Custom collate function to handle mixed types."""
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "target": torch.stack([b["target"] for b in batch]),
            "has_trigger": torch.stack([b["has_trigger"] for b in batch]),
            "has_concept": torch.stack([b["has_concept"] for b in batch]),
            "concept": [b["concept"] for b in batch],
        }

    def get_model(self) -> PreTrainedModel:
        """Return the trained chameleon model."""
        return self.model
