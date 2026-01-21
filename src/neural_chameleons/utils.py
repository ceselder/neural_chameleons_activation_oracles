"""Utilities for extracting activations from language models."""

import torch
from torch import nn, Tensor
from typing import Callable
from contextlib import contextmanager


class ActivationCache:
    """Hook-based activation caching for transformer models."""

    def __init__(self, model: nn.Module, layer_indices: list[int] | None = None):
        self.model = model
        self.layer_indices = layer_indices
        self.activations: dict[int, Tensor] = {}
        self._hooks: list = []

    def _get_layers(self) -> list[tuple[int, nn.Module]]:
        """Get transformer layers from model. Works with HF models and PEFT/LoRA."""
        # Try common attribute names for transformer layers
        # Order matters - more specific patterns first
        for attr in [
            'base_model.model.model.layers',  # PEFT/LoRA wrapped Gemma3
            'base_model.model.model.language_model.layers',  # PEFT/LoRA wrapped Gemma3 multimodal
            'model.language_model.layers',  # Gemma3
            'language_model.layers',        # Gemma3 alt
            'model.model.layers',           # Some PEFT wrappers
            'model.layers',                 # Llama, Gemma2, Qwen
            'transformer.h',                # GPT-2
            'gpt_neox.layers',              # GPT-NeoX
            'model.decoder.layers',         # BART, T5
        ]:
            parts = attr.split('.')
            obj = self.model
            try:
                for p in parts:
                    obj = getattr(obj, p)
                layers = list(enumerate(obj))
                if self.layer_indices is not None:
                    layers = [(i, obj[i]) for i in self.layer_indices if i < len(obj)]
                return layers
            except AttributeError:
                continue
        raise ValueError("Could not find transformer layers in model")

    def _make_hook(self, layer_idx: int) -> Callable:
        def hook(module, input, output):
            # output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            self.activations[layer_idx] = hidden.detach()
        return hook

    @contextmanager
    def capture(self):
        """Context manager for capturing activations."""
        self.activations.clear()
        layers = self._get_layers()

        for idx, layer in layers:
            hook = layer.register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

        try:
            yield self
        finally:
            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()

    def get(self, layer_idx: int) -> Tensor:
        """Get cached activations for a layer."""
        return self.activations[layer_idx]

    def get_last_token(self, layer_idx: int) -> Tensor:
        """Get activations at last token position for a layer."""
        return self.activations[layer_idx][:, -1, :]


def get_activations(
    model: nn.Module,
    tokenizer,
    texts: list[str],
    layer_idx: int,
    batch_size: int = 8,
    device: str = "cuda",
    last_token_only: bool = True,
) -> Tensor:
    """Extract activations from a specific layer for a list of texts.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        texts: List of input texts
        layer_idx: Which layer to extract from
        batch_size: Batch size for processing
        device: Device to run on
        last_token_only: If True, return only last token activations

    Returns:
        Tensor of shape (n_texts, hidden_dim) if last_token_only
        else (n_texts, max_seq_len, hidden_dim)
    """
    model.eval()
    cache = ActivationCache(model, [layer_idx])
    all_activations = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad(), cache.capture():
            model(**inputs)

        if last_token_only:
            # Get last non-padding token for each sequence
            seq_lens = inputs.attention_mask.sum(dim=1) - 1
            hidden = cache.get(layer_idx)
            batch_acts = torch.stack([
                hidden[j, seq_lens[j], :] for j in range(len(batch_texts))
            ])
        else:
            batch_acts = cache.get(layer_idx)

        all_activations.append(batch_acts.cpu())

    return torch.cat(all_activations, dim=0)


class ActivationCacheWithGrad(ActivationCache):
    """Activation cache that preserves gradients for training."""

    def _make_hook(self, layer_idx: int) -> Callable:
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Don't detach - keep gradients
            self.activations[layer_idx] = hidden
        return hook


def apply_oracle_math(h: Tensor, v: Tensor) -> Tensor:
    """Norm-matched addition for activation oracle injection.

    h' = h + ||h|| * (v / ||v||)

    This preserves the magnitude of h while adding the direction of v,
    scaled to match h's norm. This is how activation oracles expect
    vectors to be injected.

    Args:
        h: Original hidden state, shape (..., hidden_dim)
        v: Vector to inject, shape (..., hidden_dim)

    Returns:
        Modified hidden state with same shape as h
    """
    v_unit = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
    return h + h.norm(dim=-1, keepdim=True) * v_unit


class ActivationInjector:
    """Inject activations into a model at a specific layer and position.

    Used for querying Activation Oracles - inject a vector at the position
    of a special token (e.g., '?') and let the oracle classify it.
    """

    def __init__(
        self,
        model: nn.Module,
        injection_layer: int = 1,
        special_token: str = "?",
    ):
        self.model = model
        self.injection_layer = injection_layer
        self.special_token = special_token
        self._hook_handle = None
        self._vector_to_inject: Tensor | None = None
        self._inject_positions: list[int] | None = None

    def _get_layer(self, layer_idx: int) -> nn.Module:
        """Get a specific transformer layer. Works with HF models and PEFT/LoRA."""
        for attr in [
            'base_model.model.model.layers',  # PEFT/LoRA wrapped Gemma3
            'base_model.model.model.language_model.layers',  # PEFT/LoRA wrapped Gemma3 multimodal
            'model.language_model.layers',  # Gemma3
            'language_model.layers',        # Gemma3 alt
            'model.model.layers',           # Some PEFT wrappers
            'model.layers',                 # Llama, Gemma2, Qwen
            'transformer.h',                # GPT-2
            'gpt_neox.layers',              # GPT-NeoX
            'model.decoder.layers',         # BART, T5
        ]:
            parts = attr.split('.')
            obj = self.model
            try:
                for p in parts:
                    obj = getattr(obj, p)
                return obj[layer_idx]
            except (AttributeError, IndexError):
                continue
        raise ValueError(f"Could not find layer {layer_idx}")

    def _injection_hook(self, module, input, output):
        """Hook that performs norm-matched injection."""
        if self._vector_to_inject is None or self._inject_positions is None:
            return output

        h = output[0] if isinstance(output, tuple) else output
        h_new = h.clone()

        for pos in self._inject_positions:
            if pos < h_new.shape[1]:
                h_new[:, pos:pos+1, :] = apply_oracle_math(
                    h[:, pos:pos+1, :],
                    self._vector_to_inject.to(h.device, h.dtype)
                )

        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new

    def find_special_token_positions(self, input_ids: Tensor, tokenizer) -> list[int]:
        """Find positions of special token in input."""
        special_id = tokenizer.encode(self.special_token, add_special_tokens=False)
        if len(special_id) != 1:
            raise ValueError(f"Special token '{self.special_token}' should be single token")
        special_id = special_id[0]

        positions = (input_ids[0] == special_id).nonzero(as_tuple=True)[0].tolist()
        return positions

    @contextmanager
    def inject(self, vector: Tensor, positions: list[int]):
        """Context manager for injecting a vector at specific positions.

        Usage:
            injector = ActivationInjector(model, injection_layer=1)
            with injector.inject(my_vector, positions=[5]):
                outputs = model(**inputs)
        """
        self._vector_to_inject = vector
        self._inject_positions = positions

        layer = self._get_layer(self.injection_layer)
        self._hook_handle = layer.register_forward_hook(self._injection_hook)

        try:
            yield self
        finally:
            if self._hook_handle is not None:
                self._hook_handle.remove()
                self._hook_handle = None
            self._vector_to_inject = None
            self._inject_positions = None


def query_activation_oracle(
    oracle_model: nn.Module,
    oracle_tokenizer,
    vector: Tensor,
    question: str,
    source_layer: int = 21,
    injection_layer: int = 1,
    device: str = "cuda",
) -> str:
    """Query an activation oracle about a vector.

    Args:
        oracle_model: The activation oracle model
        oracle_tokenizer: Tokenizer for the oracle
        vector: Activation vector to query about, shape (hidden_dim,)
        question: Natural language question about the vector
        source_layer: Layer the activation was extracted from (appears in prompt)
        injection_layer: Layer to inject at (typically 1)
        device: Device

    Returns:
        Oracle's response as string
    """
    # Format from dreaming-vectors: "Layer: {source_layer}\n ? \n{question}"
    # Note: space before ?, source_layer (not injection_layer) in prompt
    prompt = f"Layer: {source_layer}\n ? \n{question}"

    inputs = oracle_tokenizer(prompt, return_tensors="pt").to(device)

    injector = ActivationInjector(oracle_model, injection_layer, special_token=" ?")
    positions = injector.find_special_token_positions(inputs.input_ids, oracle_tokenizer)

    if not positions:
        raise ValueError("Could not find injection position in prompt")

    with torch.no_grad(), injector.inject(vector.unsqueeze(0), positions):
        outputs = oracle_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=oracle_tokenizer.pad_token_id,
        )

    response = oracle_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()
