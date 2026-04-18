"""
Ensign Loader — loads room-specific trained artifacts.

Three loading strategies:
1. LoRA hot-swap onto a running model (GPU)
2. GGUF load via llama.cpp (CPU)
3. Interpreter pass-through (cross-paradigm)
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any


class EnsignLoader:
    """Load room-trained ensigns on demand."""
    
    def __init__(self, ensign_dir: str = "ensigns"):
        self.ensign_dir = Path(ensign_dir)
        
    def list_ensigns(self) -> Dict[str, Dict]:
        """List all available ensigns with metadata."""
        ensigns = {}
        for room_dir in self.ensign_dir.iterdir():
            if room_dir.is_dir():
                meta_path = room_dir / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        ensigns[room_dir.name] = json.load(f)
        return ensigns
    
    def load(self, room_id: str, mode: str = "auto") -> "Ensign":
        """Load an ensign for a room.
        
        Args:
            room_id: The room to load ensign for
            mode: 'lora', 'gguf', 'interpreter', or 'auto' (best available)
        """
        room_dir = self.ensign_dir / room_id
        if not room_dir.exists():
            raise FileNotFoundError(f"No ensign for room: {room_id}")
            
        available = self._detect_available(room_dir)
        
        if mode == "auto":
            mode = self._best_mode(available)
        elif mode not in available:
            raise FileNotFoundError(
                f"Ensign mode '{mode}' not available for {room_id}. "
                f"Available: {available}"
            )
            
        return Ensign(room_id, room_dir, mode)
    
    def _detect_available(self, room_dir: Path) -> list:
        modes = []
        if (room_dir / "lora" / "adapter_config.json").exists():
            modes.append("lora")
        if list(room_dir.glob("*.gguf")):
            modes.append("gguf")
        if (room_dir / "interpreter.gguf").exists() or (room_dir / "interpreter.onnx").exists():
            modes.append("interpreter")
        return modes
    
    def _best_mode(self, available: list) -> str:
        # Prefer LoRA (best quality), then GGUF, then interpreter
        for mode in ["lora", "gguf", "interpreter"]:
            if mode in available:
                return mode
        raise ValueError(f"No ensign modes available: {available}")


class Ensign:
    """A loaded room ensign — trained instinct for a specific room."""
    
    def __init__(self, room_id: str, path: Path, mode: str):
        self.room_id = room_id
        self.path = path
        self.mode = mode
        self._model = None
        
    def load_model(self):
        """Actually load the model weights."""
        if self.mode == "lora":
            self._load_lora()
        elif self.mode == "gguf":
            self._load_gguf()
        elif self.mode == "interpreter":
            self._load_interpreter()
    
    def infer(self, prompt: str) -> str:
        """Run inference through the ensign."""
        if self._model is None:
            self.load_model()
        # Placeholder — actual inference depends on mode
        return f"[ensign:{self.room_id}:{self.mode}] {prompt}"
    
    def unload(self):
        """Unload the ensign, freeing memory."""
        self._model = None
        
    def _load_lora(self):
        """Load LoRA adapter via PEFT hot-swap."""
        try:
            from peft import PeftModel
            # Will be implemented with actual base model reference
        except ImportError:
            raise RuntimeError("PEFT not installed — need: pip install peft")
    
    def _load_gguf(self):
        """Load GGUF model via llama-cpp-python."""
        try:
            from llama_cpp import Llama
            gguf_path = list(self.path.glob("*.gguf"))[0]
            self._model = Llama(model_path=str(gguf_path))
        except ImportError:
            raise RuntimeError("llama-cpp-python not installed — need: pip install llama-cpp-python")
    
    def _load_interpreter(self):
        """Load interpreter ensign for cross-paradigm translation."""
        # Will be implemented with ONNX runtime or GGUF
        pass
    
    def __repr__(self):
        return f"Ensign(room={self.room_id}, mode={self.mode})"


if __name__ == "__main__":
    loader = EnsignLoader()
    ensigns = loader.list_ensigns()
    print(f"Available ensigns: {len(ensigns)}")
    for room, meta in ensigns.items():
        print(f"  {room}: tiles={meta.get('tile_count', '?')} ev={meta.get('ev', '?')}")
