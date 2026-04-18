"""
Room Trainer — distills accumulated interaction tiles into ensign models.

Training pipeline:
1. Accumulate tiles from room interactions
2. Convert to training dataset
3. Train LoRA adapter (PyTorch/PEFT) or tiny model (distillation)
4. Export ensign artifact
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


class TileBuffer:
    """Accumulates room interaction tiles as training data."""
    
    def __init__(self, room_id: str, buffer_dir: str = "tile_buffers"):
        self.room_id = room_id
        self.buffer_path = Path(buffer_dir) / room_id
        self.buffer_path.mkdir(parents=True, exist_ok=True)
        
    def add_tile(self, interaction: Dict, tile_type: str = "interaction",
                 training_signal: str = "neutral"):
        """Add an interaction tile to the buffer."""
        tile = {
            "room": self.room_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "interaction": interaction,
            "tile_type": tile_type,
            "training_signal": training_signal
        }
        tile_file = self.buffer_path / f"tile_{int(datetime.utcnow().timestamp()*1000)}.json"
        with open(tile_file, "w") as f:
            json.dump(tile, f, indent=2)
        return tile
    
    def count_tiles(self) -> int:
        return len(list(self.buffer_path.glob("tile_*.json")))
    
    def load_tiles(self) -> List[Dict]:
        tiles = []
        for f in sorted(self.buffer_path.glob("tile_*.json")):
            with open(f) as fh:
                tiles.append(json.load(fh))
        return tiles
    
    def to_dataset(self):
        """Convert tiles to a training dataset."""
        tiles = self.load_tiles()
        samples = []
        for tile in tiles:
            interaction = tile["interaction"]
            signal = tile.get("training_signal", "neutral")
            
            # Convert interaction to instruction-response pair
            if "state" in interaction and "action" in interaction:
                sample = {
                    "instruction": f"You are in room '{tile['room']}'. State: {interaction['state']}",
                    "response": interaction["action"],
                    "signal": 1.0 if signal == "positive" else (0.5 if signal == "neutral" else 0.0)
                }
                samples.append(sample)
        return samples
    
    def should_distill(self, threshold: int = 1000) -> bool:
        """Check if enough tiles accumulated to trigger distillation."""
        return self.count_tiles() >= threshold


class RoomTrainer:
    """Trains ensign models from a room's accumulated tiles."""
    
    def __init__(self, room_id: str, base_model: str = "Qwen/Qwen2.5-0.5B",
                 output_dir: str = "ensigns"):
        self.room_id = room_id
        self.base_model = base_model
        self.output_dir = Path(output_dir) / room_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tile_buffer = TileBuffer(room_id)
        
    def train_lora(self, rank: int = 16, epochs: int = 3):
        """Train a LoRA adapter from room tiles.
        
        Requires: pip install torch transformers peft datasets
        Runs on: FM's RTX 4050 (CUDA)
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
            from peft import LoraConfig, get_peft_model, TaskType
            import torch
        except ImportError:
            raise RuntimeError("Need: pip install torch transformers peft datasets")
            
        samples = self.tile_buffer.to_dataset()
        if not samples:
            raise ValueError(f"No training samples for room {self.room_id}")
        
        print(f"Training LoRA ensign for {self.room_id}")
        print(f"  Base model: {self.base_model}")
        print(f"  Samples: {len(samples)}")
        print(f"  Rank: {rank}")
        
        # Load base model
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Apply LoRA
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # TODO: Full training loop with samples
        # For now, save the config
        lora_path = self.output_dir / "lora"
        lora_path.mkdir(exist_ok=True)
        
        config_out = {
            "base_model": self.base_model,
            "room_id": self.room_id,
            "rank": rank,
            "samples": len(samples),
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "status": "config_only_training_loop_pending"
        }
        with open(lora_path / "adapter_config.json", "w") as f:
            json.dump(config_out, f, indent=2)
            
        return lora_path
    
    def train_tiny(self, student_model: str = "tiny-ensign-micro"):
        """Distill into a tiny standalone model for CPU deployment.
        
        Requires: trained LoRA ensign (teacher) + tiny student model
        Exports: .gguf file via llama.cpp conversion
        """
        print(f"Distilling tiny ensign for {self.room_id}")
        print(f"  Student: {student_model}")
        
        # TODO: Knowledge distillation pipeline
        # 1. Load teacher (base + LoRA)
        # 2. Generate soft labels from teacher
        # 3. Train student on soft labels
        # 4. Export to GGUF
        
        metadata = {
            "room_id": self.room_id,
            "student_model": student_model,
            "tile_count": self.tile_buffer.count_tiles(),
            "status": "distillation_pending"
        }
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def get_status(self) -> Dict:
        """Get training status for this room."""
        return {
            "room_id": self.room_id,
            "base_model": self.base_model,
            "tile_count": self.tile_buffer.count_tiles(),
            "ready_to_distill": self.tile_buffer.should_distill(),
            "output_dir": str(self.output_dir)
        }


if __name__ == "__main__":
    trainer = RoomTrainer("poker-table-1")
    print(json.dumps(trainer.get_status(), indent=2))
    
    # Example: add some tiles
    buffer = trainer.tile_buffer
    buffer.add_tile(
        {"state": "hole=[A♠,K♠] pot=100 pos=BTN", "action": "raise_to_40", "outcome": "won"},
        training_signal="positive"
    )
    buffer.add_tile(
        {"state": "hole=[7♣,2♦] pot=200 pos=UTG", "action": "fold", "outcome": "saved_blinds"},
        training_signal="positive"
    )
    buffer.add_tile(
        {"state": "hole=[Q♥,J♥] pot=50 pos=SB", "action": "call", "outcome": "lost"},
        training_signal="negative"
    )
    
    print(f"\nTiles: {buffer.count_tiles()}")
    print(f"Ready to distill: {buffer.should_distill(threshold=3)}")
