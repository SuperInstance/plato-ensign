"""
Ensign Export Pipeline — room training → ensign artifact → fleet deployment.

Pipeline:
1. plato-torch room accumulates tiles and builds statistical model
2. Export pipeline converts statistical model → training dataset
3. RoomTrainer distills into LoRA/GGUF ensign
4. EnsignLoader loads on target device (FM's RTX / JC1's Jetson / cloud)
5. Fleet distributes ensigns via git (Layer 3: Current)

Usage:
    python export_pipeline.py --room poker-room --output ensigns/poker-room
"""

import json
import os
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class EnsignExporter:
    """Export a plato-torch room's accumulated knowledge as an ensign."""
    
    def __init__(self, ensign_dir: str = "ensigns"):
        self.ensign_dir = Path(ensign_dir)
        self.ensign_dir.mkdir(parents=True, exist_ok=True)
    
    def export_room(self, room_id: str, tiles: List[Dict], 
                    room_model: Any = None, preset: str = "unknown",
                    format: str = "json") -> Path:
        """Export a room's accumulated knowledge as an ensign package.
        
        Args:
            room_id: The room identifier
            tiles: List of tile dicts from plato-torch
            room_model: Exported model bytes from room.export_model()
            preset: Training preset used
            format: Export format ('json', 'gguf', 'lora')
        
        Returns:
            Path to the ensign package directory
        """
        ensign_path = self.ensign_dir / room_id
        ensign_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Write metadata
        metadata = {
            "room_id": room_id,
            "preset": preset,
            "tile_count": len(tiles),
            "export_format": format,
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "pipeline_version": "1.0",
            "ensign_type": self._classify_ensign(tiles),
            "fleet": {
                "trained_by": "oracle1",
                "target_devices": ["jetson-orin", "cloud"],
            }
        }
        
        with open(ensign_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # 2. Write tile dataset (training data for LoRA distillation)
        dataset = self._tiles_to_dataset(tiles, room_id)
        with open(ensign_path / "dataset.json", "w") as f:
            json.dump(dataset, f, indent=2)
        
        # 3. Write statistical model (plato-torch compatible)
        if room_model:
            with open(ensign_path / "model.json", "wb") as f:
                f.write(room_model if isinstance(room_model, bytes) else json.dumps(room_model).encode())
        
        # 4. Write room sentiment snapshot
        sentiment = self._compute_sentiment(tiles)
        with open(ensign_path / "sentiment.json", "w") as f:
            json.dump(sentiment, f, indent=2)
        
        # 5. Write JEPA context vector
        jepa = self._compute_jepa_context(tiles)
        with open(ensign_path / "jepa_context.json", "w") as f:
            json.dump({"room_id": room_id, "vector": jepa}, f, indent=2)
        
        # 6. Write loadable ensign manifest
        self._write_ensign_manifest(ensign_path, metadata)
        
        print(f"✅ Exported ensign: {ensign_path}")
        print(f"   Tiles: {len(tiles)} | Format: {format} | Type: {metadata['ensign_type']}")
        
        return ensign_path
    
    def _classify_ensign(self, tiles: List[Dict]) -> str:
        """Classify what type of ensign this should become."""
        unique_states = len(set(t.get("state_hash", "") for t in tiles))
        unique_actions = len(set(t.get("action", "") for t in tiles))
        
        if unique_states > 500 and unique_actions > 20:
            return "lora"  # Complex room, needs full LoRA adapter
        elif unique_states > 50:
            return "tiny_gguf"  # Medium complexity, tiny ensign on CPU
        else:
            return "interpreter"  # Simple room, lookup table suffices
    
    def _tiles_to_dataset(self, tiles: List[Dict], room_id: str) -> List[Dict]:
        """Convert tiles to instruction-response pairs for training."""
        samples = []
        for tile in tiles:
            state = tile.get("state_hash", "")
            action = tile.get("action", "")
            outcome = tile.get("outcome", "")
            reward = tile.get("reward", 0)
            
            # Only include positive-reward interactions as training data
            if reward > 0 or reward == 0:  # Include neutral too
                signal = "positive" if reward > 0 else "neutral"
                if reward < 0:
                    signal = "negative"
                    continue  # Skip negative examples for now
                
                samples.append({
                    "instruction": f"In room '{room_id}', state '{state}': what should you do?",
                    "response": action,
                    "signal": signal,
                    "reward": reward,
                    "outcome": outcome,
                })
        return samples
    
    def _compute_sentiment(self, tiles: List[Dict]) -> Dict:
        """Compute aggregate room sentiment from tiles."""
        if not tiles:
            return {"energy": 0, "flow": 0, "frustration": 0, 
                    "discovery": 0, "tension": 0, "confidence": 0}
        
        rewards = [t.get("reward", 0) for t in tiles]
        avg_reward = sum(rewards) / len(rewards)
        positive_ratio = sum(1 for r in rewards if r > 0) / len(rewards)
        
        return {
            "energy": min(1.0, len(tiles) / 100),
            "flow": round(avg_reward, 3),
            "frustration": round(1.0 - positive_ratio, 3),
            "discovery": round(len(set(t.get("action", "") for t in tiles)) / max(len(tiles), 1), 3),
            "tension": round(1.0 - min(rewards + [0]), 3) if rewards else 0,
            "confidence": round(positive_ratio, 3),
            "tile_count": len(tiles),
        }
    
    def _compute_jepa_context(self, tiles: List[Dict]) -> List[float]:
        """Compute 6-dimensional JEPA context vector."""
        sent = self._compute_sentiment(tiles)
        return [sent["energy"], sent["flow"], sent["frustration"],
                sent["discovery"], sent["tension"], sent["confidence"]]
    
    def _write_ensign_manifest(self, path: Path, metadata: Dict):
        """Write the manifest that EnsignLoader reads."""
        ensign_type = metadata["ensign_type"]
        manifest = {
            "room_id": metadata["room_id"],
            "ensign_type": ensign_type,
            "tile_count": metadata["tile_count"],
            "exported_at": metadata["exported_at"],
            "files": {
                "metadata": "metadata.json",
                "dataset": "dataset.json",
                "model": "model.json",
                "sentiment": "sentiment.json",
                "jepa_context": "jepa_context.json",
            }
        }
        
        # Add format-specific entries
        if ensign_type == "lora":
            manifest["files"]["lora_config"] = "lora/adapter_config.json"
        elif ensign_type == "tiny_gguf":
            manifest["files"]["gguf_model"] = "model.gguf"
        elif ensign_type == "interpreter":
            manifest["files"]["interpreter"] = "interpreter.json"
        
        with open(path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
    
    def list_exports(self) -> Dict[str, Dict]:
        """List all exported ensigns."""
        exports = {}
        for ensign_dir in self.ensign_dir.iterdir():
            if ensign_dir.is_dir():
                meta_path = ensign_dir / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        exports[ensign_dir.name] = json.load(f)
        return exports
    
    def pack_for_deployment(self, room_id: str, output_file: Optional[str] = None) -> Path:
        """Pack an ensign for fleet deployment (tar archive)."""
        import tarfile
        
        ensign_path = self.ensign_dir / room_id
        if not ensign_path.exists():
            raise FileNotFoundError(f"No ensign for room: {room_id}")
        
        output_file = output_file or f"{room_id}-ensign.tar.gz"
        output_path = Path(output_file)
        
        with tarfile.open(output_path, "w:gz") as tar:
            for file in ensign_path.rglob("*"):
                if file.is_file():
                    tar.add(file, arcname=file.relative_to(self.ensign_dir))
        
        print(f"📦 Packed: {output_path} ({output_path.stat().st_size} bytes)")
        return output_path


class FleetDistributor:
    """Distribute ensigns across the fleet via git (Layer 3: Current)."""
    
    def __init__(self, fleet_repo: str = "SuperInstance/plato-ensign"):
        self.fleet_repo = fleet_repo
    
    def ship_to_fleet(self, ensign_path: Path, commit_msg: str = "") -> Dict:
        """Ship an ensign to the fleet repo.
        
        Copies the ensign into the fleet repo's ensigns/ directory,
        commits, and pushes.
        """
        ensign_path = Path(ensign_path)
        room_id = ensign_path.name
        
        # In production, this would git clone/pull, copy, commit, push
        # For now, document the protocol
        return {
            "action": "ship_ensign",
            "room_id": room_id,
            "repo": self.fleet_repo,
            "path": f"ensigns/{room_id}/",
            "protocol": "git_push",
            "layer": "current",
            "status": "protocol_defined_implementation_pending"
        }
    
    def deploy_to_edge(self, room_id: str, target: str = "jetson") -> Dict:
        """Deploy ensign to edge device.
        
        For JC1's Jetson: git pull on the vessel repo, then load ensign.
        """
        return {
            "action": "deploy_edge",
            "room_id": room_id,
            "target": target,
            "protocol": "git_pull + EnsignLoader.load()",
            "status": "protocol_defined"
        }


if __name__ == "__main__":
    # Demo: export a sample room
    exporter = EnsignExporter(ensign_dir="/tmp/ensigns")
    
    # Simulate tiles from a poker room
    sample_tiles = []
    for i in range(50):
        actions = ["raise", "call", "fold", "check", "all-in"]
        outcomes = ["won pot", "lost", "folded correctly", "doubled up", "saved chips"]
        import random
        action = random.choice(actions)
        reward = random.uniform(-1, 1)
        sample_tiles.append({
            "state_hash": hashlib.md5(f"poker-state-{i % 10}".encode()).hexdigest()[:8],
            "action": action,
            "outcome": random.choice(outcomes),
            "reward": reward,
            "agent": f"agent-{i % 3}",
            "room_id": "poker-table-1",
        })
    
    path = exporter.export_room(
        room_id="poker-table-1",
        tiles=sample_tiles,
        preset="reinforce",
        format="json",
    )
    
    print(f"\nExported to: {path}")
    print(f"Files: {[f.name for f in path.iterdir()]}")
    
    # List all exports
    print(f"\nAll ensigns: {json.dumps(exporter.list_exports(), indent=2)}")
