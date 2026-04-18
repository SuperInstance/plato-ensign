# plato-ensign

> *"Walk into a room. The room hands you its wisdom."*

## What Is This?

The **Ensign Protocol** trains room-specific AI artifacts from accumulated agent interactions. When an agent "walks into" a PLATO room, the room's trained ensign loads instantly — like putting on glasses that let you see the room through accumulated instinct.

## Three Ensign Types

| Type | Format | Size | Target |
|------|--------|------|--------|
| **LoRA Adapter** | .safetensors | 5-50MB | GPU agents (FM/JC1) |
| **Tiny Model** | .gguf | 10-100MB | CPU-only agents, greenhorns |
| **Interpreter** | .gguf/.onnx | 50-200MB | Cross-paradigm translation |

## How It Works

```
Agent plays poker → room accumulates tiles → tiles train ensign →
next agent enters → loads ensign → plays better because the room learned
```

1. **Accumulate:** Every room interaction generates training tiles
2. **Distill:** Periodically, tiles are distilled into trained models (FM's RTX 4050)
3. **Deploy:** Ensigns ship as .gguf/.safetensors to room registries
4. **Load:** Agent enters room → ensign hot-loads → instant instinct

## Quick Start

```python
from plato_ensign import EnsignLoader, RoomTrainer

# Train an ensign from accumulated room tiles
trainer = RoomTrainer(room_id="poker-table-1", base_model="Qwen/Qwen2.5-0.5B")
trainer.train_lora(output_path="ensigns/poker-table/lora/")
trainer.train_tiny(output_path="ensigns/poker-table/tiny.gguf")

# Load an ensign when entering a room
loader = EnsignLoader()
ensign = loader.load("poker-table-1")  # hot-swaps LoRA or loads GGUF
result = ensign.infer("Should I raise with AK suited in late position?")
```

## Architecture

```
plato-ensign/
├── src/
│   ├── ensign_loader.py      # Load ensigns (LoRA hot-swap or GGUF)
│   ├── room_trainer.py       # Train ensigns from tile buffers
│   ├── tile_buffer.py        # Accumulate interaction tiles
│   ├── distillation.py       # Knowledge distillation engine
│   └── interpreter.py        # Cross-paradigm translator
├── ensigns/                  # Trained ensign artifacts (gitignored, large)
│   └── poker-table/
│       ├── lora/
│       ├── tiny.gguf
│       └── metadata.json
├── research/
│   └── paper-ensign-protocol.md
└── tests/
```

## Fleet Integration

- **FM** (RTX 4050): Trains ensigns using PyTorch + PEFT + QLoRA
- **JC1** (Jetson): Deploys and benchmarks ensigns on edge
- **Oracle1** (Cloud): Serves ensigns via subcontractor API
- **Rooms** (holodeck-rust): Accumulate tiles, trigger distillation

## Research

Full paper: `research/paper-ensign-protocol.md`

## License

MIT
