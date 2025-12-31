# RHEO SNN - Brain & Survival Lab

A **Spiking Neural Network (SNN)** simulation featuring biologically-inspired learning with metabolic constraints and hormonal modulation.

##  Project Novelty

### Metabolic Constraints
Unlike traditional neural networks, RHEO neurons have:
- **Limited Energy**: Each neuron has a metabolic energy budget that depletes when firing
- **Fatigue**: Sustained activity leads to elevated firing thresholds
- **Recovery**: Energy regenerates over time, with layer-specific rates

### Hormonal Modulation
The network features neuromodulators that affect behavior:
- **Dopamine (DA)**: Reward signal for R-STDP learning
- **Acetylcholine (ACh)**: Attention/alertness when near obstacles
- **Serotonin (5HT)**: Stress response that spikes on failure
- **Exploration Noise**: Increases after failed epochs to try new paths

##  Project Structure

```
rheo-snn/
├── main.py                    # Main application entry point
├── brain_weights/             # Saved neural network weights
│   └── brain_weights.npz      # Trained brain (auto-generated)
├── experiments/               # Simulation stats and logs
│   ├── simulation_stats.json  # Latest experiment
│   └── simulation_stats_*.json # Timestamped backups
├── src/
│   ├── __init__.py
│   ├── network.py             # Layer-based SNN with R-STDP
│   ├── layer.py               # LIF neuron layer with metabolism
│   ├── encoding.py            # Sensor-to-spike encoding
│   ├── decoding.py            # Spike-to-motor decoding
│   ├── monitor.py             # Performance tracking & analysis
│   ├── environment/
│   │   ├── simulation.py      # Pygame environment with physics
│   │   ├── editor.py          # Visual map editor
│   │   └── maps/              # Custom JSON map files
│   └── utils/
│       └── file_manager.py    # Directory & file utilities
└── README.md
```

##  Quick Start

### Requirements
```bash
pip install numpy pygame
```

### Run the Simulation
```bash
python -m main
```

##  Controls

### Main Menu
| Button | Function |
|--------|----------|
| **START** | Begin simulation on selected map |
| **SETTINGS** | Adjust sensors, range, hidden neurons |
| **LOAD** | Load previously trained brain |
| **ANALYSIS** | View detailed network statistics |
| **MAP EDITOR** | Create custom environments |
| **EXPERIMENTS** | Open experiment logs folder |

### During Simulation
| Key | Action |
|-----|--------|
| `+` / `-` | Increase/decrease simulation speed |
| `0` | Toggle Turbo Mode (10x speed) |
| `ESC` | Return to menu (auto-saves brain) |

##  Map Editor

Create custom environments with the visual editor.

### Controls
| Key/Mouse | Action |
|-----------|--------|
| **Left Click** | Draw walls (drag to paint) |
| **Right Click** | Erase walls |
| **W** | Select Wall tool |
| **E** | Select Erase tool |
| **F** | Place Food spawn |
| **G** | Place Goal (gold target) |
| **S** | Set agent Spawn point |
| **Ctrl+S** | Save map |
| **Ctrl+L** | Load map |
| **ESC** | Exit editor |

Maps are saved to `src/environment/maps/` as JSON files.

##  Brain Persistence

### Auto-Save
The brain automatically saves to `brain_weights/brain_weights.npz` when:
- Returning to menu (ESC key)
- Completing an epoch with a successful goal

### Manual Load
Click **LOAD** in the menu to restore a previously trained brain. The network dimensions (sensors, hidden neurons) must match.

##  Experiment Logs

Performance data is automatically saved to `experiments/`:
- **simulation_stats.json**: Latest session (overwritten)
- **simulation_stats_YYYYMMDD_HHMMSS.json**: Timestamped backups

### Logged Metrics
| Metric | Description |
|--------|-------------|
| `firing_rates` | Neural activity per layer over time |
| `energy_levels` | Metabolic health per layer |
| `rewards` | Reward signal history |
| `weights_mean/std` | Synaptic weight evolution |
| `success_epochs` | Goal completion times (learning curve) |

##  Network Architecture

```
Input Layer (Sensors)      Hidden Layer (Processing)     Output Layer (Motors)
     10 neurons       →         50 neurons          →        2 neurons
   High recovery            Recurrent connections          High energy cost
   Low energy cost          Moderate metabolism            Learnable via R-STDP
```

### Weight Matrices
| Connection | Shape | Learnable |
|------------|-------|-----------|
| Input → Hidden | (50, 10) | No |
| Hidden → Hidden | (50, 50) | No |
| Hidden → Output | (2, 50) | **Yes (R-STDP)** |

##  Learning Mechanism

### R-STDP (Reward-modulated Spike-Timing Dependent Plasticity)
1. **Eligibility Trace**: Tracks recent pre-post spike correlations
2. **Reward Signal**: Dopamine from goal (+500) or penalty (-50)
3. **Weight Update**: `Δw = learning_rate × eligibility × dopamine`

### Exploration
After epoch timeout:
- Serotonin spikes (stress)
- Eligibility trace weakens recent paths
- Exploration noise increases 3x
- Forces agent to try new strategies

##  Configuration

Edit `main.py` to adjust defaults:
```python
DEFAULT_CONFIG = {
    'num_sensors': 10,    # Ray sensors
    'sensor_range': 180,  # Sensor distance (px)
    'num_hidden': 50,     # Hidden layer neurons
}
```

---