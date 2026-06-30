# Microglia-Neuron-Astrocyte Model

`MicrogliaResearchModel` is a Python Mesa agent-based model for studying interactions among neurons, microglia, astrocytes, amyloid burden, lipid-droplet-associated microglial dysfunction, and pro/anti-inflammatory signaling in a simplified CNS tissue slice.

The model is intended for computational experiments, parameter sweeps, and hypothesis exploration. It is not a clinical or diagnostic model.

## Installation

```bash
pip install MicrogliaResearchModel
```

For local development from this repository:

```bash
pip install -e .
```

## Quick Start

```python
from MicrogliaResearchModel import run_sim

model, df = run_sim(
    steps=100,
    seed=42,
    initial_amyloid_patches=5,
    amyloid_carrying_capacity=5.0,
)

print(df.tail())
```

The returned dataframe contains population counts, cytokine totals, amyloid burden metrics, lipid burden metrics, DAM/LDAM fractions, and genotype-related reporters.

## Command Line

After installation, the package exposes a console command:

```bash
microglia-research-model --steps 100 --seed 42 --output_csv results.csv --no_plot
```

Useful CLI options include:

- `--steps`: number of model ticks to simulate.
- `--output_csv`: CSV path for model reporters.
- `--no_csv`: skip saving the reporter dataframe to CSV.
- `--summary_png`: save the summary plot instead of showing it interactively.
- `--no_plot`: skip plotting.
- `--apoe_genotype`: APOE genotype label, such as `3/3`, `3/4`, or `4/4`.
- `--trem2_mutation_rate`: fraction of microglia initialized with reduced TREM2 activity.

## Model Scope

The model uses coarse phenotype-like states:

- Microglia: `M0`, `M1`, `M2`, with DAM and LDAM modifiers.
- Astrocytes: `A0`, `A1`, `A2`.
- Neurons: healthy, damaged, or dead.

These labels are abstractions for model state transitions, not literal fixed biological identities. Reactive glial biology is more continuous and context-dependent than a three-state model can capture.

Current model features include:

- M1/M2 and A1/A2 resolution back toward homeostasis under persistent low signal.
- Pro/anti inflammatory phenotype thresholds centered around a balanced ratio of 1.0.
- Logistic amyloid growth with a configurable carrying capacity.
- Microglial sensing and eating efficiencies connected to baseline sweep parameters.
- Lipid-burden, amyloid-exposure, DAM, LDAM, APOE, and TREM2 effects on microglial behavior.
- Amyloid concentration and amyloid-positive patch reporters.

## Biological Assumptions

The model is motivated by literature on lipid-droplet-accumulating microglia, disease-associated microglia, reactive astrocyte states, and microglial responses to tissue injury. These mechanisms are represented as simplified, parameterized abstractions so that experiments can test qualitative behavior under different assumptions.

Important caveats:

- `M1/M2` and `A1/A2` are coarse model states.
- LDAM toxic lipid transfer is modeled as an effective toxic burden, not strict lipid mass conservation.
- Amyloid values are continuous local burden values, while amyloid patch counts report occupied grid patches.
- Parameter values should be calibrated or sensitivity-tested before biological interpretation.
