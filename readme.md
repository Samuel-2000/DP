maze_rl/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── run.py
├── configs/
│   ├── default.yaml
│   ├── lstm.yaml
│   ├── transformer.yaml
│   └── multimemory.yaml
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── environment.py
│   │   ├── agent.py
│   │   └── utils.py
│   ├── networks/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── lstm.py
│   │   ├── transformer.py
│   │   └── multimemory.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── optimizers.py
│   └── evaluation/
│       ├── __init__.py
│       ├── benchmark.py
│       ├── visualization.py
│       └── metrics.py
├── experiments/
│   ├── __init__.py
│   ├── train.py
│   ├── compare.py
│   └── analyze.py
├── scripts/
│   ├── train_all.sh
│   ├── benchmark.sh
│   └── visualize_results.py
├── models/
├── logs/
├── results/
│   ├── benchmarks/
│   ├── plots/
│   └── videos/
└── tests/
    ├── test_environment.py
    ├── test_networks.py
    └── test_training.py