# GridMazeWorld Environment & Training Algorithms

**Author:** Samuel Kuchta  
xkucht11@stud.fit.vutbr.cz  

---

## Overview

This project implements the **GridMazeWorld** environment along with training algorithms for reinforcement learning agents.

The maze environment layout, vectorisation, network models and REINFORCE trainer are inspired by:  
https://github.com/michal-hradis/maze-rl  

---

## Running Experiments

To run the full experimental pipeline:

```bash
python tools/run_experiments.py
```

---

## Running Custom Tasks

Custom task configurations can be explored in:

```
parser.py
```

Examples are provided in the file.

---

## Setup & Usage Guide

### 0. Install Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

we need opencv for c++ if not already installed.

on linux:
```bash
apt-get install -y libopencv-dev

```
on windows download, extract and put into %PATH%: https://sourceforge.net/projects/opencvlibrary/


example on google collab: https://colab.research.google.com/drive/1o8cO3dCzL7U-lZhuhCCrZ7CBpT8TyppX?usp=sharing

---

### 1. Try Human Play Mode (Need desktop with display for visualisation)

Get familiar with the environment by playing manually (action keys shown in console (W/A/S/D/B/Space)):

```bash
python run.py test --play --epochs 1 --dynamic-complexity --curriculum-stages complex --grid-size 19
```

---

### 2. Watch a Trained LSTM Agent (Need desktop with display for visualisation)

Inspect how a trained agent behaves:

```bash
python run.py test --model experiments/lstm768_example.pt --epochs 10 --visualize --show-trail --task-class complex --complexity-level 0.5 --grid-size 19 --max-steps 200
```

**Controls during visualization:**

- `O` — show current agent observations  
- `P` — pause/resume simulation  
- Other controls are shown in the console  

---

### 3. Understand the Environment

Watch `env.mp4` to understand what the agent perceives in the environment.

---

### 4. Run Full Experiments

Expected runtime: ~2 days (Geforce GTX 1660 super, Intel Core I5-8500). Recommended: CUDA-enabled PyTorch + GPU with 6GB+ VRAM  

```bash
python tools/run_experiments.py
```

---

### 5. While Training Runs

Read the accompanying thesis:

```
xkucht11_DIP.pdf
```

---

### 6. Explore the Codebase (Optional)

Review source code to better understand implementation details.

---

### 7. Compare Results

After experiments are complete, compare outputs with results reported in the thesis.
The exact values may vary due to differences in GPU hardware, as CUDA network training operations are stochastic. Therefore, even when the environment is generated deterministically using the same seed, different systems may produce slightly different results. However, the overall trends and behaviors should remain consistent across machines. Repeated runs on the same machine should, however, produce identical results.

```