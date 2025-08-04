# AI-Powered Snake Game

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An AI agent that learns to play the classic Snake game using Deep Q-learning (DQN) with PyTorch and Pygame.

---

### ✨ Features

- **AI Training:** Implemented with the Deep Q-Network (DQN) algorithm.
- **Live Visualization:** Real-time gameplay screen to observe the AI's learning progress.
- **Performance Tracking:** Plots scores and mean scores to track learning progress.

---

### 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** PyTorch, Pygame, NumPy

---

### 🚀 Getting Started

**1. Clone the repository:**
```bash
git clone https://github.com/masddffee/MIT.git
cd MIT
```

**2. Install dependencies:**
```bash
# It is recommended to use a virtual environment
python -m venv env
source env/bin/activate # On Windows use `env\Scripts\activate`

pip install torch pygame numpy matplotlib
```

**3. Run the training script:**
```bash
python agent.py
```

**4. Run the game with a pre-trained model:**
```bash
python play_snake.py # Make sure a model.pth file exists
```

---
