# RECSYS-RL Group Repository

Welcome to our group repository for the RECSYS-RL coursework. This repository is organized by coursework themes and contains separate folders for each assignment and project. The structure is designed to keep work modular and manageable across different domains.

---

## Repository Structure

```
RECSYS-RL/
│
├── information-retrieval/
│   └── critical-review/
│       ├── instructions/         # Assignment description and notes
│       ├── report/               # Your LaTeX or PDF report files
│       └── slides/               # Presentation slides
│
├── recommender-systems/
│   └── assignment-1/
│       ├── code/                 # All Python scripts (models, training, eval)
│       ├── data/                 # Dataset files like .csv or .dat
│       ├── instructions/         # Assignment brief and grading rubric
│       └── report/               # Final report in LaTeX/PDF
│       └── README.md             # Instructions for this assignment
│
├── reinforcement-learning/
│   └── assignment-2/
│       ├── code/
│       ├── instructions/
│       └── report/
│       └── README.md
```

---

## Getting Started

To get this repository on your local machine:

### Step 1: Clone the Repository
Open your terminal and run:

```bash
git clone https://github.com/adala-wanyande/RECSYS-RL.git
```

### Step 2: Navigate to Your Working Folder

Example if you're working on recommender systems assignment:
```bash
cd RECSYS-RL/recommender-systems/assignment-1/code
```

---

## Guidelines for Contribution

To avoid conflicts and keep work clean:

1. **Only work within your assigned folders.**
   - E.g., if you're assigned evaluation code, only modify files in `assignment-1/code/` and save outputs in `data/` or `report/` if necessary.

2. **Always pull latest changes before starting:**
   ```bash
   git pull origin main
   ```

3. **After making changes:**
   ```bash
   git add .
   git commit -m "Add your short but clear message here"
   git push origin main
   ```

---

## Tools Recommended

- VSCode (with Python, Git extensions)
- Python 3.10+
- Virtual environment (optional but preferred)

---

## Communication

If you're unsure where to put your work or how to do something, feel free to ask on our WhatsApp chat or comment directly in the repo via issues or pull requests.
