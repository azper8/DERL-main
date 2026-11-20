  
# Dual‑Policy Evolutionary Reinforcement Learning for Deadline-Constrained Dynamic Workflow Scheduling in Edge Computing

## 📦 Environment Setup

You can set up the environment using the following methods:

### Using Conda (Recommended)

```bash

conda env create -f environment.yml -n your-env-name

conda activate your_env_name

````

> Make sure your Python version is `>=3.8`.


## 🚀 Quick Start

1.  Run  the  main  program:

```bash

python main.py

```
2. Test the saved model:

```bash

python eval.py

```
3. Launch the HTTP-based distributed system:
```bash
python network/run_all.py
```

> To run the code in a server, you need to customize `main.py` or `eval.py` to a service script.
> For multiple independent runs, `main.py` should use a different random seed everytime.


## 📊 Example Output

After training and testing, results will be saved in the `logs/` directory, including reward, total makespan and deadline violations, which can be used for visualization and evaluation.