# ml-05

Implement and evaluate more complex models when simpler techniques aren't enough. 

---

## Get Started

1. Copy template repo to your GitHub account. 
2. Clone your repo down to your machine. 
3. Open your project repository folder in VS Code.
 
---

## Windows Setup Instructions

Open a PowerShell terminal in VS Code. 
Create local project virtual environment, activate it, and install packages. 
When asked to use the new .venv click yes. 

Important: Install from requirements first with scikit-learn commented out. 
Then remove the leading hashmark (around line 187) and re-run the command to install scikit-learn.
See requirements.txt for more information. 


```shell
py -m venv .venv
.\.venv\Scripts\activate
py -m pip install --upgrade pip setuptools wheel
py -m pip install -r requirements.txt
py -m pip install -r requirements.txt

```
---

## Mac/Linux Setup Instructions

Open a default terminal in VS Code. 
Create local project virtual environment, activate it, and install packages. 
Important: Install from requirements first with scikit-learn commented out. 
Then remove the leading hashmark (around line 187) and re-run the command to install scikit-learn.
See requirements.txt for more information. 

```zsh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements.txt
```
---

## Tell VS Code to use .venv

Open the Command Palette: Press Ctrl + Shift + P (Windows) or Cmd + Shift + P (Mac/Linux)
Then type: Python: Select Interpreter
Press Enter.

Look for the interpreter with .venv in the path.
Click on that interpreter to select it.
Confirm it's selected: You should see the Python version and .venv path in the lower-left status bar of VS Code.

---

## Working on the Project

Open the .ipynb Jupyter notebook file in VS Code. 
Run the entire notebook before you start to edit. 
As you make progress, use Git to add, commit, and push your work to your GitHub repo.

```shell
git add .
git commit -m "describe the change here"
git push -u origin main
```

For additional information and suggestions, follow the process for professional Project Setup / Initialization / Standard Workflow in
[pro-analytics-01](https://github.com/denisecase/pro-analytics-01)

## Dataset Source

The Wine Quality Dataset is made available by the UCI Machine Learning Repository.

[https://archive.ics.uci.edu/ml/datasets/Wine+Quality](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)

Data originally published by:
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.  
Modeling wine preferences by data mining from physicochemical properties.  
In *Decision Support Systems*, Elsevier, 47(4):547–553, 2009.

Direct download link to raw csv:

<https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv>