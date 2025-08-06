<h1 align="center">🔬 AttriLens-Mol: Attribute Guided Reinforcement Learning for Molecular Property Prediction with Large Language Models</h1>

<p align="center">
  <strong>Official GitHub Repository for the AttriLens-Mol</strong><br>
</p>

---
## 🧠 Overview
This repository introduces **AttriLens-Mol**, an **attribute-guided reinforcement learning framework** for molecular property prediction with large language models.  

<p align="center">
  <img src="https://github.com/user-attachments/assets/00a47fd6-17f9-4f97-a06f-4f37609ded69" width="100%" />
</p>

---

## 📦 What’s Included

```text
├── Dataset/ # Dataset of Random Frorest and AttriLens-Mol train set
├── Random Forest/ # Script for building random forest models for different tasks
├── AttriLens_train.py / # Pre-training script for AttriLens-Mol
├── requirements.txt / # requirements for env
```

---
## 🛠️ Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/szu-tera/AttriLens-Mol.git
cd AttriLens-Mol
pip install -r requirements.txt
```

## 🚀 Usage

### 🔧 1. Pretrain AttriLens-Mol
python AttriLens_train.py # Before running the script, edit line 611 and 639 in the file:
                           # json_file_path = ""
                           # output_dir=""
                           # and set it to your data file path and ckpt output file.


### 🌲 2. Random Forest Building
cd Random Forest
python Random Forest_{task}.py # Before running the script, edit line 13-15 in the file:
                               # train_df = pd.read_csv('bace_gen_train.csv')
                               # valid_df = pd.read_csv('bace_gen_valid.csv')
                               # test_df = pd.read_csv('bace_gen_test.csv')


## 📜 More Data

We will release all training data and preprocessing scripts once the paper is accepted.  
Stay tuned and ⭐ star this repo to get notified!


