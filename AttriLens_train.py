import difflib
import logging

import requests
from rdkit import Chem
from rdkit.Chem import Descriptors
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import re
import pandas as pd
from peft import LoraConfig
# Mapping common property name aliases to RDKit standard descriptor names
PROPERTY_ALIASES = {
    # e.g., all variations of 'MolWt' like 'molweight', 'mw', etc.
    "molweight": "MolWt",
    "molecularweight": "MolWt",
    "mol_weight": "MolWt",
    "molecular_weight": "MolWt",
    "mw": "MolWt",
    "exactmolweight": "ExactMolWt",
    "logp": "MolLogP",
    "xlogp": "MolLogP",
    "xlogp3": "MolLogP",
    "partitioncoefficient": "MolLogP",
    "octanolwater": "MolLogP",
    "tpsa": "TPSA",
    "psa": "TPSA",
    "polarsurfacearea": "TPSA",
    "topologicalpolarsurfacearea": "TPSA",
    "donors": "NumHDonors",
    "numhdonors": "NumHDonors",
    "hbd": "NumHDonors",
    "hydrogenbonddonors": "NumHDonors",
    "acceptors": "NumHAcceptors",
    "numhacceptors": "NumHAcceptors",
    "hba": "NumHAcceptors",
    "hydrogenbondacceptors": "NumHAcceptors",
    "rotatable": "NumRotatableBonds",
    "numrotatablebonds": "NumRotatableBonds",
    "rotatablebonds": "NumRotatableBonds",
    "nrb": "NumRotatableBonds",
    "fsp3": "FractionCSP3",
    "fractioncsp3": "FractionCSP3",
    "heavyatomcount": "HeavyAtomMolWt",
    "numheavyatoms": "HeavyAtomMolWt",
    "ringcount": "RingCount",
    "numrings": "RingCount",
    "molarrefractivity": "MolMR",
    "molmr": "MolMR",
    "polarizability": "MolMR",
    "balaban": "BalabanJ",
    "bertz": "BertzCT",
    "ipcindex": "Ipc",
    "chi0": "Chi0",
    "chi1": "Chi1",
    "chi2n": "Chi2n",
    "chi3n": "Chi3n",
    "chi4n": "Chi4n",
    "chi0v": "Chi0v",
    "chi1v": "Chi1v",
    "chi2v": "Chi2v",
    "chi3v": "Chi3v",
    "chi4v": "Chi4v",
    "kappa1": "Kappa1",
    "kappa2": "Kappa2",
    "kappa3": "Kappa3",
    "alpha": "HallKierAlpha",
    "hallkieralpha": "HallKierAlpha",
}
# Advantageous property ranges per task (BBBP, BACE, Clintox)
# These ranges define what is considered "good" for each molecular property in different tasks
ADVANTAGE_RANGES = {
    "BBBP": {
        "MolWt": (100, 450),
        "ExactMolWt": (100, 450),
        "MolLogP": (1, 4),
        "TPSA": (0, 70),
        "LabuteASA": (0, 100),
        "NumHDonors": (0, 2),
        "NumHAcceptors": (0, 4),
        "NumRotatableBonds": (0, 6),
        "FractionCSP3": (0.2, 0.6),
        "RingCount": (1, 4),
        "NumAromaticRings": (0, 3),
        "NumSaturatedRings": (0, 2),
        "NumAliphaticRings": (0, 2),
        "NumHeteroatoms": (0, 6),
        "HeavyAtomMolWt": (50, 430),
        "NumValenceElectrons": (40, 80),
        "BertzCT": (100, 500),
        "Ipc": (0, 600),
        "BalabanJ": (0.5, 3.5),
        "Kappa1": (0, 6),
        "Kappa2": (0, 3),
        "Kappa3": (0, 1.5),
        "Chi0": (0, 10),
        "Chi1": (0, 8),
        "Chi2n": (0, 10),
        "Chi3n": (0, 10),
        "Chi4n": (0, 10),
        "Chi0v": (0, 12),
        "Chi1v": (0, 10),
        "Chi2v": (0, 10),
        "Chi3v": (0, 10),
        "Chi4v": (0, 10),
        "NumRadicalElectrons": (0, 0),
        "HallKierAlpha": (-2.0, 1.5),

    },

    "BACE": {
        "MolWt": (200, 500),
        "ExactMolWt": (200, 520),
        "MolLogP": (1, 5),
        "TPSA": (0, 60),
        "LabuteASA": (0, 140),
        "NumHDonors": (0, 3),
        "NumHAcceptors": (0, 6),
        "NumRotatableBonds": (0, 8),
        "FractionCSP3": (0.2, 0.6),
        "RingCount": (1, 5),
        "NumAromaticRings": (0, 3),
        "NumSaturatedRings": (0, 3),
        "NumAliphaticRings": (0, 2),
        "NumHeteroatoms": (0, 8),
        "HeavyAtomMolWt": (100, 480),
        "NumValenceElectrons": (50, 90),
        "BertzCT": (200, 600),
        "Ipc": (0, 800),
        "BalabanJ": (0.5, 4.0),
        "Kappa1": (0, 6),
        "Kappa2": (0, 4),
        "Kappa3": (0, 2),
        "Chi0": (0, 12),
        "Chi1": (0, 10),
        "Chi2n": (0, 12),
        "Chi3n": (0, 12),
        "Chi4n": (0, 12),
        "Chi0v": (0, 14),
        "Chi1v": (0, 12),
        "Chi2v": (0, 12),
        "Chi3v": (0, 12),
        "Chi4v": (0, 12),
        "NumRadicalElectrons": (0, 0),
        "HallKierAlpha": (-2.0, 2.0),
    },

    "Clintox": {
        "MolWt": (50, 450),
        "ExactMolWt": (50, 470),
        "MolLogP": (-1, 3),
        "TPSA": (0, 80),
        "LabuteASA": (0, 110),
        "NumHDonors": (0, 3),
        "NumHAcceptors": (0, 6),
        "NumRotatableBonds": (0, 7),
        "FractionCSP3": (0.1, 0.7),
        "RingCount": (0, 4),
        "NumAromaticRings": (0, 2),
        "NumSaturatedRings": (0, 3),
        "NumAliphaticRings": (0, 2),
        "NumHeteroatoms": (0, 8),
        "HeavyAtomMolWt": (50, 440),
        "NumValenceElectrons": (30, 75),
        "BertzCT": (50, 450),
        "Ipc": (0, 650),
        "BalabanJ": (0.5, 3.8),
        "Kappa1": (0, 6),
        "Kappa2": (0, 4),
        "Kappa3": (0, 2),
        "Chi0": (0, 10),
        "Chi1": (0, 8),
        "Chi2n": (0, 10),
        "Chi3n": (0, 10),
        "Chi4n": (0, 10),
        "Chi0v": (0, 12),
        "Chi1v": (0, 10),
        "Chi2v": (0, 10),
        "Chi3v": (0, 10),
        "Chi4v": (0, 10),
        "NumRadicalElectrons": (0, 0),
        "HallKierAlpha": (-2.5, 2.0),
    }
}
# Mapping property names to RDKit descriptor functions
# Used to compute properties from SMILES with RDKit
RDKIT_FUNCTIONS = {
    "MolWt": Descriptors.MolWt,
    "molecularweight": Descriptors.MolWt,
    "polar_surface_area": "Descriptors.TPSA",
    "hydrogendonorcount": "Descriptors.NumHDonors",
    "MinPartialCharge": "Descriptors.MinPartialCharge",
    "ExactMolWt": Descriptors.ExactMolWt,
    "NumValenceElectrons": Descriptors.NumValenceElectrons,
    "NumRadicalElectrons": Descriptors.NumRadicalElectrons,
    "MaxPartialCharge": Descriptors.MaxPartialCharge,
    "MinPartialCharge": Descriptors.MinPartialCharge,
    "MaxAbsPartialCharge": Descriptors.MaxAbsPartialCharge,
    "MinAbsPartialCharge": Descriptors.MinAbsPartialCharge,
    "FpDensityMorgan1": Descriptors.FpDensityMorgan1,
    "FpDensityMorgan2": Descriptors.FpDensityMorgan2,
    "FpDensityMorgan3": Descriptors.FpDensityMorgan3,
    "Chi0": Descriptors.Chi0,
    "Chi0n": Descriptors.Chi0n,
    "Chi0v": Descriptors.Chi0v,
    "Chi1": Descriptors.Chi1,
    "Chi1n": Descriptors.Chi1n,
    "Chi1v": Descriptors.Chi1v,
    "Chi2n": Descriptors.Chi2n,
    "Chi2v": Descriptors.Chi2v,
    "Chi3n": Descriptors.Chi3n,
    "Chi3v": Descriptors.Chi3v,
    "Chi4n": Descriptors.Chi4n,
    "Chi4v": Descriptors.Chi4v,
    "HallKierAlpha": Descriptors.HallKierAlpha,
    "Kappa1": Descriptors.Kappa1,
    "Kappa2": Descriptors.Kappa2,
    "Kappa3": Descriptors.Kappa3,
    "LabuteASA": Descriptors.LabuteASA,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "HeavyAtomMolWt": Descriptors.HeavyAtomMolWt,
    "NumHeteroatoms": Descriptors.NumHeteroatoms,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "NumValenceElectrons": Descriptors.NumValenceElectrons,
    "NumAromaticRings": Descriptors.NumAromaticRings,
    "NumAliphaticRings": Descriptors.NumAliphaticRings,
    "NumSaturatedRings": Descriptors.NumSaturatedRings,
    "NumAromaticCarbocycles": Descriptors.NumAromaticCarbocycles,
    "NumAromaticHeterocycles": Descriptors.NumAromaticHeterocycles,
    "NumAliphaticCarbocycles": Descriptors.NumAliphaticCarbocycles,
    "NumAliphaticHeterocycles": Descriptors.NumAliphaticHeterocycles,
    "NumSaturatedCarbocycles": Descriptors.NumSaturatedCarbocycles,
    "NumSaturatedHeterocycles": Descriptors.NumSaturatedHeterocycles,
    "MolLogP": Descriptors.MolLogP,
    "MolMR": Descriptors.MolMR,
    "TPSA": Descriptors.TPSA,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumHDonors": Descriptors.NumHDonors,
    "FractionCSP3": Descriptors.FractionCSP3,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "RingCount": Descriptors.RingCount,
    "molecularweight": "MolWt",
    "molweight": "MolWt",
    "mol_weight": "MolWt",
    "molecular_weight": "MolWt",
    "exactmolecularweight": "ExactMolWt",
    "octanol-water": "MolLogP",
    "polar_surface_area": "TPSA",

    "logp": "MolLogP",
    "log_p": "MolLogP",
    "xlogp": "MolLogP",
    "partitioncoefficient": "MolLogP",

    "tpsa": "TPSA",
    "topologicalpolarsurfacearea": "TPSA",
    "topological_polar_surface_area": "TPSA",
    "polar_surface_area": "TPSA",
    "psa": "TPSA",

    "numhdonors": "NumHDonors",
    "hbd": "NumHDonors",
    "hbdonors": "NumHDonors",
    "hydrogendonorcount": "NumHDonors",

    "numhacceptors": "NumHAcceptors",
    "hba": "NumHAcceptors",
    "hbacceptors": "NumHAcceptors",
    "hydrogenacceptorcount": "NumHAcceptors",


    "numrotatablebonds": "NumRotatableBonds",
    "rotatablebonds": "NumRotatableBonds",
    "nrb": "NumRotatableBonds",


    "fractioncsp3": "FractionCSP3",
    "fsp3": "FractionCSP3",


    "heavyatomcount": "HeavyAtomCount",
    "numheavyatoms": "HeavyAtomCount",


    "ringcount": "RingCount",
    "numrings": "RingCount",

    "molmr": "MolMR",
    "molarrefractivity": "MolMR",
}



# Template for XML-format output
XML_COT_FORMAT = """
<name>
{name}
</name>
<answer>
{answer}
</answer>
"""
# Load tokenizer and model from local path
import json
model_name = "./r1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Apply LoRA (Low-Rank Adaptation) for efficient fine-tuning
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],  # 适用于 Qwen、LLaMA 结构
    task_type="CAUSAL_LM",
)


# Load base model and wrap with PEFT for LoRA
model = AutoModelForCausalLM.from_pretrained(model_name)
model = get_peft_model(model, lora_config)
import re

# Extracts content from <answer> XML tags
def extract_xml_answer(text: str) -> str:
    match = re.search('<answer>(.*)</answer>', text, re.DOTALL)
    if match:
        answer = match.group(1)
    else:
        answer = ''
    return answer.strip()

# Reward: 2 if extracted answer matches the target string; otherwise 0
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]

    print("=" * 80)
    print(f"[PROMPT]: {prompts[0][-1]['content']}")
    print(f"[GENERATIONS]:")
    print(f"[GROUND TRUTH]: {answer[0]}")
    for i, (r, ext) in enumerate(zip(responses, extracted_responses)):
        print(f"--- Generation {i + 1} ---")
        print(r)
        print(f"Extracted Answer: {ext}")
        print(f"[GROUND TRUTH]: {answer[0]}")
        print()

    return [2 if a in r else 0.0 for r, a in zip(extracted_responses, answer)]

# Tries to match a user-given property name to a standard RDKit property name based on alias, RDKit, or fallback
def fuzzy_match_advantage_key(prop_name: str, task_name: str, cutoff: float = 0.1):
    if task_name not in ADVANTAGE_RANGES:
        return None

    advantage_keys = ADVANTAGE_RANGES[task_name].keys()
    normalized = normalize_property_name(prop_name)

    # Step 1: alias fuzzy match
    alias_keys = list(PROPERTY_ALIASES.keys())
    alias_match = difflib.get_close_matches(normalized, alias_keys, n=1, cutoff=cutoff)
    if alias_match:
        alias_target = PROPERTY_ALIASES[alias_match[0]]
        if alias_target in advantage_keys:
            print(f"[Match Debug] Original: {prop_name}, Normalized: {normalized}, Matched: {alias_target} (alias)")
            return alias_target

    # Step 2: RDKit fuzzy match
    rdkit_key = fuzzy_match_property(normalized, cutoff=cutoff)
    if rdkit_key and rdkit_key in RDKIT_FUNCTIONS:
        rdkit_mapped = RDKIT_FUNCTIONS[rdkit_key]
        if isinstance(rdkit_mapped, str):
            rdkit_mapped = rdkit_mapped.split('.')[-1]
        elif callable(rdkit_mapped):
            for k, v in Descriptors.__dict__.items():
                if v == rdkit_mapped:
                    rdkit_mapped = k
                    break
        if rdkit_mapped in advantage_keys:
            print(f"[Match Debug] Original: {prop_name}, Normalized: {normalized}, Matched: {rdkit_mapped} (RDKit)")
            return rdkit_mapped

    # Step 3: fallback match with advantage keys
    matches = difflib.get_close_matches(normalized, list(advantage_keys), n=1, cutoff=cutoff)
    if matches:
        print(f"[Match Debug] Original: {prop_name}, Normalized: {normalized}, Matched: {matches[0]} (fallback)")
        return matches[0]

    # ❌ All failed
    print(f"[Match Debug] Original: {prop_name}, Normalized: {normalized}, Matched: No match")
    return None

# Match a property name using fuzzy string similarity against RDKit keys
def fuzzy_match_property(prop_name: str, cutoff: float = 0.1):
    candidates = list(RDKIT_FUNCTIONS.keys())
    match = difflib.get_close_matches(prop_name, candidates, n=1, cutoff=cutoff)
    return match[0] if match else None

# Parses <name>...</name> content and converts it into list of properties + corresponding +/-1 label
def parse_name_and_status(text: str):

    match = re.search(r"<name>(.*?)</name>", text, re.DOTALL)
    if not match:
        return [], []

    content = match.group(1).strip()
    items = [item.strip() for item in content.split(",") if item.strip()]

    props = []
    statuses = []
    for item in items:
        parts = item.split(":")
        if len(parts) != 2:
            continue
        prop = parts[0].strip()
        status_str = parts[1].strip().lower()
        status = 1 if "improve" in status_str else -1 if "not improve" in status_str else -1

        props.append(prop)
        statuses.append(status)
    return props, statuses

# Determine the task name (BBBP/BACE/Clintox) from prompt
def extract_task_from_prompt(prompts):
    if prompts and isinstance(prompts, list) and isinstance(prompts[0], list):
        for message in prompts[0]:
            if message.get("role") == "system":
                system_content = message.get("content", "").lower()
                if "bbbp" in system_content:
                    return "BBBP"
                elif "bace" in system_content:
                    return "BACE"
                elif "toxic" in system_content:
                    return "Clintox"
                return "UNKNOWN_TASK"

    return "UNKNOWN_TASK"

# Normalize property name to lower-case, no spaces/hyphens
def normalize_property_name(raw_name: str) -> str:
    return raw_name.lower().replace(" ", "").replace("_", "").replace("-", "")

# Extract SMILES string from the prompt list
def extract_smiles_from_prompt(prompts):
    content = prompts[1]['content']
    if content and content.strip():
        smiles = content.strip().split()[-1]
        return smiles
    return None


# Computes reward by comparing predicted property change directions against RDKit-calculated ones,
# and whether values fall within advantageous task-specific ranges.
def advantage_reward_func(prompts, completions, answer=None, **kwargs):
    rewards = []

    task_name = kwargs.get('task_name')
    if not task_name:
        task_name = extract_task_from_prompt(prompts)

    advantage_ranges = ADVANTAGE_RANGES.get(task_name, {})

    for idx, (prompt, completion) in enumerate(zip(prompts, completions)):
        content = completion[0].get("content", "") if isinstance(completion, list) and len(completion) > 0 else ""

        props, improve_labels = parse_name_and_status(content)
        if not props or not improve_labels:
            print(f"[{idx}] Missing props or labels, reward=0.0")
            rewards.append(0.0)
            continue

        smiles = extract_smiles_from_prompt(prompt)
        if not smiles:
            print(f"[{idx}] Missing SMILES, reward=0.0")
            rewards.append(0.0)
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"[{idx}] Invalid SMILES: {smiles}, reward=0.0")
            rewards.append(0.0)
            continue

        total_score = 0
        count = 0
        print(f"\n[{idx}] SMILES: {smiles}")
        print(f"Content: {content}")
        print("Predicted Label  RDKit Label  Advantage")

        for prop, label in zip(props, improve_labels):
            fuzzy_prop = fuzzy_match_advantage_key(prop, task_name)
            if not fuzzy_prop:
                print(f"  Ignoring property: {prop} (no match)")
                continue
            if fuzzy_prop not in RDKIT_FUNCTIONS or fuzzy_prop not in advantage_ranges:
                print(f"  Ignoring property: {fuzzy_prop} (no function or range defined)")
                continue

            try:
                value = RDKIT_FUNCTIONS[fuzzy_prop](mol)
            except Exception as e:
                print(f"  Failed to compute property {fuzzy_prop}: {e}")
                continue

            min_val, max_val = advantage_ranges[fuzzy_prop]
            rdkit_label = 1 if (min_val <= value <= max_val) else -1

            advantage = abs(label + rdkit_label) / 2
            total_score += advantage
            count += 1

            print(f"  {prop:10}  {label:8}  {rdkit_label:8}  {advantage:6.2f}  (value={value:.4f}, range=[{min_val},{max_val}])")

        if count > 0:
            reward = total_score / count
        else:
            reward = 0.0

        print(f"Average advantage: {reward:.4f}")
        rewards.append(reward)

    return rewards



# Reward 1.0 if output format is valid XML with <name> and <answer>; else -2.0

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r".*?<name>.*?</name>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [1.0 if match else -2.0 for match in matches]

# Penalize outputs with too few/many property predictions (based on count in <name> field)
def choice_number_reward_func(completions, min_number = 3 ,max_number = 10, **kwargs) -> list[float]:
    pattern = r"<name>(.*?)</name>"
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    for response in responses:
        match = re.search(pattern, response)
        if match:
            choices = match.group(1).split(',')
            if min_number <= len(choices) <= max_number:
                rewards.append(0.0)
            else:
                rewards.append(-1.0)
        else:
            rewards.append(-1.5)
    return rewards

# Extract accuracy from a list of model completions and true labels
def compute_acc(eval_preds):
    predictions, labels = eval_preds
    preds = [extract_xml_answer(p[0]['content']) if isinstance(p, list) else "" for p in predictions]
    targets = [label for label in labels]

    correct = sum(p.strip() == t.strip() for p, t in zip(preds, targets))
    acc = correct / len(targets) if targets else 0.0
    return {"accuracy": acc}


# Extracts answer from format like "...#### answer"
def extract_xml_answer(text: str) -> str:
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return match.group(1).strip() if match else ""


# Convert dataset from JSON format to (prompt, answer) pair list
def extract_hash_answer(text: str):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# Convert dataset from JSON format to (prompt, answer) pair list
def process_smiles_data(json_file_path):
    """
    Process the JSON-formatted dataset from a file.

    Args:
    json_file_path (str): Path to the JSON file containing the dataset.

    Returns:
    list: A list of processed samples with 'prompt' and 'answer' fields.
    """
    processed_data = []

    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for sample in data:

        input_smiles = sample['input']
        output = sample['output']

        # 构造 prompt
        prompt = [
            {'role': 'system', 'content': sample['instruction']},
            {'role': 'user', 'content': input_smiles}
        ]
        answer = str(output)
        processed_sample = {
            'prompt': prompt,
            'answer': answer
        }
        processed_data.append(processed_sample)

    return processed_data

# Placeholder JSON path, please update with actual file
json_file_path = ""
processed_dataset = process_smiles_data(json_file_path)
print(processed_dataset[0])



from trl import GRPOConfig, GRPOTrainer
# Set up GRPOConfig with all relevant hyperparameters for training
training_args = GRPOConfig(
    use_vllm=False,
    learning_rate=5e-5,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0,
    lr_scheduler_type="cosine",
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_generations=8,
    max_prompt_length=400,
    max_completion_length=2000,
    num_train_epochs=1,
    save_steps=20,
    max_grad_norm=0.5,
    vllm_gpu_memory_utilization=0.2,
    report_to="tensorboard",
    output_dir="",
)
# Initialize trainer with multiple reward functions
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        soft_format_reward_func,
        correctness_reward_func,
        choice_number_reward_func,
        advantage_reward_func
    ],
    args=training_args,
    train_dataset=processed_dataset,
)

trainer.train(resume_from_checkpoint=False)
