# Quantum Test Case Selection - Circuits Extraction and Execution

This project implements and evaluates two quantum approaches for Test Case Selection:

- **Single-objective QAOA-TCS**
- **IGDec-QAOA**

The core idea is to:

1. Extract the **trained circuits**
2. Re-execute them in a **real setting** 

---

## 1. Requirements

You must use:

- **Python 3.10.\***  

---

## 2. Environment Setup

From the project root:

```bash
python3.10 -m venv qiskit_env
source qiskit_env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 3. How to Run the Project

### 3.1 QAOA-TCS

Move to the folder:

```bash
cd qaoa_tcs
```

Run:

```bash
python single_obj.py
```

---

### 3.2 IGDec-QAOA

Move to the folder:

```bash
cd ../igdec_qaoa
```

Run:

```bash
python loch_qaoa_tcm_extract_circuits.py
```