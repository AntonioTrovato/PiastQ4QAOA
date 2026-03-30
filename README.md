# Quantum Test Case Selection - Circuits Extraction and Execution

This project implements and evaluates three quantum approaches for Test Case Selection:

- **Multi-objective QAOA-TCS**
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
python multi_obj.py [-l]
python single_obj.py [-l]
```

---

### 3.2 IGDec-QAOA

Move to the folder:

```bash
cd ../igdec_qaoa
```

Run:

```bash
python loch_qaoa_tcm_extract_circuits.py [-l]
python loch_qaoa_elev_two_extract_circuits.py [-l]
python loch_qaoa_elev_three_extract_circuits.py [-l]
```

---

## 4. Lightweight Mode (`-l`)

The `-l` flag enables a **lightweight execution mode**:

- **QAOA-TCS**: runs only with `reps = 1`
- **IGDec-QAOA**: uses only `sampling_1` and runs 30 iterations only on that sampling

---

## 5. Overall Pipeline

Each script:

Reloads the saved optimized quantum circuits and executed with batching constraints:

| Parameter | Value |
|-----------|-------|
| Total shots | `2048 × 30 = 61440` |
| Batches | `307 × 200 shots` + `1 × 40 shots` |

---

## 6. Quick Full Execution Example

**Full run:**

```bash
python3.10 -m venv qiskit_env
source qiskit_env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

cd qaoa_tcs
python multi_obj.py
python single_obj.py

cd ../igdec_qaoa
python loch_qaoa_tcm_extract_circuits.py
python loch_qaoa_elev_two_extract_circuits.py
python loch_qaoa_elev_three_extract_circuits.py
```

**Lightweight version:**

```bash
cd qaoa_tcs
python multi_obj.py -l
python single_obj.py -l

cd ../igdec_qaoa
python loch_qaoa_tcm_extract_circuits.py -l
python loch_qaoa_elev_two_extract_circuits.py -l
python loch_qaoa_elev_three_extract_circuits.py -l
```