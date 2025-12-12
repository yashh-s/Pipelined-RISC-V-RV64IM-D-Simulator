# Pipelined RV64IM + D-Extension Processor Simulator

This project was developed as part of the **Computer Architecture (CA)** course at IIIT Bangalore.  
Team members: **Yash Sultania** and **Saharsh S Hiremath**

A Python-based simulator implementing a **5-stage pipelined RISC‑V processor** that supports:

- **RV64I** base integer instructions  
- **M-extension** (multiply/divide)  
- **D-extension** (double‑precision floating point)  
- Cycle‑accurate tracing of pipeline stages  
- Hazard detection and forwarding  
- Branch resolution and pipeline flushing  

This project was completed as a hands‑on exploration of CPU microarchitecture, pipelining, ISA parsing, and hazard management.

---

## 🔷 Project Overview

The simulator models a realistic 5‑stage pipeline:

1. **IF** – Instruction Fetch  
2. **ID** – Decode & Register Read  
3. **EX** – ALU / FPU operations  
4. **MEM** – Data memory access  
5. **WB** – Writeback to register file  

It executes RISC‑V programs instruction-by-instruction, printing pipeline state every cycle.

This project demonstrates understanding of:

- Pipeline hazards (data, control)  
- Forwarding paths  
- Stalling and flushing  
- Double‑precision floating-point execution  
- Instruction decoding & register management  

---

## 🔷 Key Features

### ✅ Pipeline Implementation  
- Full 5-stage RISC‑V pipeline  
- Pipeline registers between every stage  
- Register file for integer + floating-point registers  

### ✅ Hazard Handling  
- **Load‑use detection** and automatic stalls  
- **Data forwarding** from EX/MEM and MEM/WB  
- **Branch flush** on taken branches  

### ✅ D‑Extension Floating Point Support  
Includes operations such as:

- FADD.D, FSUB.D, FMUL.D, FDIV.D  
- FCVT instructions  
- FMV instructions  
- FSGNJ, FSGNJN, FSGNJX  
- FEQ.D, FLT.D, FLE.D  

### ✅ Instruction Parsing  
Handles standard RISC‑V syntax such as:

```
LW x1, 0(x2)
ADDI x3, x0, 10
FADD.D f2, f3, f4
```

---

## 🔷 Repository Structure (planned)

```
.
├── main.py                  # Meant to be driver script (However currently is the entire project)
├── simulator/
│   └── processor.py         # Yet to be written (main.py works independently)
├── tests/
│   └── example_prog.txt     # Example instruction sequence (Yet to be added)
├── README.md
├── requirements.txt 
```

Currently everything is inside **main.py**, but the project will later be split into `processor.py` and a cleaner `main.py`.

---

## 🔷 How to Run (simple)

```bash
python3 main.py
```

This will:

- Load the hardcoded instructions  
- Execute the pipeline cycle-by-cycle  
- Print all pipeline registers every cycle  
- Summarize total cycles and completed instructions  

---

## 🔷 Skills Demonstrated

- Pipeline microarchitecture design  
- Hazard detection and resolution  
- Forwarding network design  
- Floating‑point datapath concepts  
- ISA parsing and simulation  
- Python system‑level programming  
- Debugging complex state machines  

---

## 👥 Contributors

**Yash Sultania**  
**Saharsh S Hiremath**  
*(Computer Architecture Course Project, IIIT Bangalore)*

---

