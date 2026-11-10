# GPU Memory Zeroization: Empirical Evaluation, Robustness, and Verification

**Research Implementation | Experimental Analysis | ICACSDF 2026**

This repository contains the CUDA kernel implementations, experimental configurations, binary-analysis scripts, validation tools, datasets, and supporting documentation developed for the research study **GPU Memory Zeroization: Empirical Evaluation, Robustness, and Verification**.

**Authors:** Tripti Sharma, Arshad Ali, Guruansh Singh, Deepali Kamthania

**Presented at:** International Conference on Advances in Cyber Security and Digital Forensics (ICACSDF 2026)

**Venue:** UPES, Dehradun, India

**Conference Website:** [ICACSDF 2026](https://www.icacsdf.org/)

**Keywords:** GPU Security · CUDA · Memory Zeroization · Compiler Optimization · Binary Analysis · Structural Verification · Digital Forensics

---

## Overview

GPU memory zeroization is an important security mechanism intended to remove sensitive data from memory after computation. However, source-level zeroization operations do not necessarily survive compiler optimization or execute as intended in the final GPU binary.

This research empirically evaluates the robustness of GPU memory zeroization across different CUDA compilation environments and investigates the limitations of verifying zeroization through static binary analysis.

The study examines how different zeroization techniques behave under compiler optimization, whether their structural signatures remain observable in compiled machine code, and how reliably those signatures can be classified.

The research also evaluates cross-tool agreement, compilation reproducibility, controlled validation cases, and bounded runtime behavior.

## Research Objectives

- Evaluate the compilation survivability of seven GPU memory zeroization constructs.
- Investigate how kernel context, compiler optimization, CUDA toolkit version, and GPU architecture affect zeroization signatures.
- Analyze compiled CUDA binaries using NVIDIA disassembly utilities.
- Develop a structural classification system for zeroization evidence.
- Compare independent binary-analysis paths for consistency.
- Identify limitations of static structural verification.
- Evaluate reproducibility and selected runtime behavior.

## Architecture

The following section is reserved for the experimental architecture diagram.

<!-- Add the architecture image to figures/research_pipeline.png and uncomment the line below. -->

<!-- ![GPU Memory Zeroization Research Architecture](figures/research_pipeline.png) -->

## Experimental Methodology

The core experiment evaluates **2,016 CUDA binary configurations** across the following dimensions:

| Experimental Dimension | Configuration |
|---|---|
| Zeroization constructs | 7 (Z0–Z6) |
| Kernel contexts | 6 (K0–K5) |
| Optimization levels | 4 |
| GPU architecture targets | 6 |
| CUDA toolkit versions | 2 |
| Total core configurations | 2,016 |

### Zeroization Constructs

| ID | Description |
|---|---|
| Z0 | No-scrub baseline |
| Z1 | Plain zeroization loop |
| Z2 | Volatile zeroization loop |
| Z3 | Plain zeroization with synchronization barrier |
| Z4 | Volatile zeroization with synchronization barrier |
| Z5 | Hardened LMHC-style zeroization |
| Z6 | Inline-PTX hardened zeroization |

### Kernel Contexts

| ID | Description |
|---|---|
| K0 | Pure allocation |
| K1 | Fill followed by scrub |
| K2 | Fill, use, and scrub |
| K3 | Reduction-like computation |
| K4 | Tiled GEMM-like computation |
| K5 | Attention-style computation |

### Compilation Environment

The study evaluates:

- **CUDA toolkits:** 12.8 and 13.3
- **Optimization levels:** `-O0`, `-O1`, `-O2`, `-O3`
- **Architecture targets:** `sm_75`, `sm_80`, `sm_86`, `sm_87`, `sm_89`, `sm_90`
- **Binary analysis:** `cuobjdump` and `nvdisasm`
- **Execution environment:** NVIDIA CUDA with WSL2

Compilation across an architecture target does not imply runtime execution on that GPU architecture.

## Binary Analysis and Structural Classification

Compiled CUDA binaries are analyzed through two disassembly and parsing paths:

**Path A — cuobjdump**

Uses `cuobjdump -sass` to extract SASS instructions for structural feature analysis.

**Path B — nvdisasm**

Uses `nvdisasm -c -json` to obtain an alternative representation of compiled instructions.

The extracted evidence is used to identify structural characteristics associated with zeroization, including zero-store, fence, and barrier signatures.

Each artifact is assigned one of five categories:

| Category | Meaning |
|---|---|
| `FULL_SIGNATURE` | Complete recognized structural signature |
| `PARTIAL_SIGNATURE` | Some, but not all, recognized structural components |
| `STORE_ONLY` | Recognized zero-store evidence without the complete signature |
| `ABSENT` | No recognized zeroization signature |
| `UNDECIDABLE` | Insufficient evidence for a reliable classification |

These categories describe observable binary structure rather than a formal proof of secure memory erasure.

## Key Experimental Results

### 1. Compilation and Structural Classification

All **2,016 core configurations compiled successfully**.

| Classification | Number of Artifacts |
|---|---:|
| `FULL_SIGNATURE` | 576 |
| `PARTIAL_SIGNATURE` | 864 |
| `STORE_ONLY` | 96 |
| `ABSENT` | 480 |
| `UNDECIDABLE` | 0 |
| **Total** | **2,016** |

The hardened Z5 and Z6 constructs retained their complete recognized structural signatures in all 288 configurations evaluated for each construct.

### 2. Cross-Tool Agreement

The `cuobjdump`-based and `nvdisasm`-based analysis paths agreed on all eight extracted features and structural classifications across the 2,016 core artifacts.

This demonstrates consistency between the implemented analysis paths but does not establish independent semantic correctness.

### 3. Controlled Validation

A controlled validation corpus containing **13 structural cases (N0–N12)** was used to evaluate classification behavior and verification boundaries.

The corpus included 55 configurations, of which 47 matched their intended structural conditions.

A significant finding was that the presence of a recognizable zeroization signature does not necessarily prove that cleanup executes along every possible program path.

### 4. Reproducibility

Fifty stratified configurations were selected for additional compilation runs.

Across 150 repeat compilations, the evaluated binary and disassembly outputs were byte-identical to their corresponding original outputs.

These findings apply to the tested environment and toolchain configuration.

### 5. Runtime Sanity Checks

Bounded runtime diagnostics were conducted using an **NVIDIA RTX 4050 Laptop GPU (`sm_89`)**.

Eighteen instrumented diagnostic variants were evaluated across five repetitions each, producing 90 successful launches.

The observed results were consistent with the expected behavior of the tested scrub-bearing and no-scrub variants at the instrumented observation point.

These tests do not establish general absence of recoverable GPU memory residue.

## Repository Structure

```text
gpu-zeroization-robustness/
├── configs/                 # Experiment configurations
├── kernels/
│   ├── generated/           # Generated CUDA kernels
│   ├── metadata/            # Kernel metadata
│   ├── controlled/          # Controlled validation cases
│   └── runtime_sanity/      # Instrumented runtime kernels
├── scripts/                 # Generation and analysis scripts
├── tests/                   # Automated Python tests
├── validation_tools/        # Supporting validation utilities
├── results/                 # Experimental results and summaries
├── figures/                 # Research visualizations
├── docs/                    # Detailed research documentation
├── .gitignore
└── README.md
```

Large generated compilation artifacts and selected review materials are retained outside the Git-tracked repository contents. Private review mappings are excluded from publication.

## Getting Started

### Prerequisites

The complete compilation workflow requires a compatible NVIDIA CUDA development environment, including the toolkit versions used in the study.

The analysis pipeline additionally uses NVIDIA binary inspection utilities such as `cuobjdump` and `nvdisasm`.

Python is required for the experiment-generation, analysis, and validation scripts.

### Clone the Repository

```bash
git clone https://github.com/triptiiSharma/gpu-zeroization-robustness.git
cd gpu-zeroization-robustness
```

*The repository URL above is the intended publication location; it will work after the repository has been created and pushed.*

### Run Automated Tests

Install `pytest` in your Python environment:

```bash
python -m pip install pytest
```

Then run:

```bash
python -m pytest tests -q
```

The local test suite was last verified with **49 passing tests and 12 passing subtests**.

### Reproducing the Experiments

The experiment configuration files, CUDA kernel sources, and supporting Python scripts are included in the repository.

Detailed methodology and validation information are available in the documentation linked below.

Reproducing the complete compilation matrix requires the appropriate CUDA toolchains and experimental environment. Running the Python test suite alone does not reproduce the full research experiment.

## Documentation

| Document | Description |
|---|---|
| [Research Plan](docs/PLAN.md) | Experimental methodology and research phases |
| [Environment Audit](docs/AUDIT.md) | Initial environment and feasibility investigation |
| [Data Dictionary](docs/DATA_DICTIONARY.md) | Dataset fields and classification terminology |
| [Validation Report](docs/VALIDATION_REPORT.md) | Validation results and checks |
| [Final Results Ledger](docs/FINAL_RESULTS_LEDGER.md) | Consolidated experimental findings |
| [Cross-Tool Validation](docs/PHASE5_CROSS_TOOL_VALIDATION.md) | Independent disassembly comparison and validation results |
| [Reproducibility](docs/PHASE5_REPRODUCIBILITY.md) | Repeat compilation and reproducibility checks |
| [Structural Reliability](docs/PHASE6_STRUCTURAL_RELIABILITY.md) | Structural verification reliability and boundaries |

## Limitations

The study evaluates the compilation and structural verification of GPU memory zeroization. Its findings should not be interpreted as a universal guarantee of secure erasure.

Important limitations include:

- Static structural signatures do not prove that cleanup executes on every possible runtime path.
- No complete control-flow post-dominance proof is performed.
- Agreement between binary-analysis tools does not constitute independent semantic ground truth.
- Most architecture targets were evaluated through compilation and static analysis rather than execution on corresponding physical GPUs.
- Runtime experiments were bounded and used instrumented diagnostic kernels.
- The study does not establish the complete absence of recoverable GPU memory residue.

## Publication

**Paper:** GPU Memory Zeroization: Empirical Evaluation, Robustness, and Verification

**Authors:** Tripti Sharma, Arshad Ali, Guruansh Singh, Deepali Kamthania

**Conference:** International Conference on Advances in Cyber Security and Digital Forensics (ICACSDF 2026)

**Location:** UPES, Dehradun, India

**Website:** https://www.icacsdf.org/

The conference paper is not distributed through this repository. An official publication link or DOI may be added when available.

---

**Research repository accompanying the work presented at ICACSDF 2026.**
