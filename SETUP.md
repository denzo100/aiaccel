# Setup & Testing Guide for Eyeriss Accelerator Simulation

## Prerequisites

### System Requirements
- **OS:** Linux (Ubuntu 18.04+, Debian, or similar) or macOS
- **Python:** 3.8 or higher
- **RAM:** 4GB minimum (8GB recommended)
- **Disk Space:** 2GB for dependencies + simulation outputs

### Install Timeloop & Accelergy

#### Option A: Using Conda (Recommended)
```bash
# Create environment
conda create -n eyeriss python=3.8
conda activate eyeriss

# Install Timeloop and Accelergy
pip install timeloop-accelergy accelergy pyyaml
```

#### Option B: Build from Source
```bash
# Clone Timeloop repository
git clone https://github.com/NVlabs/timeloop.git
cd timeloop

# Follow build instructions in README
# Requires: cmake, g++, boost libraries
```

#### Verify Installation
```bash
timeloop-mapper --version
accelergy --version
```

---

## Running Simulations

### Step 1: Clone This Repository
```bash
git clone https://github.com/denzo100/aiaccel.git
cd aiaccel
```

### Step 2: Run 16×16 Array Simulation
```bash
# Create output directory
mkdir -p output_16x16

# Run mapper
timeloop-mapper \
  -c arch/eyeriss_16x16.yaml \
  -d prob/bert_768x768.yaml \
  -m mapper/mapper.yaml \
  -o output_16x16/

# Expected runtime: 2-5 minutes
```

### Step 3: Analyze Results
```bash
# View statistics
cat output_16x16/timeloop-mapper.stats.txt

# Expected output metrics:
# - Total cycles
# - Runtime
# - Energy consumption
# - Memory traffic
```

### Step 4: Run 32×32 and 64×64 (Analytical Scaling)
```bash
# For scaled results, modify arch configuration:
# Edit mapper/mapper.yaml and update array dimensions

# Or use our pre-computed scaling:
cat partA_working/RESULTS_SUMMARY.md
```

---

## Understanding the Output Files

### Architecture Files (`arch/`)
- `eyeriss_16x16.yaml` - 16×16 PE array configuration
- Defines: PE types, memory hierarchy, interconnect, energy models

### Problem Files (`prob/`)
- `bert_768x768.yaml` - BERT attention workload (FP16)
- Defines: Tensor dimensions, data types, loop structures

### Mapper Files (`mapper/`)
- `mapper.yaml` - Mapping constraints and directives
- Defines: Loop order, tiling strategy, memory allocation

### Output Files (`output_*/`)
- `timeloop-mapper.stats.txt` - Performance metrics
- `timeloop-mapper.ERT.yaml` - Energy roofline model
- `timeloop-mapper.ART.yaml` - Area roofline model

---

## Analysis Files

Read the pre-computed analyses in order:

### Part A: Simulation Results
```bash
cat partA_working/RESULTS_SUMMARY.md
```
**Contains:** Energy, throughput, and power metrics for 16×16, 32×32, 64×64

### Part B: Precision Comparison
```bash
cat partB_analysis/PRECISION_ANALYSIS.md
```
**Contains:** INT8 vs FP16 trade-offs (50% energy reduction)

### Part C: Roofline Model
```bash
cat partC_analysis/ROOFLINE_MODEL.md
```
**Contains:** Memory wall analysis, bandwidth utilization, bottleneck identification

### Part D: GPU Baseline
```bash
cat partD_analysis/GPU_COMPARISON.md
```
**Contains:** Eyeriss vs V100 comparison (edge vs datacenter)

### Part E: Power & Thermal
```bash
cat partE_analysis/POWER_THERMAL.md
```
**Contains:** Power breakdown, thermal analysis, thermal management

---

## Customization: Run Your Own Workload

### Example: Modify Problem Size
Edit `prob/bert_768x768.yaml`:
```yaml
# Change matrix dimensions
problem:
  shape:
    R: 768      # Change this value
    S: 768      # Change this value
    C: 768
    K: 768
    N: 1
```

Then run:
```bash
timeloop-mapper \
  -c arch/eyeriss_16x16.yaml \
  -d prob/bert_768x768.yaml \
  -m mapper/mapper.yaml \
  -o output_custom/
```

### Example: Modify Frequency
Edit `arch/eyeriss_16x16.yaml`:
```yaml
core:
  frequency: 1000    # MHz (change from 1000 to 500 for lower frequency)
```

---

## Troubleshooting

### "Command not found: timeloop-mapper"
**Solution:** Activate conda environment:
```bash
conda activate eyeriss
```

### "YAML file not found"
**Solution:** Make sure you're in the correct directory:
```bash
cd aiaccel
ls arch/eyeriss_16x16.yaml  # Should return the file path
```

### "Simulation hangs or takes >10 minutes"
**Solution:** 
- Check available RAM: `free -h`
- Try smaller array size first (8×8)
- Reduce problem size temporarily

### "Output files are empty"
**Solution:**
- Check for errors in the log files
- Verify YAML syntax: `python -m yaml < prob/bert_768x768.yaml`

---

## Verification Checklist

After running simulations, verify:

- [ ] `output_16x16/timeloop-mapper.stats.txt` exists (non-empty)
- [ ] Cycles ≈ 294,912 (from Part A)
- [ ] Energy ≈ 9.83 mJ (from Part A)
- [ ] Power ≈ 33.3 W (from Part E)
- [ ] All 5 analysis markdown files are readable
- [ ] Can compare your results against Part A summary

---

## Next Steps

1. **Run baseline 16×16 simulation** (verify setup works)
2. **Compare against Part A results** (should match closely)
3. **Read all analysis documents** (understand the findings)
4. **Try modifications** (change array size, frequency, precision)
5. **Share results with group** (document any differences)

---

## Questions?

- Check Timeloop documentation: https://github.com/NVlabs/timeloop
- Review Accelergy docs: https://github.com/NVlabs/accelergy
- Ask in the repository issues: https://github.com/denzo100/aiaccel/issues

---

## Performance Tips

- **Faster simulation:** Use smaller arrays (8×8) for quick testing
- **Parallel runs:** Run multiple array sizes in separate terminals
- **Energy accuracy:** Ensure Accelergy is properly installed (provides realistic energy models)
- **Debugging:** Add `-v` flag for verbose output: `timeloop-mapper -v ...`
