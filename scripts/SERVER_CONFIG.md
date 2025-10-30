# Server Configuration Guide

## Your Server Specs
- **RAM**: 125 GB
- **CPU**: 16 cores (AMD EPYC-Rome Processor)
- **Platform**: Vultr cloud instance

---

## Recommended Parallel Configuration

### **Strategy: Maximum Throughput (RECOMMENDED)**

With 125 GB RAM, you're **CPU-bound, not RAM-bound**. Run as many jobs as you have cores!

```bash
# Edit scripts/parallel_optimization.sh
MAX_PARALLEL_JOBS=16  # Full CPU utilization!
```

**Performance Estimate**:
- **16 symbols**: All run simultaneously → **~90 minutes total**
- **10 symbols**: All run simultaneously → **~90 minutes total**
- **32 symbols**: First 16 run, then next 16 → **~180 minutes total**

---

## Resource Usage Breakdown

### Per Optimization Job
```
RAM per job:     ~1.5-2 GB
CPU per job:     ~1 core
Time per job:    ~60-90 minutes
```

### With 16 Parallel Jobs
```
Total RAM used:  ~24-32 GB (only 25% of your 125 GB!)
Total CPU used:  ~16 cores (100% utilization)
RAM headroom:    ~93 GB free (plenty!)
```

**Verdict**: Your RAM is WAY oversized for this workload. CPU is the bottleneck.

---

## Configuration Options

### Option 1: Maximum Parallelism (RECOMMENDED)
```bash
MAX_PARALLEL_JOBS=16
```
**Pros**:
- Full CPU utilization
- Process 16 symbols simultaneously
- Simple configuration

**Cons**:
- None! You have the resources.

### Option 2: Conservative (if running other workloads)
```bash
MAX_PARALLEL_JOBS=12
```
**Pros**:
- Leaves 4 cores free for other tasks
- Still very fast

**Cons**:
- Slightly longer for >12 symbols

### Option 3: Ultra-Conservative (not recommended for you)
```bash
MAX_PARALLEL_JOBS=8
```
**Only use if**: You have other heavy processes running

---

## How to Configure

### Method 1: Edit the script directly
```bash
# On your Ubuntu server:
cd /opt/pine/rooney-capital-v1
nano scripts/parallel_optimization.sh

# Change this line (currently line 7):
MAX_PARALLEL_JOBS=6  # Change to 16

# Save and exit (Ctrl+X, Y, Enter)
```

### Method 2: Override at runtime
```bash
# Run with custom parallel jobs:
MAX_PARALLEL_JOBS=16 ./scripts/parallel_optimization.sh
```

---

## Performance Comparison

| Parallel Jobs | Time for 16 Symbols | RAM Used | CPU Used |
|--------------|---------------------|----------|----------|
| 4 | ~360 min (6 hours) | ~8 GB | 25% |
| 8 | ~180 min (3 hours) | ~16 GB | 50% |
| 12 | ~120 min (2 hours) | ~24 GB | 75% |
| **16** | **~90 min (1.5 hours)** | **~32 GB** | **100%** ✅ |

---

## Alternative: Random Forest Internal Parallelism

If you want to optimize fewer symbols but make each one faster, you can enable Random Forest's internal parallelism:

```python
# In rf_cpcv_random_then_bo.py, Random Forest uses n_jobs parameter
# Currently defaults to -1 (uses all cores for RF fitting)

# Trade-off options:
# Option A: 16 parallel jobs × n_jobs=1 per RF = 16 symbols at once
# Option B: 8 parallel jobs × n_jobs=2 per RF = faster per symbol
# Option C: 4 parallel jobs × n_jobs=4 per RF = even faster per symbol
```

**For your use case (likely optimizing 10-20 symbols):**
- **Option A (16 parallel)** is best → Process all symbols simultaneously

---

## Final Recommendation

```bash
# Edit scripts/parallel_optimization.sh line 7:
MAX_PARALLEL_JOBS=16

# Your system can easily handle this!
```

This will:
- ✅ Use 100% of your CPU (efficient!)
- ✅ Use only ~25% of your RAM (plenty of headroom)
- ✅ Process up to 16 symbols simultaneously
- ✅ Complete 16 symbols in ~90 minutes

**Your server is perfect for this workload!** 🚀
