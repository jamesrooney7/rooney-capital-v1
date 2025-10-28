# 2-Year Chunk Extraction Workflow

This document describes the memory-efficient approach for extracting training data from 2010-2024.

## Problem

Extracting 14 years of data (2010-2024) for 18 symbols uses too much RAM:
- 16 symbols × 14 years × 1.56 GB/symbol/year ≈ **350+ GB RAM** (not feasible!)

## Solution: 2-Year Rolling Chunks

Extract data in **2-year chunks** with **1 warmup year** + **1 extraction year**:

- **Chunk 1**: Extract 2010-2011 data → Keep only 2011 trades
- **Chunk 2**: Extract 2011-2012 data → Keep only 2012 trades
- **Chunk 3**: Extract 2012-2013 data → Keep only 2013 trades
- ... (continue through 2024)

This ensures:
- ✅ Each year has 1 year of warmup data for indicators/percentiles
- ✅ RAM capped at ~50 GB (16 symbols × 2 years × 1.56 GB)
- ✅ Full parallelism (16 symbols processed simultaneously)

## Workflow

### Step 1: Run Chunked Extraction

```bash
# Run in background (will take several hours)
nohup bash research/extract_2year_chunks.sh > extraction_chunks.log 2>&1 &

# Monitor progress
tail -f extraction_chunks.log

# Watch RAM usage (should stay around 50GB per chunk)
watch -n 30 free -h
```

This will:
- Run 14 time chunks (2011, 2012, ..., 2024)
- Process 16 symbols in parallel per chunk
- Save intermediate files to `data/training_chunks/`
  - Example: `ES_2011.csv`, `ES_2012.csv`, ..., `ES_2024.csv`

**Expected output:** 18 symbols × 14 years = **252 chunk CSV files**

### Step 2: Concatenate Chunks

Once extraction is complete, merge the chunks:

```bash
# Concatenate all chunks into final CSV files
python3 research/concatenate_chunks.py
```

This will:
- Read all chunk files for each symbol (e.g., `ES_2011.csv` through `ES_2024.csv`)
- Concatenate them in chronological order
- Save final training files to `data/training/`
  - Example: `ES_transformed_features.csv`

**Expected output:** 18 final CSV files (one per symbol)

## Directory Structure

```
data/
├── training_chunks/          # Intermediate chunk files
│   ├── ES_2011.csv
│   ├── ES_2012.csv
│   ├── ...
│   ├── ES_2024.csv
│   ├── NQ_2011.csv
│   └── ...
└── training/                 # Final concatenated files
    ├── ES_transformed_features.csv
    ├── NQ_transformed_features.csv
    └── ...

logs/
└── extraction_chunks/        # Per-symbol-year logs
    ├── ES_2011.log
    ├── ES_2012.log
    └── ...
```

## Technical Details

### RAM Usage Per Chunk

```
16 symbols × 2 years × 1.56 GB/symbol/year ≈ 50 GB
```

- Safe for servers with 64+ GB RAM
- Leaves room for OS and other processes

### Year Filtering

The `--filter-year` parameter in `extract_training_data.py`:
- Runs backtest on 2 years of data (e.g., 2010-2011)
- Calculates all indicators with full warmup period
- **Only saves trades with entry in the target year** (e.g., 2011)
- Discards warmup year trades automatically

### Resume Capability

Both scripts check for existing files:
- **Extraction**: Skips chunks that already exist (checks `data/training_chunks/`)
- **Concatenation**: Warns about missing years but continues with available chunks

To re-extract a specific year, delete the chunk file:
```bash
rm data/training_chunks/ES_2015.csv
```

Then re-run `extract_2year_chunks.sh` - it will only process missing chunks.

## Monitoring

### Check Progress

```bash
# How many chunks completed?
ls data/training_chunks/*.csv | wc -l

# Expected: 252 (18 symbols × 14 years)

# Which symbols/years are done?
ls data/training_chunks/ | sort
```

### Check RAM Usage

```bash
# Current RAM
free -h

# Top memory consumers
ps aux --sort=-%mem | head -20
```

### Check Running Processes

```bash
# How many extractions running?
ps aux | grep extract_training_data.py | grep -v grep | wc -l

# Expected: Up to 16 (during extraction)
```

## Troubleshooting

### "No chunk files found" during concatenation

Check if extraction completed:
```bash
ls data/training_chunks/ | grep ES
```

If files are missing, re-run extraction.

### Out of Memory Errors

Reduce parallelism to 8 symbols:
```bash
# Edit extract_2year_chunks.sh
MAX_PARALLEL=8  # Change from 16 to 8
```

This reduces RAM from 50GB → 25GB.

### Missing Years in Output

The concatenation script warns about missing years but continues. Check:
```bash
# Find missing chunks
for symbol in ES NQ RTY YM GC SI HG CL NG PL 6A 6B 6C 6E 6J 6M 6N 6S; do
    for year in {2011..2024}; do
        if [ ! -f "data/training_chunks/${symbol}_${year}.csv" ]; then
            echo "Missing: ${symbol}_${year}.csv"
        fi
    done
done
```

## Cleanup

After successful concatenation, you can delete intermediate chunks to save disk space:

```bash
# ONLY do this after verifying final files are correct!
# This saves ~10-20 GB of disk space

# Verify final files exist
ls data/training/*.csv

# If all 18 symbols present, safe to delete chunks
rm -rf data/training_chunks/
rm -rf logs/extraction_chunks/
```

## Time Estimates

Assuming one symbol for full period takes **T hours**:

- **Per chunk** (2 years): ~0.13T hours
- **14 chunks sequential**: ~1.7T hours total
- **With 16-way parallelism**: Same as single symbol extraction time

Example: If ES takes 2 hours for full 2010-2024:
- Total wallclock time: ~3.5 hours (14 chunks × 0.26 hours/chunk)
- All 18 symbols complete in ~3.5 hours

## Next Steps

After extraction completes:
1. Verify final CSV files in `data/training/`
2. Check row counts are reasonable (~1000-3000 trades per symbol)
3. Proceed with ML training using `rf_cpcv_random_then_bo.py`
