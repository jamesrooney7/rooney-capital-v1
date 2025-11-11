# GitHub → Server Workflow

**Last Updated:** 2025-11-11

---

## 🎯 Key Principle: GitHub is Source of Truth

**All code lives in GitHub.** The server is just where code **runs**, not where it's edited or stored permanently.

---

## 📋 Workflow

### 1. Code Development (in GitHub)
```bash
# Claude makes changes and commits to branch
git commit -m "Fix feature X"
git push -u origin claude/feature-branch
```

### 2. Pull to Server (you run on server)
```bash
# SSH into your server
cd /home/user/rooney-capital-v1

# Check current branch
git branch

# Pull latest changes
git pull origin <branch-name>

# Or switch to a different branch
git fetch origin
git checkout <branch-name>
git pull origin <branch-name>
```

### 3. Run on Server
```bash
# Activate environment (if needed)
source venv/bin/activate  # or conda activate rooney-capital

# Run training scripts
python research/train_rf_three_way_split.py --symbol ES --bo-trials 50

# Run backtests
python research/generate_portfolio_backtest_data.py --symbol NG --start-date 2023-01-01

# Run portfolio optimizer
python research/portfolio_optimizer_greedy_train_test.py
```

### 4. Results Stay on Server
```bash
# Results are generated on server but NOT committed to GitHub
ls results/ES_optimization/ES_rf_model.pkl  # ← Server only
ls results/NG_optimization/NG_rf_best_trades.csv  # ← Server only
```

---

## 📂 What Lives Where

### ✅ **IN GITHUB** (Tracked by Git)

**Source Code:**
```
src/                    - Production source code
  ├── config/           - Configuration modules
  ├── data/             - Data loading utilities
  ├── indicators/       - Technical indicators
  ├── models/           - Model loading utilities
  ├── services/         - Core services (execution, monitoring)
  └── strategy/         - Trading strategies (IbsStrategy)

research/               - Research & optimization scripts
  ├── extract_training_data.py
  ├── train_rf_three_way_split.py
  ├── generate_portfolio_backtest_data.py
  ├── portfolio_optimizer_greedy_train_test.py
  └── archive/          - Old scripts (reference only)

deployment/             - Production deployment code
scripts/                - Utility scripts
dashboard/              - Dashboard code (if applicable)
```

**Documentation:**
```
*.md files              - All markdown documentation
  ├── README.md
  ├── QUICK_START.md
  ├── THREE_WAY_SPLIT_GUIDE.md
  ├── END_TO_END_OPTIMIZATION_GUIDE.md
  ├── SYSTEM_GUIDE.md
  └── docs/             - Additional documentation
```

**Configuration Examples:**
```
Data/
  └── Databento_contract_map.yml  - Contract specifications

deployment/env/
  └── *.env.example     - Environment variable templates (NO SECRETS)
```

---

### 🚫 **SERVER ONLY** (Gitignored - Never in GitHub)

**1. Environment & Secrets** 🔐
```
.env                    - Environment variables (API keys, credentials)
config.yml              - Runtime configuration
*.env files             - Any environment files
```

**2. Data Files** 📊
```
data/bars/              - Historical price data (CSV/Parquet)
  ├── ES_hourly.csv     - ~15 years of hourly bars per symbol
  ├── ES_daily.csv      - ~15 years of daily bars per symbol
  ├── NG_hourly.csv
  └── ... (18 symbols)

*.csv                   - Any CSV data files
*.parquet               - Parquet data files
*.db                    - Database files
```

**3. Training Outputs** 🤖
```
results/                - ALL results and trained models
  ├── {SYMBOL}_optimization/
  │   ├── {SYMBOL}_rf_model.pkl           - Trained Random Forest model
  │   ├── {SYMBOL}_rf_metadata.json       - Model metadata (features, threshold)
  │   ├── {SYMBOL}_transformed_features.csv  - Training data (~15k trades)
  │   ├── {SYMBOL}_rf_best_trades.csv     - Best model filtered trades
  │   ├── training_log.txt                - Training logs
  │   └── ... (various optimization outputs)
  └── portfolio/
      └── ... (portfolio optimization results)
```

**4. Logs** 📝
```
*.log                   - All log files
logs/                   - Log directory
/var/log/rooney/        - Service logs (production)
```

**5. Python Artifacts** 🐍
```
__pycache__/            - Python bytecode cache
*.pyc, *.pyo            - Compiled Python files
venv/, .venv/           - Virtual environments
*.egg-info/             - Package metadata
```

---

## 🔄 Common Operations

### Check Which Branch You're On
```bash
git branch                    # Shows current branch with *
git status                    # Shows branch + any uncommitted changes
```

### Pull Latest Code
```bash
# If on the right branch already
git pull origin <branch-name>

# If need to switch branches first
git fetch origin
git checkout <branch-name>
git pull origin <branch-name>
```

### See Recent Commits
```bash
git log --oneline -10         # Last 10 commits
git log --oneline --graph     # Visual commit history
```

### Check for Updates Without Pulling
```bash
git fetch origin              # Download latest from GitHub
git log HEAD..origin/<branch-name>  # See what's new
```

---

## ⚠️ Important Notes

### DO NOT Edit Code on Server
- **Always edit code via Claude → GitHub**
- Server is **read-only** for code (only run it, don't edit)
- If you manually edit on server, changes will be lost on next `git pull`

### DO NOT Commit Large Files
The following should **NEVER** be committed to GitHub:
- ❌ `.env` files (contains secrets)
- ❌ CSV/Parquet data files (too large)
- ❌ Trained model `.pkl` files (generated, not source)
- ❌ `results/` directory (generated outputs)
- ❌ Log files

### Results are Ephemeral
- Training outputs (`results/`) are generated on server
- If you need to share results, use metrics/summaries, not raw files
- Models can be regenerated by re-running training scripts
- Keep data backups separate from code repository

---

## 📊 Data File Sizes (Approximate)

**Why data files aren't in GitHub:**
```
data/bars/ES_hourly.csv     ~50 MB   (15 years × 24 hours × 250 days)
data/bars/ES_daily.csv      ~2 MB    (15 years × 250 days)

18 symbols × ~52 MB each    ~936 MB  (Total raw data)

results/ directory          ~500 MB+ (Models + feature CSVs)

TOTAL data on server:       ~1.5 GB+ (Too large for GitHub)
```

GitHub has a 100 MB single file limit and repositories should ideally stay under 1 GB.

---

## 🔍 When Things Get Confusing

### "Where is file X?"

**Ask yourself:**
1. **Is it code/documentation?** → GitHub (versioned)
2. **Is it data/results/secrets?** → Server only (gitignored)

### "Why can't Claude see my results?"

Claude can only see:
- ✅ Files tracked in GitHub (code, docs)
- ✅ Files you explicitly show via terminal commands on server

Claude cannot directly see:
- ❌ Data files on your server
- ❌ Results in `results/` directory
- ❌ Your `.env` file

**Solution:** Run commands on server and paste output to Claude:
```bash
# Show results
ls -lh results/NG_optimization/

# Show CSV summary
head -5 results/NG_optimization/NG_rf_best_trades.csv

# Show metrics
tail -n +2 results/NG_optimization/NG_rf_best_trades.csv | awk -F',' '{sum+=$7} END {print sum}'
```

### "Which branch should I use?"

Check what Claude is working on:
```bash
git branch          # Shows current branch
git log --oneline   # Shows recent commits
```

Usually Claude will tell you which branch to checkout.

---

## 🚀 Complete Example Workflow

```bash
# 1. Claude fixes a bug and commits to GitHub
# (happens automatically via Claude)

# 2. You pull the fix to your server
cd /home/user/rooney-capital-v1
git fetch origin
git checkout claude/fix-feature-X
git pull origin claude/fix-feature-X

# 3. You run the fixed code on your server
python research/train_rf_three_way_split.py --symbol ES --bo-trials 50

# 4. Results are generated on server (not in GitHub)
ls results/ES_optimization/ES_rf_model.pkl  # ✅ Exists on server

# 5. You share results with Claude via terminal output
tail -20 results/ES_optimization/training_log.txt

# 6. If training is successful, you might merge the branch
git checkout main
git merge claude/fix-feature-X
git push origin main
```

---

## 📝 Summary

| Item | Location | Synced to GitHub? |
|------|----------|-------------------|
| Source code (`.py`) | GitHub → Server | ✅ Yes |
| Documentation (`.md`) | GitHub → Server | ✅ Yes |
| Configuration examples | GitHub → Server | ✅ Yes |
| `.env` (secrets) | Server only | ❌ Never |
| Data files (CSV/Parquet) | Server only | ❌ No (too large) |
| Trained models (`.pkl`) | Server only | ❌ No (generated) |
| Results directory | Server only | ❌ No (generated) |
| Logs | Server only | ❌ No (noise) |

**Remember:** Code flows **GitHub → Server**, results stay **Server only**.

---

**Last Updated:** 2025-11-11 (After repository cleanup)
