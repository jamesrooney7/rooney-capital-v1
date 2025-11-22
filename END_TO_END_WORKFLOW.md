# End-to-End Workflow: Research to Production

Complete guide to the Rooney Capital trading system workflow from strategy research through live deployment.

---

## Table of Contents

- [System Overview](#system-overview)
- [Complete Workflow Diagram](#complete-workflow-diagram)
- [Phase 1: Strategy Research](#phase-1-strategy-research)
- [Phase 2: ML Meta-Labeling](#phase-2-ml-meta-labeling)
- [Phase 3: Portfolio Optimization](#phase-3-portfolio-optimization)
- [Phase 4: Live Deployment](#phase-4-live-deployment)
- [Data Locations](#data-locations)
- [What Runs Where](#what-runs-where)
- [Development vs Production](#development-vs-production)
- [Complete Example: Adding a New Instrument](#complete-example-adding-a-new-instrument)
- [Maintenance Workflows](#maintenance-workflows)
- [Troubleshooting](#troubleshooting)

---

## System Overview

The Rooney Capital system is a **multi-stage quantitative trading pipeline**:

1. **Strategy Factory** → Systematically discover profitable base strategies
2. **ML Meta-Labeling** → Train ensemble models to filter signals
3. **Portfolio Optimization** → Select optimal strategy subset with risk constraints
4. **Live Execution** → Deploy to production with real-time data and order routing

**Key Philosophy**:
- **Local development**: Research, backtesting, ML training, portfolio optimization
- **Server deployment**: Live trading worker, monitoring dashboard, data storage
- **Git repository**: Code and models (via Git LFS), NOT data or credentials
- **Systematic validation**: Every stage has quality checks before proceeding

---

## Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RESEARCH (Local Machine)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase 1: STRATEGY RESEARCH                                                 │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ 1. Download historical data (Databento)                            │    │
│  │    → data/databento/{SYMBOL}_daily.csv                             │    │
│  │    → data/databento/{SYMBOL}_hourly.csv                            │    │
│  │                                                                     │    │
│  │ 2. Run Strategy Factory optimization                               │    │
│  │    → research/strategy_factory/optimize_strategies.py              │    │
│  │    → Output: research/strategy_factory/results/{SYMBOL}_*.json     │    │
│  │                                                                     │    │
│  │ 3. Select best strategy variant per instrument                     │    │
│  │    → Analyze Sharpe, robustness, drawdown                          │    │
│  │    → Document optimal parameters                                   │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                  ↓                                           │
│  Phase 2: ML META-LABELING                                                  │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ 4. Extract training data with optimal strategy params              │    │
│  │    → research/extract_training_data.py --symbol ES                 │    │
│  │    → Output: data/training/ES_training_data.csv (30+ features)     │    │
│  │                                                                     │    │
│  │ 5. Train ML meta-labeling model (3-way temporal split)             │    │
│  │    → research/ml_meta_labeling/train_rf_three_way_split.py         │    │
│  │    → Output: src/models/ES_rf_model.pkl (Git LFS)                  │    │
│  │    → Output: src/models/ES_best.json (Git)                         │    │
│  │                                                                     │    │
│  │ 6. Validate model performance on test set (2021-2024)              │    │
│  │    → Check test_metrics in ES_best.json                            │    │
│  │    → Ensure Sharpe > 1.2, DSR > 1.0                                │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                  ↓                                           │
│  Phase 3: PORTFOLIO OPTIMIZATION                                            │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ 7. Generate per-strategy equity curves                             │    │
│  │    → research/generate_portfolio_backtest_data.py                  │    │
│  │    → Output: data/portfolio/{SYMBOL}_{VARIANT}_equity.csv          │    │
│  │                                                                     │    │
│  │ 8. Run greedy portfolio optimizer                                  │    │
│  │    → research/portfolio_optimization/greedy_optimizer.py           │    │
│  │    → Constraints: max 2 positions, max correlation 0.7             │    │
│  │    → Output: Portfolio config (instruments + sizes)                │    │
│  │                                                                     │    │
│  │ 9. Update config.yml with selected instruments                     │    │
│  │    → Set symbols, sizes, commission overrides                      │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                  ↓
                    GIT COMMIT & PUSH (models + config)
                                  ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PRODUCTION (Server: /opt/pine/)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase 4: LIVE DEPLOYMENT                                                   │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ 10. Pull latest code on server                                     │    │
│  │     → cd /opt/pine/rooney-capital-v1                               │    │
│  │     → git pull origin main                                         │    │
│  │     → git lfs pull  # Download model files                         │    │
│  │                                                                     │    │
│  │ 11. Update server configuration                                    │    │
│  │     → /opt/pine/runtime/config.yml                                 │    │
│  │     → /opt/pine/runtime/.env (credentials)                         │    │
│  │                                                                     │    │
│  │ 12. Restart trading services                                       │    │
│  │     → sudo systemctl restart pine-runner.service                   │    │
│  │     → sudo systemctl restart rooney-dashboard.service              │    │
│  │                                                                     │    │
│  │ 13. Validate deployment                                            │    │
│  │     → journalctl -u pine-runner.service -f                         │    │
│  │     → Check "Indicator warmup completed" for all instruments       │    │
│  │     → Verify ML models loaded: 100% feature coverage               │    │
│  │     → Monitor first hour for proper signal generation              │    │
│  │                                                                     │    │
│  │ 14. Monitor live trading                                           │    │
│  │     → Dashboard: http://server:8501                                │    │
│  │     → TradersPost: Check positions and fills                       │    │
│  │     → Logs: sudo journalctl -u pine-runner.service --since today   │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Strategy Research

**Goal**: Discover profitable base strategies through systematic parameter optimization.

**Location**: Local development machine

**Tools**: Strategy Factory (`research/strategy_factory/`)

### 1.1 Download Historical Data

**What**: Download Databento historical data for backtesting

**Where**: Local machine (`data/databento/`)

**Commands**:
```bash
# Download daily data (for reference feeds and long-term indicators)
python research/download_databento_data.py \
    --symbol ES \
    --dataset GLBX.MDP3 \
    --start 2010-01-01 \
    --end 2024-12-31 \
    --schema ohlcv-1d \
    --output data/databento/ES_daily.csv

# Download hourly data (for intraday strategy signals)
python research/download_databento_data.py \
    --symbol ES \
    --dataset GLBX.MDP3 \
    --start 2010-01-01 \
    --end 2024-12-31 \
    --schema ohlcv-1h \
    --output data/databento/ES_hourly.csv
```

**Output**:
- `data/databento/ES_daily.csv` → Daily OHLCV bars
- `data/databento/ES_hourly.csv` → Hourly OHLCV bars

**Storage**:
- ❌ **NOT in Git** (data files excluded by `.gitignore`)
- 💻 **Local only** (historical data for research)
- ☁️ **Optional**: Store in cloud backup (S3, Dropbox, etc.)

---

### 1.2 Run Strategy Factory Optimization

**What**: Test 54 strategy variants across parameter grids to find best configurations

**Where**: Local machine (`research/strategy_factory/`)

**Commands**:
```bash
# Run all 54 strategies for a single instrument
python research/strategy_factory/optimize_strategies.py \
    --symbol ES \
    --data-dir data/databento \
    --output-dir research/strategy_factory/results \
    --parallel 4

# Expected runtime: 2-6 hours per instrument (depending on CPU cores)
```

**Output**:
```
research/strategy_factory/results/
├── ES_ibs_baseline_v1.json          # Sharpe: 1.2, Trades: 450
├── ES_ibs_filtered_v2.json          # Sharpe: 1.8, Trades: 280 ← BEST
├── ES_ibs_adaptive_exit_v3.json     # Sharpe: 1.5, Trades: 320
└── ...50 more variants
```

**Decision Criteria**:
1. **Sharpe Ratio** > 1.5 (after transaction costs)
2. **Trade Frequency** 100-400 trades/year (sufficient for ML training)
3. **Max Drawdown** < 20%
4. **Robustness** Consistent across walk-forward periods

**Documentation**: See [STRATEGY_FACTORY_GUIDE.md](STRATEGY_FACTORY_GUIDE.md)

---

### 1.3 Select Optimal Strategy Parameters

**What**: Manually review results and select best variant per instrument

**Where**: Spreadsheet or notebook analysis

**Process**:
```bash
# Analyze all results
python research/strategy_factory/analyze_results.py \
    --results-dir research/strategy_factory/results \
    --symbol ES

# Output: Ranked strategies with statistical metrics
```

**Example Decision**:
```
ES → ibs_filtered_v2 (Sharpe 1.8, 280 trades, 12% drawdown)
NQ → ibs_baseline_v1 (Sharpe 1.6, 310 trades, 15% drawdown)
RTY → ibs_adaptive_exit_v3 (Sharpe 1.7, 250 trades, 14% drawdown)
```

**Next Step**: Use these parameters in ML training data extraction.

---

## Phase 2: ML Meta-Labeling

**Goal**: Train ensemble models (LightGBM + Random Forest) to filter strategy signals.

**Location**: Local development machine

**Tools**: ML meta-labeling pipeline (`research/ml_meta_labeling/`)

### 2.1 Extract Training Data

**What**: Run backtests with optimal strategy parameters, extract 30+ features per trade

**Where**: Local machine → `data/training/{SYMBOL}_training_data.csv`

**Commands**:
```bash
# Extract features using best strategy parameters from Phase 1
python research/extract_training_data.py \
    --symbol ES \
    --start 2010-01-01 \
    --end 2024-12-31 \
    --strategy-params research/strategy_factory/results/ES_ibs_filtered_v2.json \
    --data-dir data/databento \
    --output data/training/ES_training_data.csv

# Expected output: CSV with columns:
# - Date/Time, Entry Price, Exit Price, Profit, Return
# - 30+ features: ibs, atr, rsi, volume_z, cross_symbol z-scores, etc.
```

**Output File** (`data/training/ES_training_data.csv`):
```
Date/Time,Entry,Exit,Profit,Return,ibs,atr,rsi,volz,es_hourly_z_score,...
2010-01-04 10:30,1150.25,1152.00,175.00,0.0015,0.25,1.2,45,0.8,-0.3,...
2010-01-05 14:00,1148.50,1145.75,-275.00,-0.0024,0.85,1.4,62,1.2,0.5,...
...2800 rows (one per trade over 15 years)
```

**Storage**:
- ❌ **NOT in Git** (large CSV files excluded)
- 💻 **Local only** (training data for ML)

**Documentation**: See [docs/ml/DATA_TRANSFORMATION_TO_ML_WORKFLOW.md](docs/ml/DATA_TRANSFORMATION_TO_ML_WORKFLOW.md)

---

### 2.2 Train ML Model (Three-Way Temporal Split)

**What**: Train Random Forest meta-labeling model with rigorous validation

**Where**: Local machine → `src/models/{SYMBOL}_rf_model.pkl`

**Commands**:
```bash
# Train with three-way split: 2010-2018 / 2019-2020 / 2021-2024
python research/ml_meta_labeling/train_rf_three_way_split.py \
    --symbol ES \
    --data data/training/ES_training_data.csv \
    --train-end 2018-12-31 \
    --threshold-end 2020-12-31 \
    --rs-trials 120 \
    --bo-trials 300 \
    --embargo-days 3 \
    --output src/models

# Expected runtime: 3-8 hours (120 random + 300 Bayesian trials)
```

**Training Process**:
1. **Training Set (2010-2018)**: Hyperparameter tuning with CPCV
   - Random Search: 120 trials across parameter grid
   - Bayesian Optimization: 300 trials to find optimal config
   - Metric: Sharpe ratio on out-of-fold predictions
2. **Validation Set (2019-2020)**: Threshold optimization
   - Grid search over probability cutoffs (0.40 - 0.70)
   - Select threshold maximizing Sharpe ratio
3. **Test Set (2021-2024)**: Final evaluation (NEVER TOUCHED UNTIL NOW!)
   - Apply best hyperparameters + best threshold
   - Report unbiased performance estimate

**Output Files**:

**`src/models/ES_rf_model.pkl`** (Git LFS):
- Trained scikit-learn RandomForestClassifier
- ~50-100 MB per model

**`src/models/ES_best.json`** (Git):
```json
{
  "symbol": "ES",
  "model_type": "RandomForest",
  "threshold": 0.58,
  "features": ["ibs", "atr", "rsi", "volz", ...],
  "hyperparameters": {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 10,
    "min_samples_leaf": 5
  },
  "performance": {
    "train_sharpe": 2.1,
    "validation_sharpe": 1.8,
    "test_sharpe": 1.6,
    "test_dsr": 1.3,
    "test_win_rate": 0.58,
    "test_avg_profit": 87.50
  }
}
```

**Storage**:
- ✅ **In Git** (via Git LFS for .pkl, direct commit for .json)
- 🚀 **Deployed to server** (via git pull)

**Documentation**: See [docs/ml/THREE_WAY_SPLIT_GUIDE.md](docs/ml/THREE_WAY_SPLIT_GUIDE.md)

---

### 2.3 Validate Model Performance

**What**: Ensure model meets production quality standards

**Where**: Local machine (check output JSON)

**Validation Checklist**:
```bash
# Check test set performance (hold-out 2021-2024)
cat src/models/ES_best.json | jq '.performance'

# Required minimums for deployment:
# ✅ test_sharpe > 1.2
# ✅ test_dsr > 1.0 (Deflated Sharpe Ratio after multiple testing correction)
# ✅ test_win_rate > 0.52
# ✅ test_trades > 100 (sufficient sample size)
```

**Red Flags**:
- ⚠️ **test_sharpe << validation_sharpe** (>30% decay) → Overfitting to validation set
- ⚠️ **test_dsr < 0.5** → Performance not statistically significant
- ⚠️ **test_trades < 50** → Insufficient sample, unreliable metrics

**If validation fails**: Retrain with simpler model, different feature set, or adjust date ranges.

---

## Phase 3: Portfolio Optimization

**Goal**: Select optimal subset of strategies to trade simultaneously with position sizing.

**Location**: Local development machine

**Tools**: Portfolio optimizer (`research/portfolio_optimization/`)

### 3.1 Generate Per-Strategy Equity Curves

**What**: Backtest each trained model to produce daily equity curves

**Where**: Local machine → `data/portfolio/{SYMBOL}_{VARIANT}_equity.csv`

**Commands**:
```bash
# Generate equity curve for each instrument
for symbol in ES NQ RTY YM CL HG SI; do
    python research/generate_portfolio_backtest_data.py \
        --symbol $symbol \
        --model src/models/${symbol}_rf_model.pkl \
        --start 2021-01-01 \
        --end 2024-12-31 \
        --output data/portfolio/${symbol}_equity.csv
done

# Expected output: Daily P&L time series for correlation analysis
```

**Output File** (`data/portfolio/ES_equity.csv`):
```
Date,Equity,Daily_PnL,Drawdown
2021-01-04,100000,0,0
2021-01-05,100125,125,-0.001
2021-01-06,99950,-175,-0.005
...1000 rows (daily)
```

**Storage**:
- ❌ **NOT in Git** (generated files for portfolio optimization)
- 💻 **Local only**

---

### 3.2 Run Greedy Portfolio Optimizer

**What**: Select optimal strategy subset maximizing Sharpe with correlation constraints

**Where**: Local machine → Portfolio configuration

**Commands**:
```bash
# Run greedy optimizer with constraints
python research/portfolio_optimization/greedy_optimizer.py \
    --equity-dir data/portfolio \
    --max-positions 2 \
    --max-correlation 0.7 \
    --min-sharpe 1.2 \
    --output portfolio_config.json

# Expected output: Selected strategies ranked by marginal Sharpe contribution
```

**Example Output**:
```json
{
  "selected_strategies": [
    {"symbol": "ES", "size": 1, "sharpe": 1.8, "correlation": 0.45},
    {"symbol": "CL", "size": 1, "sharpe": 1.6, "correlation": 0.32},
    {"symbol": "RTY", "size": 2, "sharpe": 1.5, "correlation": 0.51},
    {"symbol": "HG", "size": 1, "sharpe": 1.4, "correlation": 0.28}
  ],
  "portfolio_metrics": {
    "sharpe": 2.1,
    "max_drawdown": 0.12,
    "avg_correlation": 0.39,
    "total_trades_per_year": 850
  }
}
```

**Decision**: Deploy these 4 strategies with specified position sizes.

**Documentation**: See [docs/portfolio/PORTFOLIO_INTEGRATION_GUIDE.md](docs/portfolio/PORTFOLIO_INTEGRATION_GUIDE.md)

---

### 3.3 Update Configuration

**What**: Update `config.yml` with selected instruments and sizes

**Where**: Local machine → `config.yml`

**Edit**:
```yaml
# config.yml (local copy for testing)
symbols: ["ES", "CL", "RTY", "HG"]
max_positions: 2

contracts:
  ES:
    size: 1
    commission: 1.00
  CL:
    size: 1
    commission: 1.00
  RTY:
    size: 2
    commission: 1.00
  HG:
    size: 1
    commission: 1.00
```

**Testing (CRITICAL)**:
```bash
# Test configuration locally before deploying to server
export PINE_RUNTIME_CONFIG=config.yml
export DATABENTO_API_KEY=your_api_key_here
export TRADERSPOST_WEBHOOK_URL=your_webhook_url_here
export POLICY_KILLSWITCH=true  # SAFETY: No live orders during testing

python -m runner.main

# Verify:
# ✅ All 4 models load successfully
# ✅ All cross-symbol feeds initialize
# ✅ Warmup completes without errors
# ✅ Feature verification shows 100% coverage

# Ctrl+C to stop after validation
```

**Commit to Git**:
```bash
git add config.yml src/models/
git commit -m "Deploy portfolio: ES, CL, RTY, HG with optimized sizes"
git push origin main
```

---

## Phase 4: Live Deployment

**Goal**: Deploy trained models to production server for live trading.

**Location**: Production server (`/opt/pine/rooney-capital-v1/`)

**Tools**: systemd services, Git, journalctl

### 4.1 Pull Latest Code on Server

**What**: Update production code with new models and configuration

**Where**: SSH into production server

**Commands**:
```bash
# SSH to server
ssh user@trading-server.example.com

# Navigate to production directory
cd /opt/pine/rooney-capital-v1

# Pull latest code (requires Git authentication)
git pull origin main

# Download model files (Git LFS)
git lfs pull

# Verify models updated
ls -lh src/models/*.pkl  # Check modification dates

# Check what changed
git log -1 --stat
```

**Verification**:
```bash
# Ensure all required models exist
for symbol in ES CL RTY HG; do
    ls src/models/${symbol}_rf_model.pkl src/models/${symbol}_best.json
done

# Should see 8 files (4 .pkl + 4 .json)
```

---

### 4.2 Update Server Configuration

**What**: Update runtime config and credentials

**Where**: `/opt/pine/runtime/` (separate from code repository)

**Files to Update**:

**`/opt/pine/runtime/config.yml`** (production config):
```yaml
contract_map: Data/Databento_contract_map.yml
models_path: src/models
symbols: ["ES", "CL", "RTY", "HG"]  # Updated from portfolio optimization
databento_api_key: ${DATABENTO_API_KEY}
traderspost_webhook: ${TRADERSPOST_WEBHOOK_URL}
starting_cash: 250000
backfill: true
backfill_minutes: 180
max_positions: 2  # Updated constraint

contracts:
  ES:
    size: 1  # Updated from portfolio optimization
    commission: 1.00
  CL:
    size: 1
    commission: 1.00
  RTY:
    size: 2  # Updated
    commission: 1.00
  HG:
    size: 1
    commission: 1.00
```

**`/opt/pine/runtime/.env`** (credentials - NEVER in Git):
```bash
DATABENTO_API_KEY=db-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TRADERSPOST_WEBHOOK_URL=https://webhooks.traderspost.io/trading/webhook/xxxxxxxx
POLICY_KILLSWITCH=false  # IMPORTANT: Set to true to halt trading in emergency
PINE_RUNTIME_CONFIG=/opt/pine/runtime/config.yml
```

**Security Check**:
```bash
# Ensure .env is not world-readable
chmod 600 /opt/pine/runtime/.env

# Verify environment variables load correctly
source /opt/pine/runtime/.env
echo $DATABENTO_API_KEY | head -c 10  # Should show "db-xxxxxxx"
```

---

### 4.3 Restart Trading Services

**What**: Restart systemd services to load new code and models

**Where**: Production server (requires sudo)

**Commands**:
```bash
# Stop services gracefully (allows positions to close)
sudo systemctl stop pine-runner.service
sudo systemctl stop rooney-dashboard.service

# Verify services stopped
sudo systemctl status pine-runner.service  # Should show "inactive (dead)"

# Start services with new configuration
sudo systemctl start pine-runner.service
sudo systemctl start rooney-dashboard.service

# Verify services started successfully
sudo systemctl status pine-runner.service  # Should show "active (running)"
sudo systemctl status rooney-dashboard.service
```

**Monitor Startup**:
```bash
# Follow live logs during warmup
sudo journalctl -u pine-runner.service -f

# Expected log sequence:
# [INFO] Loading runtime config from /opt/pine/runtime/config.yml
# [INFO] Loaded 4 models: ES, CL, RTY, HG
# [INFO] Connecting to Databento live gateway...
# [INFO] Subscribed to 4 symbols + 8 reference feeds
# [INFO] Starting indicator warmup (backfilling 180 minutes)...
# [INFO] ES: Warmup complete (180 bars processed)
# [INFO] CL: Warmup complete (180 bars processed)
# [INFO] RTY: Warmup complete (180 bars processed)
# [INFO] HG: Warmup complete (180 bars processed)
# [INFO] Indicator warmup completed - Ready for trading
# [INFO] Strategy started: max_positions=2, symbols=['ES', 'CL', 'RTY', 'HG']
```

**Warmup Duration**: Typically 5-10 minutes depending on market hours.

---

### 4.4 Validate Deployment

**What**: Ensure system is operating correctly before allowing live trading

**Where**: Production server + monitoring dashboard

**Validation Checklist**:

#### ✅ Service Health
```bash
# Check all services are running
sudo systemctl status pine-runner.service
sudo systemctl status rooney-dashboard.service

# Expected: Both show "active (running)"
```

#### ✅ Model Loading
```bash
# Verify all models loaded
sudo journalctl -u pine-runner.service --no-pager | grep "Loaded.*model"

# Expected output:
# [INFO] Loaded ES model: 29 features, threshold=0.58
# [INFO] Loaded CL model: 29 features, threshold=0.62
# [INFO] Loaded RTY model: 28 features, threshold=0.55
# [INFO] Loaded HG model: 29 features, threshold=0.60
```

#### ✅ Feature Coverage
```bash
# Check ML feature verification (run after warmup completes)
cd /opt/pine/rooney-capital-v1
source venv/bin/activate
python research/validation/verify_features.py

# Expected output:
# ================================================================================
# ML SCORING RESULTS BY SYMBOL
# ================================================================================
# ✅ ES  : 29/29 features (100.0%)
# ✅ CL  : 29/29 features (100.0%)
# ✅ RTY : 28/28 features (100.0%)
# ✅ HG  : 29/29 features (100.0%)
```

#### ✅ Live Data Feed
```bash
# Verify Databento connection and data flow
sudo journalctl -u pine-runner.service --no-pager | grep -E "Databento|heartbeat"

# Expected:
# [INFO] Databento connection established
# [INFO] Heartbeat: ES last_bar=2024-11-22 14:35:00, bars_received=182
# [INFO] Heartbeat: CL last_bar=2024-11-22 14:35:00, bars_received=182
```

#### ✅ Dashboard Access
```bash
# Open browser to monitoring dashboard
# URL: http://your-server-ip:8501

# Expected:
# - System status: "LIVE TRADING ACTIVE"
# - Active instruments: ES, CL, RTY, HG
# - Current positions: 0/2
# - Recent signals: Table showing probability scores
```

#### ✅ First Signal Generation (Wait 1 Hour)
```bash
# After 1 hour of market hours, verify signals generated
sudo journalctl -u pine-runner.service --since "1 hour ago" | grep -E "Signal|Entry"

# Expected (if signals triggered):
# [INFO] ES: IBS signal detected (ibs=0.23, probability=0.65 > 0.58)
# [INFO] ES: ENTRY LONG at 4525.25 (size=1, reason=ML_APPROVED)
```

**Red Flags**:
- ❌ **No data received after 5 minutes** → Check Databento API key, network connectivity
- ❌ **Feature coverage < 100%** → Check cross-symbol feeds, reference data
- ❌ **Models failing to load** → Verify .pkl files downloaded (git lfs pull)
- ❌ **Signals generated but no orders** → Check TradersPost webhook, kill switch

**If ANY validation fails**: Set `POLICY_KILLSWITCH=true`, investigate, fix, restart.

---

### 4.5 Monitor Live Trading

**What**: Continuous monitoring during market hours

**Where**: Dashboard + TradersPost + Logs

**Monitoring Tools**:

#### 📊 **Streamlit Dashboard** (`http://server:8501`)
- Real-time P&L by instrument
- Active positions and fills
- Signal history with probability scores
- Feature calculation status
- System health metrics

#### 📱 **TradersPost Web Console** (`https://traderspost.io`)
- Verify orders submitted and filled
- Check position reconciliation
- Review execution quality (slippage, fill price)

#### 📝 **System Logs**
```bash
# View today's trading activity
sudo journalctl -u pine-runner.service --since today

# Filter for specific events
sudo journalctl -u pine-runner.service | grep -E "ENTRY|EXIT|Error|Warning"

# Monitor in real-time
sudo journalctl -u pine-runner.service -f
```

**Daily Checklist**:
- [ ] Check dashboard at market open (9:30 AM ET)
- [ ] Verify all instruments showing live data
- [ ] Review overnight positions (if any)
- [ ] Monitor first signals of the day
- [ ] Check P&L at market close (4:00 PM ET)
- [ ] Verify positions closed before 3:00 PM (auto-exit)
- [ ] Review logs for errors or warnings

**Weekly Checklist**:
- [ ] Review win rate by instrument (should match test set ±5%)
- [ ] Check average profit per trade (should match backtest ±$20)
- [ ] Monitor slippage (should be ~1 tick per order)
- [ ] Verify no duplicate orders or missed exits
- [ ] Backup logs and trade history database

**Documentation**: See [docs/operations/MONITORING_GUIDE.md](docs/operations/MONITORING_GUIDE.md)

---

## Data Locations

Clear separation between development data (local), production data (server), and version-controlled assets (Git).

### 💻 Local Development Machine

**Historical Data** (NOT in Git):
```
data/
├── databento/                    # Raw historical data from Databento
│   ├── ES_daily.csv              # Downloaded via API (10+ MB per instrument)
│   ├── ES_hourly.csv
│   ├── NQ_daily.csv
│   └── ...
├── training/                     # ML training data (generated locally)
│   ├── ES_training_data.csv      # Extracted features (2-5 MB per instrument)
│   ├── NQ_training_data.csv
│   └── ...
└── portfolio/                    # Portfolio optimization inputs
    ├── ES_equity.csv
    ├── NQ_equity.csv
    └── ...
```

**Research Outputs** (NOT in Git):
```
research/
├── strategy_factory/
│   └── results/                  # Strategy optimization results
│       ├── ES_ibs_baseline_v1.json
│       ├── ES_ibs_filtered_v2.json
│       └── ...54 variants per instrument
├── ml_meta_labeling/
│   └── logs/                     # Training logs and diagnostics
└── portfolio_optimization/
    └── results/                  # Portfolio backtest results
```

**Why NOT in Git**:
- Large files (100+ MB per instrument)
- Regenerated from Databento API as needed
- Different across team members (date ranges, instruments)

---

### 🚀 Production Server (`/opt/pine/`)

**Runtime Data** (NOT in Git):
```
/opt/pine/
├── runtime/                      # Configuration and credentials
│   ├── config.yml                # Production configuration (symbols, sizes, etc.)
│   ├── .env                      # Secrets (API keys, webhooks) - NEVER commit
│   └── heartbeat.json            # Real-time health status
├── logs/                         # Application logs
│   ├── worker_YYYYMMDD.log       # Daily trading logs
│   └── dashboard_YYYYMMDD.log    # Dashboard access logs
├── database/                     # Trade history database
│   └── trades.db                 # SQLite database (positions, fills, P&L)
└── rooney-capital-v1/            # Code repository (git clone)
    ├── src/                      # ✅ From Git
    ├── research/                 # ✅ From Git (but not used on server)
    ├── tests/                    # ✅ From Git
    └── ...
```

**Why NOT in Git**:
- Credentials contain secrets (API keys, webhook URLs)
- Logs and database change constantly during trading
- Server-specific configuration (file paths, resource limits)

---

### ✅ Version Control (Git + Git LFS)

**Production Code** (in Git):
```
src/
├── strategy/
│   ├── ibs_strategy.py           # Main trading strategy
│   ├── feature_utils.py          # Feature engineering
│   └── contract_specs.py         # Instrument specifications
├── runner/
│   ├── live_worker.py            # Live trading orchestration
│   ├── databento_bridge.py       # Market data ingestion
│   └── traderspost_client.py     # Order routing
└── models/
    ├── ES_rf_model.pkl           # ✅ Git LFS (50-100 MB)
    ├── ES_best.json              # ✅ Direct commit (5 KB)
    ├── NQ_rf_model.pkl           # ✅ Git LFS
    └── NQ_best.json              # ✅ Direct commit
```

**Research Code** (in Git):
```
research/
├── strategy_factory/
│   └── optimize_strategies.py    # Strategy parameter optimization
├── ml_meta_labeling/
│   └── train_rf_three_way_split.py  # ML model training
├── portfolio_optimization/
│   └── greedy_optimizer.py       # Portfolio construction
└── ...
```

**Configuration Templates** (in Git):
```
config.example.yml                # Template (copy to config.yml)
.env.example                      # Template (copy to .env)
Data/Databento_contract_map.yml   # Contract specifications
```

**Documentation** (in Git):
```
README.md
SYSTEM_GUIDE.md
docs/
├── ml/
├── portfolio/
├── operations/
└── ...
```

**Why IN Git**:
- Shared across team and servers
- Version controlled for auditability
- Deployed via `git pull` (no manual file copying)

---

## What Runs Where

Clear separation between local research workflows and production trading operations.

### 💻 **Local Development Machine**

**Purpose**: Research, backtesting, model training, portfolio optimization

**Activities**:
- Download historical data
- Run strategy factory optimization (54 variants × N instruments)
- Extract ML training data from backtests
- Train ML models (3-8 hours per instrument)
- Optimize portfolio composition
- Test new code changes
- Run unit and integration tests

**Tools Used**:
- Python scripts in `research/`
- Jupyter notebooks (optional)
- Databento historical data API
- Local git repository

**Does NOT**:
- ❌ Connect to live market data
- ❌ Send orders to broker
- ❌ Run 24/7
- ❌ Store production credentials

---

### 🚀 **Production Server** (`/opt/pine/`)

**Purpose**: Live trading, real-time monitoring, data collection

**Activities**:
- Connect to Databento live data stream
- Execute IBS strategy with ML filtering
- Send orders to TradersPost → Tradovate
- Log trades to SQLite database
- Serve monitoring dashboard (Streamlit)
- Generate heartbeat health checks

**Services Running**:
```bash
# Main trading worker (systemd service)
pine-runner.service
→ Runs: python -m runner.main
→ Logs: /opt/pine/logs/worker_YYYYMMDD.log
→ Restart: sudo systemctl restart pine-runner.service

# Monitoring dashboard (systemd service)
rooney-dashboard.service
→ Runs: streamlit run dashboard/app.py
→ Access: http://server:8501
→ Restart: sudo systemctl restart rooney-dashboard.service
```

**Tools Used**:
- Git repository cloned to `/opt/pine/rooney-capital-v1/`
- Production config at `/opt/pine/runtime/config.yml`
- Python virtual environment at `/opt/pine/rooney-capital-v1/venv/`
- systemd for process management

**Does NOT**:
- ❌ Run backtests (too slow for research)
- ❌ Train ML models (requires historical data)
- ❌ Download historical data (only live stream)

---

### ☁️ **Git Repository** (GitHub)

**Purpose**: Version control, code distribution, model storage

**Contents**:
- ✅ Production code (`src/`)
- ✅ Research scripts (`research/`)
- ✅ Tests (`tests/`)
- ✅ Documentation (`docs/`, `README.md`)
- ✅ ML models (via Git LFS: `src/models/*.pkl`)
- ✅ Configuration templates (`config.example.yml`, `.env.example`)
- ❌ Historical data (too large)
- ❌ Credentials (security risk)
- ❌ Logs and databases (change constantly)

**Workflow**:
```bash
# Local: After training new models
git add src/models/ES_rf_model.pkl src/models/ES_best.json
git commit -m "Retrain ES model (test Sharpe 1.6 → 1.8)"
git push origin main

# Server: Deploy new models
cd /opt/pine/rooney-capital-v1
git pull origin main
git lfs pull
sudo systemctl restart pine-runner.service
```

---

## Development vs Production

### 🧪 **Development Workflow** (Safe Testing)

**Environment Variables**:
```bash
export PINE_RUNTIME_CONFIG=config.yml  # Local copy for testing
export DATABENTO_API_KEY=db-xxxx       # Your API key
export TRADERSPOST_WEBHOOK_URL=https://webhooks.traderspost.io/...
export POLICY_KILLSWITCH=true          # ⚠️ CRITICAL: Prevents live orders
```

**Testing Process**:
```bash
# 1. Run preflight checks (validates config without trading)
python scripts/worker_preflight.py

# Expected output:
# ✅ ML Models validated
# ✅ TradersPost connection successful
# ✅ Databento connection successful
# ✅ Reference data loaded
# ✅ Data feeds registered

# 2. Run worker in test mode (with kill switch)
export POLICY_KILLSWITCH=true
python -m runner.main

# Verify:
# - Models load correctly
# - Indicators calculate properly
# - Signals generate as expected
# - NO ORDERS SENT (kill switch prevents webhook calls)

# 3. Ctrl+C to stop after validation
```

**Local Backtesting**:
```bash
# Test new strategy changes with historical data
python research/backtest_runner.py \
    --symbol ES \
    --start 2023-01-01 \
    --end 2024-01-01 \
    --model src/models/ES_rf_model.pkl \
    --data-dir data/databento

# Output: Performance metrics WITHOUT risking real capital
```

---

### 🚀 **Production Workflow** (Live Trading)

**Environment Variables** (on server):
```bash
# /opt/pine/runtime/.env
PINE_RUNTIME_CONFIG=/opt/pine/runtime/config.yml
DATABENTO_API_KEY=db-production-key-xxxx
TRADERSPOST_WEBHOOK_URL=https://webhooks.traderspost.io/trading/webhook/production-xxxx
POLICY_KILLSWITCH=false  # ⚠️ Live trading enabled (orders will execute)
```

**Deployment Checklist**:
- [ ] All models trained and validated (test Sharpe > 1.2)
- [ ] Portfolio optimization complete
- [ ] Local testing with `POLICY_KILLSWITCH=true` passed
- [ ] Server configuration updated (`config.yml` and `.env`)
- [ ] TradersPost strategies point to correct contracts
- [ ] Git committed and pushed to main branch
- [ ] Server pulled latest code (`git pull && git lfs pull`)
- [ ] Services restarted (`sudo systemctl restart pine-runner.service`)
- [ ] Validation checks passed (models loaded, features 100%, data flowing)
- [ ] Monitoring dashboard accessible
- [ ] First hour monitored manually

**Emergency Stop**:
```bash
# If anything goes wrong during live trading:

# Option 1: Kill switch (stops new orders, allows exits)
sudo systemctl restart pine-runner.service  # Loads POLICY_KILLSWITCH=true from .env

# Option 2: Complete shutdown
sudo systemctl stop pine-runner.service

# Option 3: Manually close positions via TradersPost web console
# Then investigate logs to diagnose issue
```

---

## Complete Example: Adding a New Instrument

Step-by-step workflow to add a new futures contract (e.g., Gold - GC) to the portfolio.

### Step 1: Download Historical Data (Local)

```bash
# Download daily data (for reference feeds)
python research/download_databento_data.py \
    --symbol GC \
    --dataset GLBX.MDP3 \
    --start 2010-01-01 \
    --end 2024-12-31 \
    --schema ohlcv-1d \
    --output data/databento/GC_daily.csv

# Download hourly data (for intraday strategy)
python research/download_databento_data.py \
    --symbol GC \
    --dataset GLBX.MDP3 \
    --start 2010-01-01 \
    --end 2024-12-31 \
    --schema ohlcv-1h \
    --output data/databento/GC_hourly.csv

# Expected download: ~50 MB total for 15 years
```

---

### Step 2: Strategy Factory Optimization (Local)

```bash
# Run all 54 strategy variants for GC
python research/strategy_factory/optimize_strategies.py \
    --symbol GC \
    --data-dir data/databento \
    --output-dir research/strategy_factory/results \
    --parallel 4

# Expected runtime: 3-6 hours (depending on CPU)

# Analyze results
python research/strategy_factory/analyze_results.py \
    --results-dir research/strategy_factory/results \
    --symbol GC

# Example output:
# Rank  Variant                    Sharpe  Trades  MaxDD   Robust
# 1     GC_ibs_filtered_v2         1.7     240     14%     ✅
# 2     GC_ibs_adaptive_exit_v3    1.6     210     16%     ✅
# 3     GC_ibs_baseline_v1         1.4     280     18%     ⚠️

# Decision: Use GC_ibs_filtered_v2 (best Sharpe + robustness)
```

---

### Step 3: Extract ML Training Data (Local)

```bash
# Extract features using optimal strategy parameters
python research/extract_training_data.py \
    --symbol GC \
    --start 2010-01-01 \
    --end 2024-12-31 \
    --strategy-params research/strategy_factory/results/GC_ibs_filtered_v2.json \
    --data-dir data/databento \
    --output data/training/GC_training_data.csv

# Expected output: data/training/GC_training_data.csv (~2.5 MB, 2400 trades)
```

---

### Step 4: Train ML Model (Local)

```bash
# Train Random Forest with three-way split
python research/ml_meta_labeling/train_rf_three_way_split.py \
    --symbol GC \
    --data data/training/GC_training_data.csv \
    --train-end 2018-12-31 \
    --threshold-end 2020-12-31 \
    --rs-trials 120 \
    --bo-trials 300 \
    --embargo-days 3 \
    --output src/models

# Expected runtime: 4-8 hours

# Verify test set performance
cat src/models/GC_best.json | jq '.performance'

# Expected output:
# {
#   "train_sharpe": 2.0,
#   "validation_sharpe": 1.7,
#   "test_sharpe": 1.5,      ← Must be > 1.2
#   "test_dsr": 1.2,          ← Must be > 1.0
#   "test_win_rate": 0.57,
#   "test_avg_profit": 95.30
# }

# ✅ Passes validation → Proceed to portfolio optimization
```

---

### Step 5: Portfolio Optimization (Local)

```bash
# Generate equity curve for GC
python research/generate_portfolio_backtest_data.py \
    --symbol GC \
    --model src/models/GC_rf_model.pkl \
    --start 2021-01-01 \
    --end 2024-12-31 \
    --output data/portfolio/GC_equity.csv

# Re-run greedy optimizer with GC included
python research/portfolio_optimization/greedy_optimizer.py \
    --equity-dir data/portfolio \
    --max-positions 2 \
    --max-correlation 0.7 \
    --min-sharpe 1.2 \
    --output portfolio_config_v2.json

# Example output (GC added as 5th strategy):
# Selected: ES (1.8), CL (1.6), RTY (1.5), HG (1.4), GC (1.5)
# Portfolio Sharpe: 2.1 → 2.3 (improved with GC)
# Avg Correlation: 0.39 → 0.41 (still acceptable)
```

---

### Step 6: Update Configuration (Local)

```bash
# Edit config.yml
nano config.yml

# Add GC to symbols list and contracts section:
symbols: ["ES", "CL", "RTY", "HG", "GC"]  # Added GC

contracts:
  # ... existing configs ...
  GC:
    size: 1
    commission: 1.00

# Save and test locally
export POLICY_KILLSWITCH=true  # Safety first!
python -m runner.main

# Verify:
# ✅ GC model loads successfully
# ✅ GC feeds initialize (daily + hourly)
# ✅ GC features calculate (verify_features.py shows 100%)

# Ctrl+C after validation
```

---

### Step 7: Commit and Push (Local → Git)

```bash
# Stage new model and config
git add src/models/GC_rf_model.pkl src/models/GC_best.json config.yml

# Commit with descriptive message
git commit -m "Add GC (Gold) to portfolio

- Trained on 2010-2024 data (2400 trades)
- Test Sharpe: 1.5, DSR: 1.2
- Portfolio Sharpe: 2.1 → 2.3
- Max correlation: 0.41 (acceptable)
- Size: 1 contract per signal"

# Push to GitHub
git push origin main
```

---

### Step 8: Deploy to Production (Server)

```bash
# SSH to server
ssh user@trading-server.example.com

# Pull latest code
cd /opt/pine/rooney-capital-v1
git pull origin main
git lfs pull  # Download GC_rf_model.pkl

# Update production config
nano /opt/pine/runtime/config.yml

# Add GC to symbols and contracts (same as local config)
symbols: ["ES", "CL", "RTY", "HG", "GC"]

contracts:
  GC:
    size: 1
    commission: 1.00

# Restart services
sudo systemctl restart pine-runner.service
sudo systemctl restart rooney-dashboard.service

# Monitor startup
sudo journalctl -u pine-runner.service -f

# Expected logs:
# [INFO] Loaded 5 models: ES, CL, RTY, HG, GC
# [INFO] GC: Warmup complete (180 bars processed)
# [INFO] Indicator warmup completed - Ready for trading
```

---

### Step 9: Validate Deployment (Server)

```bash
# Verify GC model loaded
sudo journalctl -u pine-runner.service --no-pager | grep "GC"

# Expected:
# [INFO] Loaded GC model: 30 features, threshold=0.59

# Check feature coverage
cd /opt/pine/rooney-capital-v1
source venv/bin/activate
python research/validation/verify_features.py

# Expected:
# ✅ GC  : 30/30 features (100.0%)

# Monitor dashboard
# Open http://server:8501
# Verify:
# - GC appears in active instruments list
# - GC data feed shows live prices
# - System status: "LIVE TRADING ACTIVE"
```

---

### Step 10: Monitor First Trades (Server)

```bash
# Watch for GC signals over next few hours
sudo journalctl -u pine-runner.service -f | grep "GC"

# Example first signal:
# [INFO] GC: IBS signal detected (ibs=0.28, probability=0.65 > 0.59)
# [INFO] GC: ENTRY LONG at 1985.30 (size=1, reason=ML_APPROVED)

# Verify in TradersPost web console:
# - Order submitted and filled
# - Fill price matches expected (± 1 tick slippage)
# - Position size correct (1 contract)

# After first exit:
# [INFO] GC: EXIT LONG at 1988.70 (profit=+$340.00, hold_time=6 bars)

# Validate P&L calculation:
# (1988.70 - 1985.30) × 100 (multiplier) = $340.00 ✅
```

**GC is now live!** Continue monitoring for first day, then weekly performance reviews.

---

## Maintenance Workflows

### Weekly: Model Retraining (Optional)

**When**: Every Sunday night or monthly, depending on performance drift

**Process**:
```bash
# 1. Download latest data (local)
for symbol in ES NQ RTY YM CL HG SI GC; do
    python research/download_databento_data.py \
        --symbol $symbol \
        --dataset GLBX.MDP3 \
        --start 2010-01-01 \
        --end $(date +%Y-%m-%d) \
        --schema ohlcv-1h \
        --output data/databento/${symbol}_hourly.csv
done

# 2. Retrain all models (local)
for symbol in ES NQ RTY YM CL HG SI GC; do
    # Extract fresh training data
    python research/extract_training_data.py \
        --symbol $symbol \
        --start 2010-01-01 \
        --end $(date +%Y-%m-%d) \
        --output data/training/${symbol}_training_data.csv

    # Retrain model
    python research/ml_meta_labeling/train_rf_three_way_split.py \
        --symbol $symbol \
        --train-end 2018-12-31 \
        --threshold-end 2020-12-31 \
        --output src/models
done

# 3. Validate all models passed quality checks
for symbol in ES NQ RTY YM CL HG SI GC; do
    test_sharpe=$(cat src/models/${symbol}_best.json | jq '.performance.test_sharpe')
    test_dsr=$(cat src/models/${symbol}_best.json | jq '.performance.test_dsr')
    echo "$symbol: Sharpe=$test_sharpe, DSR=$test_dsr"
done

# 4. Commit and deploy (if all passed)
git add src/models/
git commit -m "Weekly model retraining ($(date +%Y-%m-%d))"
git push origin main

# 5. Deploy to server (during off-hours)
ssh server "cd /opt/pine/rooney-capital-v1 && git pull && git lfs pull"
ssh server "sudo systemctl restart pine-runner.service"
```

---

### Monthly: Portfolio Rebalancing

**When**: First Sunday of each month

**Process**:
```bash
# 1. Generate equity curves with latest data (local)
for symbol in ES NQ RTY YM CL HG SI GC; do
    python research/generate_portfolio_backtest_data.py \
        --symbol $symbol \
        --model src/models/${symbol}_rf_model.pkl \
        --start 2021-01-01 \
        --end $(date +%Y-%m-%d) \
        --output data/portfolio/${symbol}_equity.csv
done

# 2. Re-run portfolio optimization
python research/portfolio_optimization/greedy_optimizer.py \
    --equity-dir data/portfolio \
    --max-positions 2 \
    --max-correlation 0.7 \
    --min-sharpe 1.2 \
    --output portfolio_config_$(date +%Y%m).json

# 3. Review changes
diff portfolio_config_current.json portfolio_config_$(date +%Y%m).json

# Example output:
# - Removed: SI (Sharpe dropped to 1.1)
# + Added: NG (Sharpe improved to 1.5)
# ~ RTY size: 2 → 1 (risk reduction)

# 4. Update config.yml if changes approved
# 5. Deploy to production (same as Step 8 above)
```

---

### Daily: Performance Monitoring

**When**: End of each trading day (4:00 PM ET)

**Process**:
```bash
# Check dashboard for daily P&L
curl -s http://server:8501/api/daily_pnl | jq

# Expected output:
# {
#   "ES": 125.00,
#   "CL": -87.50,
#   "RTY": 310.00,
#   "HG": 45.00,
#   "GC": 180.00,
#   "total": 572.50
# }

# Verify against TradersPost
# - Log in to TradersPost web console
# - Check closed trades for today
# - Reconcile P&L (should match ±$10 for commissions)

# Review logs for anomalies
ssh server "sudo journalctl -u pine-runner.service --since today | grep -E 'Error|Warning'"

# If any warnings: Investigate immediately
# If clean: No action needed
```

---

## Troubleshooting

### Problem: Model Training Fails

**Symptom**:
```
python research/ml_meta_labeling/train_rf_three_way_split.py --symbol ES
...
ValueError: Not enough samples in training set (need 100, have 45)
```

**Cause**: Insufficient training data (base strategy generated too few trades)

**Solution**:
```bash
# Option 1: Extend date range
python research/extract_training_data.py \
    --symbol ES \
    --start 2005-01-01 \  # Earlier start date
    --end 2024-12-31

# Option 2: Adjust base strategy to increase trade frequency
# Edit strategy params to lower IBS threshold (more signals)

# Option 3: Use two-way split instead of three-way
python research/ml_meta_labeling/train_rf_two_way_split.py \  # Alternate script
    --symbol ES \
    --test-end 2020-12-31  # Only train/test split
```

---

### Problem: Features Not Calculating on Server

**Symptom**:
```
python research/validation/verify_features.py
⚠️  ES  : 21/29 features (72.4%)
Missing: 6a_hourly_z_score, es_hourly_z_score, tlt_daily_z_score, ...
```

**Cause**: Cross-symbol feeds not loading during strategy initialization

**Solution**:
```bash
# Check contract map includes all required reference symbols
cat Data/Databento_contract_map.yml | grep -E "6A|TLT|VIX"

# Verify feeds loaded during startup
sudo journalctl -u pine-runner.service | grep "Converted.*historical data"

# Expected:
# [INFO] Converted 180 bars for 6A_hour
# [INFO] Converted 180 bars for TLT_day

# If missing, check Databento subscription includes these products

# Temporary fix: Expand METAL_ENERGY_SYMBOLS in ibs_strategy.py
# This forces creation of cross-symbol indicators
```

---

### Problem: Live Trading Not Starting After Deployment

**Symptom**:
```
sudo systemctl status pine-runner.service
● pine-runner.service - loaded
   Active: failed (Result: exit-code)
```

**Diagnosis**:
```bash
# Check logs for error message
sudo journalctl -u pine-runner.service -n 50

# Common errors:
```

**Error 1**: `FileNotFoundError: [Errno 2] No such file: 'src/models/ES_rf_model.pkl'`
```bash
# Solution: Download models with Git LFS
cd /opt/pine/rooney-capital-v1
git lfs pull
ls -lh src/models/*.pkl  # Verify files exist and are >1 MB (not pointers)
```

**Error 2**: `databento.DBNError: Authentication failed`
```bash
# Solution: Check API key in .env
cat /opt/pine/runtime/.env | grep DATABENTO_API_KEY

# Verify key is correct (should start with "db-")
# Test manually:
python -c "import databento as db; client = db.Historical('db-xxx'); print('OK')"
```

**Error 3**: `requests.exceptions.ConnectionError: Failed to connect to TradersPost`
```bash
# Solution: Verify webhook URL
cat /opt/pine/runtime/.env | grep TRADERSPOST_WEBHOOK_URL

# Test manually:
curl -X POST https://webhooks.traderspost.io/trading/webhook/xxx \
  -H "Content-Type: application/json" \
  -d '{"test": true}'

# Should return HTTP 200 or 204
```

---

### Problem: Signals Generated But No Orders Sent

**Symptom**:
```
sudo journalctl -u pine-runner.service | grep "Signal"
[INFO] ES: IBS signal detected (ibs=0.23, probability=0.65 > 0.58)
[INFO] ES: Signal approved by ML model
...but no "ENTRY LONG" message
```

**Cause 1**: Kill switch enabled
```bash
cat /opt/pine/runtime/.env | grep POLICY_KILLSWITCH
# If POLICY_KILLSWITCH=true, orders are blocked

# Solution: Set to false (only if intentionally trading live)
sed -i 's/POLICY_KILLSWITCH=true/POLICY_KILLSWITCH=false/' /opt/pine/runtime/.env
sudo systemctl restart pine-runner.service
```

**Cause 2**: Max positions reached
```bash
sudo journalctl -u pine-runner.service | grep "max_positions"
# [INFO] Cannot enter ES: already at max_positions (2/2)

# Solution: Wait for existing position to exit, or increase max_positions in config.yml
```

**Cause 3**: Pair IBS filter blocking trade
```bash
sudo journalctl -u pine-runner.service | grep "Pair IBS"
# [INFO] ES: Pair IBS outside range (6E ibs=0.95 > 0.85) - Signal rejected

# Solution: Review pair IBS thresholds in strategy config
# May need to adjust if too restrictive
```

---

## Summary

This end-to-end workflow document provides:

✅ **Complete pipeline**: Strategy research → ML training → Portfolio optimization → Live deployment

✅ **Clear data locations**: Local vs server vs Git, with rationale for each

✅ **Development vs production**: Safety protocols, testing procedures, deployment checklists

✅ **Concrete example**: Step-by-step guide to adding a new instrument (GC)

✅ **Maintenance workflows**: Weekly retraining, monthly rebalancing, daily monitoring

✅ **Troubleshooting**: Common issues and solutions

**Next Steps**:
- **New to the system?** Start with [QUICK_START.md](QUICK_START.md)
- **Deep dive on ML?** See [docs/ml/README.md](docs/ml/README.md)
- **Ready to deploy?** Follow [LIVE_LAUNCH_GUIDE.md](LIVE_LAUNCH_GUIDE.md)
- **Operations questions?** See [SYSTEM_GUIDE.md](SYSTEM_GUIDE.md)

---

**Document Version**: 1.0
**Last Updated**: 2024-11-22
**Maintained By**: Rooney Capital Development Team
