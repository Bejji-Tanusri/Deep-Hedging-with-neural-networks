# NIFTY Options Analytics & Deep Hedging Pipeline

A Python pipeline (exported from a Google Colab notebook) that analyzes intraday NIFTY index options data, derives implied volatility and Greeks, simulates classical Black-Scholes delta hedging, and then trains a neural-network "deep hedger" to compare against it using a CVaR (tail-risk) loss.

## What it does

The script runs end-to-end through two phases:

### Phase 1 — Options Analytics on Real Market Data
1. **Load data**: Uploads an intraday CSV (NIFTY futures + options snapshots) via Colab file upload into a pandas DataFrame.
2. **Parse & filter**: Filters to a target minute snapshot, extracts the futures price, and parses option symbols to recover strike price and Call/Put (CE/PE) type.
3. **Black-Scholes pricing**: Implements the Black-Scholes formula and analytic Greeks (Delta, Gamma, Theta).
4. **Implied volatility**: Solves for IV from observed market prices using a bisection root-finder.
5. **Visualization**: Plots the IV smile/skew and Delta/Gamma/Theta against strike price.
6. **Classical delta hedging**: Reconstructs a minute-by-minute delta-hedging simulation for one specific at-the-money NIFTY call option across the trading session, recomputing IV and delta at each step, and reports the resulting hedging P&L.

### Phase 2 — Deep Hedging vs. Black-Scholes (Simulation)
7. **Synthetic paths**: Generates Monte Carlo price paths via Geometric Brownian Motion (GBM), seeded from the real futures price, to build a large training dataset.
8. **Theoretical targets**: Computes Black-Scholes prices and deltas along every synthetic path as benchmark/reference signals.
9. **Deep hedging model**: Builds a small feed-forward (Dense) network in TensorFlow/Keras that takes `(S_t, BS_delta_t, delta_{t-1})` at each time step and predicts the new hedge ratio `delta_t`, rolled out sequentially over each path.
10. **Custom training loop**: Trains the model with the Adam optimizer using a custom **CVaR (Conditional Value at Risk)** loss — directly optimizing for tail-risk rather than mean-squared error — via `tf.GradientTape`.
11. **Evaluation**: Computes hedging P&L for both the deep hedger and the classical Black-Scholes delta hedger on train/test splits, then compares them via P&L distribution histograms, mean/std, VaR, and CVaR.

## Core P&L Model

The hedger's terminal profit & loss is computed as:

```
PL_T = -Z(S_T) + Σ δ_t · (S_{t+1} - S_t)
```

where `Z(S_T)` is the option's payoff at expiry and the sum is the cumulative gain/loss from dynamically trading the underlying to hedge. Transaction costs are assumed to be zero throughout.

## Tech Stack

- **Data**: pandas, numpy
- **Quant finance**: scipy.stats.norm (Black-Scholes/IV)
- **ML**: TensorFlow / Keras, scikit-learn (train/test split)
- **Visualization**: matplotlib, seaborn

## Input Data Format

Expects an intraday tick/minute-bar CSV with at least:
- `date`, `minute_end` (e.g. `110000` = 11:00:00)
- `symbol` (futures symbols contain `FUT`; option symbols encode strike + `CE`/`PE`)
- `last_trade_price`

## Notes & Assumptions

- Risk-free rate is hardcoded at 5% (`r = 0.05`).
- Option expiry and target strike/symbol are hardcoded for a specific session (Feb 5, 2026, 3:30 PM expiry).
- GBM synthetic paths use a fixed volatility of 0.6 and 1,000 simulated paths.
- The "deep hedger" is a single-step Dense network applied recurrently across time steps (not a true stateful RNN layer).
- No transaction costs are modeled.
- This was generated as a Colab notebook, so it includes Colab-specific cells (`google.colab.files.upload()`, `display()`) that require adaptation to run outside Colab (e.g., replace with local `pd.read_csv()` and `print()`/`IPython.display.display`).

