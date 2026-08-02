import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Master Watchlist Symbols
WATCHLIST_SYMBOLS = [
    "SPY", "QQQ", "DIA", "IWM", "SMH", "VT", "VTI", "TQQQ",
    "GOOGL", "AAPL", "AMZN", "MSFT", "NVDA", "TSLA", "META",
    "NFLX", "AMD", "XLF", "VXX", "SLV", "GLD", "UUP"
]


def format_currency(val):
    """Formats large financial figures cleanly into B (Billions), M (Millions), or T (Trillions)."""
    if val is None or pd.isna(val):
        return "N/A"
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e12:
        return f"{sign}${abs_val / 1e12:.2f}T"
    elif abs_val >= 1e9:
        return f"{sign}${abs_val / 1e9:.2f}B"
    elif abs_val >= 1e6:
        return f"{sign}${abs_val / 1e6:.2f}M"
    else:
        return f"{sign}${abs_val:,.0f}"


def format_multiple(val):
    """Formats valuation ratio multiples cleanly."""
    if val is None or pd.isna(val) or val <= 0:
        return "N/A"
    return f"{val:.2f}x"


def fetch_financial_metrics(ticker_obj):
    """
    Retrieves fundamental TTM financial metrics directly from yfinance financial statements
    (Quarterly/Annual Income Statement, Cash Flow, and Balance Sheet) for maximum accuracy.
    """
    metrics = {
        'market_cap': None, 'sales_ttm': None, 'gross_profit_ttm': None,
        'operating_profit_ttm': None, 'cashflow_ttm': None,
        'p_s': None, 'p_gp': None, 'p_fcf': None, 'ev_gp': None,
        'cash': None, 'total_debt': None, 'ev': None, 'tangible_bv': None
    }
    
    try:
        info = ticker_obj.info or {}
        
        # 1. Primary Market Cap & Enterprise Value from yfinance
        metrics['market_cap'] = info.get('marketCap')
        metrics['ev'] = info.get('enterpriseValue')

        # 2. Extract Data directly from yfinance Financial Statements
        q_financials = ticker_obj.quarterly_financials
        if q_financials.empty:
            q_financials = ticker_obj.financials

        q_cashflow = ticker_obj.quarterly_cashflow
        if q_cashflow.empty:
            q_cashflow = ticker_obj.cashflow

        q_balance = ticker_obj.quarterly_balance_sheet
        if q_balance.empty:
            q_balance = ticker_obj.balance_sheet

        # Helper to safely sum top N periods from yfinance DataFrame
        def get_ttm_sum(df, possible_labels):
            if df is not None and not df.empty:
                for label in possible_labels:
                    if label in df.index:
                        series = df.loc[label].dropna()
                        if not series.empty:
                            take_n = 4 if len(series) >= 4 else len(series)
                            return series.iloc[:take_n].sum()
            return None

        # Helper to get latest single balance sheet item from yfinance DataFrame
        def get_latest_balance_item(df, possible_labels):
            if df is not None and not df.empty:
                for label in possible_labels:
                    if label in df.index:
                        series = df.loc[label].dropna()
                        if not series.empty:
                            return series.iloc[0]
            return None

        # --- TTM INCOME STATEMENT METRICS ---
        metrics['sales_ttm'] = get_ttm_sum(q_financials, ['Total Revenue', 'Operating Revenue'])
        metrics['gross_profit_ttm'] = get_ttm_sum(q_financials, ['Gross Profit'])
        metrics['operating_profit_ttm'] = get_ttm_sum(q_financials, ['Operating Income', 'Total Operating Income'])

        # --- TTM CASH FLOW METRICS ---
        metrics['cashflow_ttm'] = get_ttm_sum(q_cashflow, ['Free Cash Flow', 'Operating Cash Flow', 'Total Cash From Operating Activities'])

        # --- BALANCE SHEET METRICS ---
        metrics['cash'] = get_latest_balance_item(q_balance, ['Cash Cash Equivalents And Short Term Investments', 'Cash And Cash Equivalents'])
        metrics['total_debt'] = get_latest_balance_item(q_balance, ['Total Debt', 'Long Term Debt And Capital Lease Obligation'])

        # Tangible Book Value Calculation directly from Balance Sheet: (Total Assets - Intangibles - Total Liabilities)
        total_assets = get_latest_balance_item(q_balance, ['Total Assets'])
        total_liabilities = get_latest_balance_item(q_balance, ['Total Liabilities Net Minority Interest', 'Total Debt'])
        goodwill_intangibles = get_latest_balance_item(q_balance, ['Other Intangible Assets', 'Goodwill And Other Intangible Assets']) or 0

        if total_assets is not None and total_liabilities is not None:
            metrics['tangible_bv'] = total_assets - total_liabilities - goodwill_intangibles

        # 3. Fallbacks to ticker.info if financial statements are sparse
        if metrics['sales_ttm'] is None:
            metrics['sales_ttm'] = info.get('totalRevenue')
        if metrics['gross_profit_ttm'] is None:
            metrics['gross_profit_ttm'] = info.get('grossProfits')
        if metrics['cashflow_ttm'] is None:
            metrics['cashflow_ttm'] = info.get('freeCashflow') or info.get('operatingCashflow')
        if metrics['cash'] is None:
            metrics['cash'] = info.get('totalCash')
        if metrics['total_debt'] is None:
            metrics['total_debt'] = info.get('totalDebt')
        if metrics['tangible_bv'] is None:
            bv = info.get('bookValue')
            sh = info.get('sharesOutstanding')
            if bv and sh:
                metrics['tangible_bv'] = bv * sh

        # 4. Valuation Multiples Calculation
        mcap = metrics['market_cap']
        ev = metrics['ev']
        gp = metrics['gross_profit_ttm']

        if mcap and mcap > 0:
            if metrics['sales_ttm'] and metrics['sales_ttm'] > 0:
                metrics['p_s'] = mcap / metrics['sales_ttm']
            if gp and gp > 0:
                metrics['p_gp'] = mcap / gp
            if metrics['cashflow_ttm'] and metrics['cashflow_ttm'] > 0:
                metrics['p_fcf'] = mcap / metrics['cashflow_ttm']

        # Enterprise Value / Gross Profit (EV/GP)
        if ev and gp and gp > 0:
            metrics['ev_gp'] = ev / gp

    except Exception as e:
        print(f"Error fetching financial metrics from yfinance: {e}")

    return metrics


def fetch_ma_stats(ticker_symbol):
    """Calculates % distance above/below 200-Day and 200-Week moving averages cleanly."""
    try:
        t = yf.Ticker(ticker_symbol)
        df = t.history(period="5y")
        if df.empty:
            return None, None, None

        closes = df['Close'].dropna()
        if len(closes) < 200:
            return None, None, None

        current_price = closes.iloc[-1]

        dma_series = closes.rolling(window=200).mean().dropna()
        pct_dma_200 = ((current_price - dma_series.iloc[-1]) / dma_series.iloc[-1]) * 100 if not dma_series.empty else None

        weekly_closes = closes.resample('W').last().dropna()
        if len(weekly_closes) >= 200:
            wma_series = weekly_closes.rolling(window=200).mean().dropna()
            pct_wma_200 = ((current_price - wma_series.iloc[-1]) / wma_series.iloc[-1]) * 100 if not wma_series.empty else None
        else:
            pct_wma_200 = None

        return current_price, pct_dma_200, pct_wma_200
    except Exception as e:
        print(f"Error calculating MAs for {ticker_symbol}: {e}")
        return None, None, None


def calculate_iv_and_rank(ticker, chain):
    """Calculates average implied volatility and estimates 1-year IV Rank percentile."""
    try:
        active_ivs = chain[chain['impliedVolatility'] > 0]['impliedVolatility']
        avg_iv = active_ivs.mean() if not active_ivs.empty else None

        hist_1y = ticker.history(period="1y")
        if not hist_1y.empty and len(hist_1y) > 20:
            log_returns = np.log(hist_1y['Close'] / hist_1y['Close'].shift(1))
            rolling_vol = log_returns.rolling(window=21).std() * np.sqrt(252) * 100
            rolling_vol = rolling_vol.dropna()

            if not rolling_vol.empty and avg_iv is not None:
                current_iv_pct = avg_iv * 100
                min_vol, max_vol = rolling_vol.min(), rolling_vol.max()
                iv_rank = ((current_iv_pct - min_vol) / (max_vol - min_vol)) * 100 if max_vol > min_vol else 50.0
                iv_rank = max(0, min(100, iv_rank))
            else:
                iv_rank = None
        else:
            iv_rank = None

        return (avg_iv * 100) if avg_iv is not None else None, iv_rank
    except Exception as e:
        print(f"Error computing IV metrics: {e}")
        return None, None


def get_vix_level():
    """Fetches the latest Cboe Volatility Index (VIX) value."""
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="1d")
        if not hist.empty:
            return hist['Close'].iloc[-1]
    except Exception:
        pass
    return None


def populate_watchlist_table(tree):
    """Fills the Watchlist Treeview table with Bull/Bear option scores & 200 MA metrics."""
    for item in tree.get_children():
        tree.delete(item)

    for sym in WATCHLIST_SYMBOLS:
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="1y")
            if hist.empty:
                continue

            current_price = hist['Close'].iloc[-1]

            closes = hist['Close'].dropna()
            if len(closes) >= 200:
                dma = closes.rolling(200).mean().iloc[-1]
                pct_dma = ((current_price - dma) / dma) * 100
                dma_str = f"{pct_dma:+.1f}%"
            else:
                dma_str = "N/A"

            expirations = t.options
            if expirations:
                sample_exps = expirations[:2]
                v_calls, v_puts = 0, 0
                for exp in sample_exps:
                    try:
                        c = t.option_chain(exp)
                        v_calls += c.calls['volume'].fillna(0).sum()
                        v_puts += c.puts['volume'].fillna(0).sum()
                    except Exception:
                        continue
                tot_v = v_calls + v_puts
                score = (v_calls / tot_v * 100) if tot_v > 0 else 50.0
                sentiment_lbl = f"{score:.0f}% Call"
            else:
                sentiment_lbl = "N/A"

            tree.insert("", tk.END, values=(sym, f"${current_price:.2f}", sentiment_lbl, dma_str))
        except Exception as e:
            print(f"Watchlist row error for {sym}: {e}")
            continue


def get_data_and_plot(symbol, root, frame_chart, frame_stats, frame_financials):
    symbol = symbol.strip().upper()
    if not symbol:
        messagebox.showerror("Error", "Please enter a valid stock symbol.")
        return

    ticker = yf.Ticker(symbol)
    
    # 1. Fetch Historical Price Data
    try:
        hist_5y = ticker.history(period="5y")
        if hist_5y.empty:
            messagebox.showerror("Error", f"Could not fetch data for symbol: '{symbol}'.")
            return
        current_price = hist_5y['Close'].iloc[-1]
    except Exception as e:
        messagebox.showerror("Error", f"Failed to retrieve data: {e}")
        return

    closes_5y = hist_5y['Close'].dropna()
    ma_200_5y = closes_5y.rolling(window=200).mean()
    ma_1000_5y = closes_5y.rolling(window=1000).mean()

    dma_200_val = ma_200_5y.iloc[-1] if len(ma_200_5y.dropna()) > 0 else None
    pct_dma_200 = ((current_price - dma_200_val) / dma_200_val * 100) if dma_200_val else None

    weekly_closes = closes_5y.resample('W').last().dropna()
    wma_200_val = weekly_closes.rolling(window=200).mean().iloc[-1] if len(weekly_closes) >= 200 else None
    pct_wma_200 = ((current_price - wma_200_val) / wma_200_val * 100) if wma_200_val else None

    # 2. Expiration Dates Processing
    expirations = ticker.options
    if not expirations:
        messagebox.showinfo("No Options Data", f"No option chains found for {symbol}.")
        return

    closest_10 = list(expirations[:10])
    jan_expirations = [exp for exp in expirations if exp.split('-')[1] == '01']
    target_expirations = sorted(list(set(closest_10 + jan_expirations)))

    # 3. Pull Option Chains
    all_chains = []
    for exp in target_expirations:
        try:
            opt = ticker.option_chain(exp)
            calls, puts = opt.calls.copy(), opt.puts.copy()
            calls['type'], puts['type'] = 'Call', 'Put'
            calls['expDate'], puts['expDate'] = exp, exp
            all_chains.extend([calls, puts])
        except Exception:
            continue

    if not all_chains:
        messagebox.showerror("Error", "Failed to retrieve option chains.")
        return

    chain = pd.concat(all_chains, ignore_index=True)
    chain['volume'] = chain['volume'].fillna(0)
    chain['openInterest'] = chain['openInterest'].fillna(0)
    chain['impliedVolatility'] = chain['impliedVolatility'].fillna(0)

    if chain['volume'].sum() == 0:
        messagebox.showinfo("No Volume", f"Zero option volume reported for {symbol}.")
        return

    # 4. Option Statistics Calculations
    tot_call_vol = chain[chain['type'] == 'Call']['volume'].sum()
    tot_put_vol = chain[chain['type'] == 'Put']['volume'].sum()
    grand_tot_vol = tot_call_vol + tot_put_vol

    call_vol_pct = (tot_call_vol / grand_tot_vol * 100) if grand_tot_vol > 0 else 0
    put_vol_pct = (tot_put_vol / grand_tot_vol * 100) if grand_tot_vol > 0 else 0

    tot_call_oi = chain[chain['type'] == 'Call']['openInterest'].sum()
    tot_put_oi = chain[chain['type'] == 'Put']['openInterest'].sum()
    grand_tot_oi = tot_call_oi + tot_put_oi

    vol_oi_ratio = (grand_tot_vol / grand_tot_oi) if grand_tot_oi > 0 else 0.0
    call_oi_pct = (tot_call_oi / grand_tot_oi * 100) if grand_tot_oi > 0 else 0
    put_oi_pct = (tot_put_oi / grand_tot_oi * 100) if grand_tot_oi > 0 else 0

    avg_iv, iv_rank = calculate_iv_and_rank(ticker, chain)

    # 5. Top Strikes Filter
    tot_vol_by_strike = chain.groupby('strike')['volume'].sum()
    vol_75th_percentile = tot_vol_by_strike.quantile(0.75)
    active_strikes = tot_vol_by_strike[tot_vol_by_strike >= vol_75th_percentile].index
    filtered_chain = chain[chain['strike'].isin(active_strikes)]

    # 6. Fetch Fundamental Financial Data Directly from yfinance Statements
    fin_data = fetch_financial_metrics(ticker)

    tot_vol = filtered_chain.groupby('strike')['volume'].sum()
    max_vol_strike = tot_vol.idxmax() if not tot_vol.empty else 0
    max_vol_val = tot_vol.max() if not tot_vol.empty else 0
    top_5_strikes = tot_vol.nlargest(5).index.tolist() if not tot_vol.empty else []

    top_5_exp_dates = {}
    for stk in top_5_strikes:
        stk_chain = chain[chain['strike'] == stk]
        top_5_exp_dates[stk] = stk_chain.groupby('expDate')['volume'].sum().idxmax() if not stk_chain.empty else "N/A"

    max_vol_chain = chain[chain['strike'] == max_vol_strike]
    max_exp_row = max_vol_chain.groupby('expDate')['volume'].sum().idxmax() if not max_vol_chain.empty else "N/A"
    exp_detail_str = f"{max_exp_row} ({pd.to_datetime(max_exp_row).strftime('%B %Y')})" if max_exp_row != "N/A" else "N/A"

    spy_price, spy_dma, spy_wma = fetch_ma_stats("SPY")
    qqq_price, qqq_dma, qqq_wma = fetch_ma_stats("QQQ")
    vix_val = get_vix_level()

    # Clear UI Elements
    for widget in frame_chart.winfo_children():
        widget.destroy()
    for widget in frame_stats.winfo_children():
        widget.destroy()
    for widget in frame_financials.winfo_children():
        widget.destroy()

    # --- Render Options & Market Stats Header ---
    txt = tk.Text(frame_stats, height=13, relief="flat", background=root.cget("bg"), font=("Consolas", 10))
    txt.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    txt.tag_config("green", foreground="#008000", font=("Consolas", 10, "bold"))
    txt.tag_config("red", foreground="#CC0000", font=("Consolas", 10, "bold"))
    txt.tag_config("bold", font=("Consolas", 10, "bold"))
    txt.tag_config("strike", foreground="#8A2BE2", font=("Consolas", 10, "bold"))

    vix_str = f"{vix_val:.2f}" if vix_val is not None else "N/A"
    txt.insert(tk.END, f"MARKET OVERVIEW  |  VIX Index: {vix_str}\n")
    
    txt.insert(tk.END, "SPY 200-Day MA: ")
    txt.insert(tk.END, f"{spy_dma:+.2f}%\t" if spy_dma else "N/A\t", "green" if (spy_dma and spy_dma >= 0) else "red")
    txt.insert(tk.END, "SPY 200-Week MA: ")
    txt.insert(tk.END, f"{spy_wma:+.2f}%\n" if spy_wma else "N/A\n", "green" if (spy_wma and spy_wma >= 0) else "red")

    txt.insert(tk.END, "QQQ 200-Day MA: ")
    txt.insert(tk.END, f"{qqq_dma:+.2f}%\t" if qqq_dma else "N/A\t", "green" if (qqq_dma and qqq_dma >= 0) else "red")
    txt.insert(tk.END, "QQQ 200-Week MA: ")
    txt.insert(tk.END, f"{qqq_wma:+.2f}%\n" if qqq_wma else "N/A\n", "green" if (qqq_wma and qqq_wma >= 0) else "red")

    txt.insert(tk.END, "-----------------------------------------------------------------------------------------\n")

    sentiment_status = "BULLISH" if call_vol_pct > 50 else "BEARISH"
    sentiment_tag = "green" if call_vol_pct > 50 else "red"

    call_blocks = int(round((call_vol_pct / 100.0) * 25))
    sentiment_bar = "█" * call_blocks + "░" * (25 - call_blocks)

    iv_str = f"{avg_iv:.1f}%" if avg_iv else "N/A"
    iv_rank_str = f"{iv_rank:.1f}%" if iv_rank is not None else "N/A"
    dma_str = f"{pct_dma_200:+.2f}%" if pct_dma_200 is not None else "N/A"
    wma_str = f"{pct_wma_200:+.2f}%" if pct_wma_200 is not None else "N/A"

    txt.insert(tk.END, f"Symbol: {symbol} (${current_price:.2f}) | 200-DMA: {dma_str} | 200-WMA: {wma_str}\n")
    txt.insert(tk.END, f"★ Implied Volatility (IV): {iv_str}  |  IV Rank: {iv_rank_str}  |  Vol/OI Ratio: {vol_oi_ratio:.2f}x\n")
    
    txt.insert(tk.END, f"★ Bull/Bear Score: [")
    txt.insert(tk.END, sentiment_bar, sentiment_tag)
    txt.insert(tk.END, f"] {call_vol_pct:.1f}% Call / {put_vol_pct:.1f}% Put  [")
    txt.insert(tk.END, sentiment_status, sentiment_tag)
    txt.insert(tk.END, "]\n")

    txt.insert(tk.END, f"★ Total Open Interest Ratio: {call_oi_pct:.1f}% Calls / {put_oi_pct:.1f}% Puts\n")
    
    top_5_str_list = []
    plot_df = filtered_chain.groupby(['strike', 'type'])['volume'].sum().unstack(fill_value=0)
    for stk in top_5_strikes:
        c_v = plot_df.loc[stk, 'Call'] if ('Call' in plot_df.columns and stk in plot_df.index) else 0
        p_v = plot_df.loc[stk, 'Put'] if ('Put' in plot_df.columns and stk in plot_df.index) else 0
        exp_dt = top_5_exp_dates.get(stk, "N/A")
        top_5_str_list.append(f"${stk:g} ({exp_dt}) [{int(c_v):,}C/{int(p_v):,}P]")
    
    txt.insert(tk.END, f"★ TOP 5 VOLUME STRIKES: ", "strike")
    txt.insert(tk.END, f"{' | '.join(top_5_str_list)}\n")
    txt.insert(tk.END, f"★ Peak Strike: ${max_vol_strike} ({max_vol_val:,.0f} contracts) | Peak Exp: {exp_detail_str}")

    txt.config(state="disabled")

    # --- Side-by-Side Charts ---
    integer_dollar_formatter = FuncFormatter(lambda x, pos: f"${int(x):,}")
    fig, (ax_5y, ax_profile) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=100)
    strike_colors = ['#800080', '#D2691E', '#1E90FF', '#008080', '#FF1493']

    ax_5y.plot(closes_5y.index, closes_5y, label='Close Price', color='#1f77b4', linewidth=1.2)
    ax_5y.plot(ma_200_5y.index, ma_200_5y, label='200-Day MA', color='#ff7f0e', linewidth=1.5)
    ax_5y.plot(ma_1000_5y.index, ma_1000_5y, label='1000-Day MA', color='#9467bd', linewidth=1.5)
    
    ax_5y.yaxis.set_major_formatter(integer_dollar_formatter)
    ax_5y.yaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
    ax_5y.axhline(current_price, color='black', linestyle=':', linewidth=1.2, label=f'Current (${current_price:.2f})')
    
    for idx, strike in enumerate(top_5_strikes):
        color = strike_colors[idx % len(strike_colors)]
        exp_date_lbl = top_5_exp_dates.get(strike, "")
        ax_5y.axhline(strike, color=color, linestyle='--', linewidth=1.0, alpha=0.7, label=f'Strike ${strike:g} ({exp_date_lbl})')

    ax_5y.set_title(f"{symbol} 5-Year Price History", fontsize=10, pad=8)
    ax_5y.set_ylabel("Price ($)", fontsize=9)
    ax_5y.legend(loc='upper left', fontsize=7, framealpha=0.8)
    ax_5y.grid(True, linestyle='--', alpha=0.5)

    plot_df = plot_df.sort_index(ascending=True)
    strikes = plot_df.index
    calls_vol = plot_df.get('Call', pd.Series(0, index=strikes))
    puts_vol = plot_df.get('Put', pd.Series(0, index=strikes))

    ax_profile.plot(calls_vol, strikes, color='green', marker='o', linewidth=2, markersize=4, label='Calls Volume')
    ax_profile.plot(puts_vol, strikes, color='red', marker='o', linewidth=2, markersize=4, label='Puts Volume')

    ax_profile.yaxis.set_major_formatter(integer_dollar_formatter)
    ax_profile.yaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
    ax_profile.axhline(current_price, color='black', linestyle=':', linewidth=1.5, label=f'Current (${current_price:.2f})')

    for idx, strike in enumerate(top_5_strikes):
        color = strike_colors[idx % len(strike_colors)]
        ax_profile.axhline(strike, color=color, linestyle='--', linewidth=1.0, alpha=0.6)

    max_vol_limit = max(calls_vol.max() if not calls_vol.empty else 1, puts_vol.max() if not puts_vol.empty else 1)
    
    for i, strike in enumerate(top_5_strikes):
        c_v = calls_vol.get(strike, 0)
        p_v = puts_vol.get(strike, 0)
        peak_x = max(c_v, p_v)
        exp_dt = top_5_exp_dates.get(strike, "N/A")
        label_color = 'green' if c_v >= p_v else 'red'
        
        ax_profile.annotate(
            f'★ ${strike:g} ({exp_dt})\n({int(c_v):,} C / {int(p_v):,} P)', 
            xy=(peak_x, strike),
            xytext=(peak_x * 1.15, strike),
            arrowprops=dict(facecolor=label_color, shrink=0.08, headwidth=4, width=1, alpha=0.7),
            va='center', ha='left', fontsize=7, fontweight='bold', color=label_color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=label_color, lw=0.8, alpha=0.85)
        )

    ax_profile.set_xlim(left=0, right=max_vol_limit * 1.55)
    ax_profile.set_xlabel("Volume (Contracts)", fontsize=9)
    ax_profile.set_ylabel("Strike Price ($)", fontsize=9)
    ax_profile.set_title(f"{symbol} Options Profile (Top 5 Strikes Marked)", fontsize=10, pad=8)
    ax_profile.legend(loc='lower right', fontsize=8)
    ax_profile.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=frame_chart)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- Render Financial Data & New EV/GP Bar Below Charts ---
    txt_fin = tk.Text(frame_financials, height=8, relief="flat", background=root.cget("bg"), font=("Consolas", 10))
    txt_fin.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    txt_fin.tag_config("title", font=("Consolas", 10, "bold"), foreground="#000080")
    txt_fin.tag_config("val", font=("Consolas", 10, "bold"), foreground="#008000")
    txt_fin.tag_config("ratio", font=("Consolas", 10, "bold"), foreground="#D2691E")
    txt_fin.tag_config("super_green", font=("Consolas", 10, "bold"), foreground="#009900")
    txt_fin.tag_config("green_tag", font=("Consolas", 10, "bold"), foreground="#2E8B57")
    txt_fin.tag_config("orange_tag", font=("Consolas", 10, "bold"), foreground="#FF8C00")
    txt_fin.tag_config("red_tag", font=("Consolas", 10, "bold"), foreground="#CC0000")

    # Line 1: TTM Income & Cash Statements from yfinance
    txt_fin.insert(tk.END, "FINANCIAL METRICS (TTM): ", "title")
    txt_fin.insert(tk.END, f"Market Cap: ")
    txt_fin.insert(tk.END, f"{format_currency(fin_data['market_cap'])}\t", "val")
    txt_fin.insert(tk.END, f"Sales: ")
    txt_fin.insert(tk.END, f"{format_currency(fin_data['sales_ttm'])}\t", "val")
    txt_fin.insert(tk.END, f"Gross Profit: ")
    txt_fin.insert(tk.END, f"{format_currency(fin_data['gross_profit_ttm'])}\t", "val")
    txt_fin.insert(tk.END, f"Operating Profit: ")
    txt_fin.insert(tk.END, f"{format_currency(fin_data['operating_profit_ttm'])}\t", "val")
    txt_fin.insert(tk.END, f"Cash Flow: ")
    txt_fin.insert(tk.END, f"{format_currency(fin_data['cashflow_ttm'])}\n", "val")

    # Line 2: Valuation Multiples
    txt_fin.insert(tk.END, "VALUATION MULTIPLES:    ", "title")
    txt_fin.insert(tk.END, f"Price / Sales (P/S): ")
    txt_fin.insert(tk.END, f"{format_multiple(fin_data['p_s'])}\t\t", "ratio")
    txt_fin.insert(tk.END, f"Price / Gross Profit (P/GP): ")
    txt_fin.insert(tk.END, f"{format_multiple(fin_data['p_gp'])}\t\t", "ratio")
    txt_fin.insert(tk.END, f"Price / Cash Flow (P/CF): ")
    txt_fin.insert(tk.END, f"{format_multiple(fin_data['p_fcf'])}\n", "ratio")

    txt_fin.insert(tk.END, "-----------------------------------------------------------------------------------------\n")

    # Line 3: Balance Sheet & Enterprise Valuation
    txt_fin.insert(tk.END, "BALANCE SHEET & EV:     ", "title")
    txt_fin.insert(tk.END, f"Cash & Equivalents: ")
    txt_fin.insert(tk.END, f"{format_currency(fin_data['cash'])}\t", "val")
    txt_fin.insert(tk.END, f"Total Debt: ")
    txt_fin.insert(tk.END, f"{format_currency(fin_data['total_debt'])}\t", "val")
    txt_fin.insert(tk.END, f"Enterprise Value (EV): ")
    txt_fin.insert(tk.END, f"{format_currency(fin_data['ev'])}\t", "val")
    txt_fin.insert(tk.END, f"Tangible Book Value: ")
    txt_fin.insert(tk.END, f"{format_currency(fin_data['tangible_bv'])}\n", "val")

    # Line 4: NEW EV / GROSS PROFIT HEALTH BAR
    ev_gp = fin_data['ev_gp']
    txt_fin.insert(tk.END, "EV / GROSS PROFIT SCORE: ", "title")

    if ev_gp is not None and ev_gp > 0:
        # Scale: 0x to 50x maps onto 25 visual blocks
        bounded_val = min(ev_gp, 50.0)
        filled_blocks = int(round((bounded_val / 50.0) * 25))
        ev_gp_bar = "█" * filled_blocks + "░" * (25 - filled_blocks)

        if ev_gp < 7.0:
            ev_status = "SUPER GREEN (< 7x EV/GP)"
            tag_name = "super_green"
        elif ev_gp <= 20.0:
            ev_status = "HEALTHY (7x - 20x EV/GP)"
            tag_name = "green_tag"
        elif ev_gp <= 50.0:
            ev_status = "ELEVATED (20x - 50x EV/GP)"
            tag_name = "orange_tag"
        else:
            ev_status = "RED / HIGH OVERVALUATION (> 50x EV/GP)"
            tag_name = "red_tag"

        txt_fin.insert(tk.END, f"[")
        txt_fin.insert(tk.END, ev_gp_bar, tag_name)
        txt_fin.insert(tk.END, f"] {ev_gp:.2f}x  [")
        txt_fin.insert(tk.END, ev_status, tag_name)
        txt_fin.insert(tk.END, "]")
    else:
        txt_fin.insert(tk.END, "N/A (Negative or Missing Gross Profit/EV)", "red_tag")

    txt_fin.config(state="disabled")


def build_app():
    root = tk.Tk()
    root.title("Options Volume & Financial Analytics Dashboard")
    root.geometry("1450x980")

    # Split Panes (Left = Scrollable Analysis Area, Right = Watchlist Table)
    paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
    paned_window.pack(fill=tk.BOTH, expand=True)

    frame_left_container = ttk.Frame(paned_window)
    frame_sidebar = ttk.Frame(paned_window, width=320)

    paned_window.add(frame_left_container, weight=4)
    paned_window.add(frame_sidebar, weight=1)

    # Scrollable Left Container Canvas
    canvas_main = tk.Canvas(frame_left_container, highlightthickness=0)
    scrollbar_main = ttk.Scrollbar(frame_left_container, orient="vertical", command=canvas_main.yview)
    frame_main = ttk.Frame(canvas_main)

    frame_main.bind("<Configure>", lambda e: canvas_main.configure(scrollregion=canvas_main.bbox("all")))
    canvas_main.create_window((0, 0), window=frame_main, anchor="nw")
    canvas_main.configure(yscrollcommand=scrollbar_main.set)

    canvas_main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar_main.pack(side=tk.RIGHT, fill=tk.Y)

    # Control Bar
    frame_controls = ttk.Frame(frame_main, padding=10)
    frame_controls.pack(side=tk.TOP, fill=tk.X)

    ttk.Label(frame_controls, text="Enter Symbol:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=3)

    entry_symbol = ttk.Entry(frame_controls, width=8, font=("Arial", 10))
    entry_symbol.insert(0, "SPY")
    entry_symbol.pack(side=tk.LEFT, padx=3)

    entry_symbol.bind("<Return>", lambda event: get_data_and_plot(entry_symbol.get(), root, frame_chart, frame_stats, frame_financials))

    # Scrollable Quick Select Bar
    ttk.Label(frame_controls, text="Quick Select:").pack(side=tk.LEFT, padx=(10, 2))
    
    canvas_qs = tk.Canvas(frame_controls, height=35, highlightthickness=0)
    scrollbar_qs = ttk.Scrollbar(frame_controls, orient="horizontal", command=canvas_qs.xview)
    scrollable_qs_frame = ttk.Frame(canvas_qs)

    scrollable_qs_frame.bind("<Configure>", lambda e: canvas_qs.configure(scrollregion=canvas_qs.bbox("all")))
    canvas_qs.create_window((0, 0), window=scrollable_qs_frame, anchor="nw")
    canvas_qs.configure(xscrollcommand=scrollbar_qs.set)

    canvas_qs.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

    for sym in WATCHLIST_SYMBOLS:
        btn = ttk.Button(
            scrollable_qs_frame, 
            text=sym, 
            width=5, 
            command=lambda s=sym: [entry_symbol.delete(0, tk.END), entry_symbol.insert(0, s), get_data_and_plot(s, root, frame_chart, frame_stats, frame_financials)]
        )
        btn.pack(side=tk.LEFT, padx=1)

    btn_fetch = ttk.Button(
        frame_controls, 
        text="Load", 
        command=lambda: get_data_and_plot(entry_symbol.get(), root, frame_chart, frame_stats, frame_financials)
    )
    btn_fetch.pack(side=tk.RIGHT, padx=5)

    # Header Stats Frame
    frame_stats = ttk.LabelFrame(frame_main, text="Market Metrics & Options Summary", padding=10)
    frame_stats.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

    # Chart Canvas Frame
    frame_chart = ttk.Frame(frame_main, padding=10)
    frame_chart.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Financial Metrics Frame (Below Charts)
    frame_financials = ttk.LabelFrame(frame_main, text="Company Fundamentals & EV/GP Valuation Metrics", padding=10)
    frame_financials.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

    # Sidebar Watchlist Summary Table
    ttk.Label(frame_sidebar, text="★ WATCHLIST METRICS SUMMARY", font=("Arial", 10, "bold")).pack(side=tk.TOP, pady=8)

    columns = ("symbol", "price", "bull_score", "dma_200")
    tree_watchlist = ttk.Treeview(frame_sidebar, columns=columns, show="headings", height=25)
    
    tree_watchlist.heading("symbol", text="Symbol")
    tree_watchlist.heading("price", text="Price")
    tree_watchlist.heading("bull_score", text="Bull Score")
    tree_watchlist.heading("dma_200", text="200-DMA")

    tree_watchlist.column("symbol", width=33, anchor="center")
    tree_watchlist.column("price", width=33, anchor="center")
    tree_watchlist.column("bull_score", width=33, anchor="center")
    tree_watchlist.column("dma_200", width=33, anchor="center")

    sb_tree = ttk.Scrollbar(frame_sidebar, orient="vertical", command=tree_watchlist.yview)
    tree_watchlist.configure(yscrollcommand=sb_tree.set)

    tree_watchlist.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
    sb_tree.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

    def on_tree_select(event):
        selected_item = tree_watchlist.selection()
        if selected_item:
            item_vals = tree_watchlist.item(selected_item[0], 'values')
            sym = item_vals[0]
            entry_symbol.delete(0, tk.END)
            entry_symbol.insert(0, sym)
            get_data_and_plot(sym, root, frame_chart, frame_stats, frame_financials)

    tree_watchlist.bind("<Double-1>", on_tree_select)

    btn_refresh_wl = ttk.Button(
        frame_sidebar, 
        text="Refresh Watchlist Table", 
        command=lambda: populate_watchlist_table(tree_watchlist)
    )
    btn_refresh_wl.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

    # Initial Loading Calls
    root.after(100, lambda: get_data_and_plot("SPY", root, frame_chart, frame_stats, frame_financials))
    root.after(500, lambda: populate_watchlist_table(tree_watchlist))

    root.mainloop()


if __name__ == "__main__":
    build_app()