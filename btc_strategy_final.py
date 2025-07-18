import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FinalBitcoinStrategy:
    def __init__(self, symbol='BTC-USD', initial_capital=100000):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.data = None
        self.signals = None
        self.results = None
        
    def fetch_data(self, period='2y'):
        """獲取比特幣歷史數據"""
        try:
            self.data = yf.download(self.symbol, period=period, interval='1d')
            # 如果是多層索引，展平列名
            if isinstance(self.data.columns, pd.MultiIndex):
                self.data.columns = [col[0] for col in self.data.columns]
            print(f"成功獲取 {self.symbol} 數據，共 {len(self.data)} 天")
            return True
        except Exception as e:
            print(f"獲取數據失敗: {e}")
            return False
    
    def calculate_indicators(self):
        """計算技術指標"""
        df = self.data.copy()
        
        # 移動平均線系統
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # 指數移動平均線
        df['EMA_5'] = df['Close'].ewm(span=5).mean()
        df['EMA_10'] = df['Close'].ewm(span=10).mean()
        df['EMA_20'] = df['Close'].ewm(span=20).mean()
        
        # 快速RSI (5日和10日)
        for period in [5, 10, 14]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
            rs = gain / loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD (多個週期)
        for fast, slow, signal in [(5, 10, 3), (8, 21, 5), (12, 26, 9)]:
            ema_fast = df['Close'].ewm(span=fast).mean()
            ema_slow = df['Close'].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            df[f'MACD_{fast}_{slow}'] = macd
            df[f'MACD_Signal_{fast}_{slow}'] = macd_signal
            df[f'MACD_Hist_{fast}_{slow}'] = macd - macd_signal
        
        # 布林帶 (多個週期)
        for period in [10, 20]:
            bb_middle = df['Close'].rolling(window=period).mean()
            bb_std = df['Close'].rolling(window=period).std()
            df[f'BB_Upper_{period}'] = bb_middle + (bb_std * 2)
            df[f'BB_Lower_{period}'] = bb_middle - (bb_std * 2)
            df[f'BB_Position_{period}'] = (df['Close'] - df[f'BB_Lower_{period}']) / (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}'])
        
        # 動量指標
        df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # 成交量指標
        df['Volume_SMA_5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio_5'] = df['Volume'] / df['Volume_SMA_5']
        df['Volume_Ratio_10'] = df['Volume'] / df['Volume_SMA_10']
        
        # 價格變化率
        df['Price_Change_1'] = df['Close'].pct_change(1)
        df['Price_Change_3'] = df['Close'].pct_change(3)
        df['Price_Change_5'] = df['Close'].pct_change(5)
        
        # 波動率
        df['Volatility_10'] = df['Close'].rolling(window=10).std() / df['Close'].rolling(window=10).mean()
        df['Volatility_20'] = df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean()
        
        # ATR
        df['TR'] = np.maximum(df['High'] - df['Low'], 
                             np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                       abs(df['Low'] - df['Close'].shift(1))))
        df['ATR_10'] = df['TR'].rolling(window=10).mean()
        df['ATR_20'] = df['TR'].rolling(window=20).mean()
        
        # 趨勢強度指標
        df['Trend_Strength'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
        
        # 相對價格位置
        df['Price_Position_10'] = (df['Close'] - df['Close'].rolling(window=10).min()) / (df['Close'].rolling(window=10).max() - df['Close'].rolling(window=10).min())
        df['Price_Position_20'] = (df['Close'] - df['Close'].rolling(window=20).min()) / (df['Close'].rolling(window=20).max() - df['Close'].rolling(window=20).min())
        
        self.data = df
        
    def generate_signals(self):
        """生成交易信號（最終優化版本）"""
        df = self.data.copy()
        
        # 初始化信號
        df['Signal'] = 0
        df['Position'] = 0
        df['Entry_Price'] = 0
        df['Stop_Loss'] = 0
        df['Take_Profit'] = 0
        df['Position_Size'] = 0
        
        # 市場環境判斷
        df['Bull_Market'] = (df['Close'] > df['SMA_50']) & (df['SMA_50'] > df['SMA_200'])
        df['Bear_Market'] = (df['Close'] < df['SMA_50']) & (df['SMA_50'] < df['SMA_200'])
        df['Sideways'] = ~df['Bull_Market'] & ~df['Bear_Market']
        
        # 動態止損比例
        base_stop = 0.02  # 2%基準止損
        
        # 超強多頭信號條件
        ultra_long_condition = (
            df['Bull_Market'] &
            (df['Close'] > df['EMA_5']) &
            (df['EMA_5'] > df['EMA_10']) &
            (df['EMA_10'] > df['EMA_20']) &
            (df['RSI_5'] > 30) & (df['RSI_5'] < 80) &
            (df['RSI_10'] > 40) & (df['RSI_10'] < 75) &
            (df['MACD_5_10'] > df['MACD_Signal_5_10']) &
            (df['MACD_8_21'] > df['MACD_Signal_8_21']) &
            (df['MACD_Hist_5_10'] > 0) &
            (df['BB_Position_10'] > 0.2) & (df['BB_Position_10'] < 0.8) &
            (df['Momentum_5'] > 0.01) &
            (df['Volume_Ratio_5'] > 1.5) &
            (df['Price_Position_10'] > 0.6) &
            (df['Trend_Strength'] > 0.02)
        )
        
        # 強多頭信號條件
        strong_long_condition = (
            df['Bull_Market'] &
            (df['Close'] > df['EMA_10']) &
            (df['EMA_10'] > df['EMA_20']) &
            (df['RSI_10'] > 35) & (df['RSI_10'] < 70) &
            (df['MACD_8_21'] > df['MACD_Signal_8_21']) &
            (df['BB_Position_20'] > 0.3) & (df['BB_Position_20'] < 0.7) &
            (df['Momentum_10'] > 0) &
            (df['Volume_Ratio_10'] > 1.2) &
            (df['Price_Position_20'] > 0.5)
        )
        
        # 超強空頭信號條件
        ultra_short_condition = (
            df['Bear_Market'] &
            (df['Close'] < df['EMA_5']) &
            (df['EMA_5'] < df['EMA_10']) &
            (df['EMA_10'] < df['EMA_20']) &
            (df['RSI_5'] < 70) & (df['RSI_5'] > 20) &
            (df['RSI_10'] < 60) & (df['RSI_10'] > 25) &
            (df['MACD_5_10'] < df['MACD_Signal_5_10']) &
            (df['MACD_8_21'] < df['MACD_Signal_8_21']) &
            (df['MACD_Hist_5_10'] < 0) &
            (df['BB_Position_10'] < 0.8) & (df['BB_Position_10'] > 0.2) &
            (df['Momentum_5'] < -0.01) &
            (df['Volume_Ratio_5'] > 1.5) &
            (df['Price_Position_10'] < 0.4) &
            (df['Trend_Strength'] < -0.02)
        )
        
        # 強空頭信號條件
        strong_short_condition = (
            df['Bear_Market'] &
            (df['Close'] < df['EMA_10']) &
            (df['EMA_10'] < df['EMA_20']) &
            (df['RSI_10'] < 65) & (df['RSI_10'] > 30) &
            (df['MACD_8_21'] < df['MACD_Signal_8_21']) &
            (df['BB_Position_20'] < 0.7) & (df['BB_Position_20'] > 0.3) &
            (df['Momentum_10'] < 0) &
            (df['Volume_Ratio_10'] > 1.2) &
            (df['Price_Position_20'] < 0.5)
        )
        
        # 出場條件
        def get_exit_condition(df, position_type):
            if position_type == 'long':
                return (
                    (df['RSI_5'] > 85) |
                    (df['RSI_10'] > 80) |
                    (df['MACD_5_10'] < df['MACD_Signal_5_10']) |
                    (df['Close'] < df['EMA_5']) |
                    (df['BB_Position_10'] < 0.1) |
                    (df['Bear_Market'])
                )
            else:  # short
                return (
                    (df['RSI_5'] < 15) |
                    (df['RSI_10'] < 20) |
                    (df['MACD_5_10'] > df['MACD_Signal_5_10']) |
                    (df['Close'] > df['EMA_5']) |
                    (df['BB_Position_10'] > 0.9) |
                    (df['Bull_Market'])
                )
        
        # 動態倉位管理
        def get_position_size(df, index, signal_strength):
            base_size = 1.0
            
            if signal_strength == 'ultra':
                return base_size * 1.5  # 超強信號使用1.5倍倉位
            elif signal_strength == 'strong':
                return base_size * 1.0  # 強信號使用標準倉位
            else:
                return base_size * 0.5  # 普通信號使用0.5倍倉位
        
        # 生成交易信號
        for i in range(50, len(df)):  # 從第50天開始，確保指標計算完整
            current_position = df.iloc[i-1]['Position']
            current_price = df.iloc[i]['Close']
            
            if current_position == 0:  # 無持倉
                signal_strength = None
                position_size = 0
                
                # 檢查多頭信號
                if ultra_long_condition.iloc[i]:
                    signal_strength = 'ultra'
                    position_size = get_position_size(df, i, 'ultra')
                    df.iloc[i, df.columns.get_loc('Signal')] = 1
                    df.iloc[i, df.columns.get_loc('Position')] = position_size
                elif strong_long_condition.iloc[i]:
                    signal_strength = 'strong'
                    position_size = get_position_size(df, i, 'strong')
                    df.iloc[i, df.columns.get_loc('Signal')] = 1
                    df.iloc[i, df.columns.get_loc('Position')] = position_size
                
                # 檢查空頭信號
                elif ultra_short_condition.iloc[i]:
                    signal_strength = 'ultra'
                    position_size = get_position_size(df, i, 'ultra')
                    df.iloc[i, df.columns.get_loc('Signal')] = -1
                    df.iloc[i, df.columns.get_loc('Position')] = -position_size
                elif strong_short_condition.iloc[i]:
                    signal_strength = 'strong'
                    position_size = get_position_size(df, i, 'strong')
                    df.iloc[i, df.columns.get_loc('Signal')] = -1
                    df.iloc[i, df.columns.get_loc('Position')] = -position_size
                
                # 如果有信號，設置止損止盈
                if signal_strength:
                    atr_multiplier = 1.5 if signal_strength == 'ultra' else 1.0
                    stop_multiplier = base_stop * atr_multiplier
                    
                    df.iloc[i, df.columns.get_loc('Entry_Price')] = current_price
                    df.iloc[i, df.columns.get_loc('Position_Size')] = abs(position_size)
                    
                    if position_size > 0:  # 多頭
                        df.iloc[i, df.columns.get_loc('Stop_Loss')] = current_price * (1 - stop_multiplier)
                        df.iloc[i, df.columns.get_loc('Take_Profit')] = current_price * (1 + stop_multiplier * 3)
                    else:  # 空頭
                        df.iloc[i, df.columns.get_loc('Stop_Loss')] = current_price * (1 + stop_multiplier)
                        df.iloc[i, df.columns.get_loc('Take_Profit')] = current_price * (1 - stop_multiplier * 3)
                        
            else:  # 有持倉
                previous_position = df.iloc[i-1]['Position']
                stop_loss = df.iloc[i-1]['Stop_Loss']
                take_profit = df.iloc[i-1]['Take_Profit']
                
                # 檢查止損止盈
                should_exit = False
                if previous_position > 0:  # 多頭持倉
                    if current_price <= stop_loss or current_price >= take_profit:
                        should_exit = True
                    elif get_exit_condition(df, 'long').iloc[i]:
                        should_exit = True
                else:  # 空頭持倉
                    if current_price >= stop_loss or current_price <= take_profit:
                        should_exit = True
                    elif get_exit_condition(df, 'short').iloc[i]:
                        should_exit = True
                
                if should_exit:
                    df.iloc[i, df.columns.get_loc('Signal')] = -1 if previous_position > 0 else 1
                    df.iloc[i, df.columns.get_loc('Position')] = 0
                    df.iloc[i, df.columns.get_loc('Entry_Price')] = 0
                    df.iloc[i, df.columns.get_loc('Stop_Loss')] = 0
                    df.iloc[i, df.columns.get_loc('Take_Profit')] = 0
                    df.iloc[i, df.columns.get_loc('Position_Size')] = 0
                else:
                    # 保持持倉
                    df.iloc[i, df.columns.get_loc('Position')] = previous_position
                    df.iloc[i, df.columns.get_loc('Entry_Price')] = df.iloc[i-1]['Entry_Price']
                    df.iloc[i, df.columns.get_loc('Stop_Loss')] = stop_loss
                    df.iloc[i, df.columns.get_loc('Take_Profit')] = take_profit
                    df.iloc[i, df.columns.get_loc('Position_Size')] = df.iloc[i-1]['Position_Size']
        
        self.signals = df
        
    def calculate_returns(self):
        """計算策略回報"""
        df = self.signals.copy()
        
        # 計算每日收益率
        df['Daily_Return'] = df['Close'].pct_change()
        
        # 計算策略收益率（考慮倉位大小）
        df['Strategy_Return'] = df['Daily_Return'] * df['Position'].shift(1)
        
        # 計算累積收益
        df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
        df['Cumulative_Market_Return'] = (1 + df['Daily_Return']).cumprod()
        
        # 計算資金曲線
        df['Portfolio_Value'] = self.initial_capital * df['Cumulative_Return']
        
        # 計算回撤
        df['Peak'] = df['Portfolio_Value'].expanding().max()
        df['Drawdown'] = (df['Portfolio_Value'] - df['Peak']) / df['Peak'] * 100
        
        self.results = df
        
    def analyze_performance(self):
        """分析策略表現"""
        df = self.results.copy()
        
        # 基本統計
        total_trades = len(df[df['Signal'] != 0])
        total_return = (df['Portfolio_Value'].iloc[-1] / self.initial_capital - 1) * 100
        
        # 計算年化報酬率
        trading_days = len(df.dropna())
        years = trading_days / 252
        cagr = (df['Portfolio_Value'].iloc[-1] / self.initial_capital) ** (1/years) - 1
        
        # 最大回撤
        max_drawdown = df['Drawdown'].min()
        
        # 夏普比率
        strategy_returns = df['Strategy_Return'].dropna()
        if len(strategy_returns) > 0 and strategy_returns.std() > 0:
            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # 勝率計算
        trades = df[df['Signal'] != 0].copy()
        if len(trades) > 0:
            trade_returns = []
            entry_price = None
            entry_signal = None
            
            for i, row in trades.iterrows():
                if entry_price is None:
                    entry_price = row['Close']
                    entry_signal = row['Signal']
                else:
                    if (entry_signal == 1 and row['Signal'] == -1) or \
                       (entry_signal == -1 and row['Signal'] == 1):
                        # 平倉
                        position_size = abs(df.loc[trades.index[trades.index.get_loc(i)-1], 'Position']) if i in trades.index else 1
                        if entry_signal == 1:  # 多頭交易
                            trade_return = (row['Close'] - entry_price) / entry_price * position_size
                        else:  # 空頭交易
                            trade_return = (entry_price - row['Close']) / entry_price * position_size
                        trade_returns.append(trade_return)
                        entry_price = None
                        entry_signal = None
            
            if len(trade_returns) > 0:
                win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns)
                avg_win = np.mean([r for r in trade_returns if r > 0]) if any(r > 0 for r in trade_returns) else 0
                avg_loss = np.mean([r for r in trade_returns if r < 0]) if any(r < 0 for r in trade_returns) else 0
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            else:
                win_rate = 0
                profit_factor = 0
        else:
            win_rate = 0
            profit_factor = 0
        
        # 波動率
        volatility = strategy_returns.std() * np.sqrt(252) if len(strategy_returns) > 0 else 0
        
        performance_metrics = {
            'Total Trades': total_trades,
            'Total Return (%)': round(total_return, 2),
            'CAGR (%)': round(cagr * 100, 2),
            'Max Drawdown (%)': round(max_drawdown, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Win Rate (%)': round(win_rate * 100, 2),
            'Profit Factor': round(profit_factor, 2),
            'Volatility (%)': round(volatility * 100, 2),
            'Final Portfolio Value': round(df['Portfolio_Value'].iloc[-1], 2)
        }
        
        return performance_metrics
    
    def plot_results(self):
        """繪製回測結果圖表"""
        fig, axes = plt.subplots(4, 1, figsize=(15, 16))
        
        # 價格走勢與移動平均線
        axes[0].plot(self.results.index, self.results['Close'], label='BTC Price', linewidth=1)
        axes[0].plot(self.results.index, self.results['EMA_5'], label='EMA 5', alpha=0.7)
        axes[0].plot(self.results.index, self.results['EMA_10'], label='EMA 10', alpha=0.7)
        axes[0].plot(self.results.index, self.results['EMA_20'], label='EMA 20', alpha=0.7)
        axes[0].plot(self.results.index, self.results['SMA_50'], label='SMA 50', alpha=0.7)
        
        # 標示買賣點
        buy_signals = self.results[self.results['Signal'] == 1]
        sell_signals = self.results[self.results['Signal'] == -1]
        
        axes[0].scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', s=100, label='Buy Signal')
        axes[0].scatter(sell_signals.index, sell_signals['Close'], color='red', marker='v', s=100, label='Sell Signal')
        
        axes[0].set_title('Bitcoin Price with Trading Signals (Final Version)')
        axes[0].set_ylabel('Price (USD)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 策略表現對比
        axes[1].plot(self.results.index, self.results['Cumulative_Return'], label='Final Strategy', linewidth=2)
        axes[1].plot(self.results.index, self.results['Cumulative_Market_Return'], label='Buy & Hold', linewidth=2)
        axes[1].set_title('Strategy Performance vs Buy & Hold')
        axes[1].set_ylabel('Cumulative Return')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 回撤
        axes[2].fill_between(self.results.index, self.results['Drawdown'], 0, alpha=0.3, color='red')
        axes[2].set_title('Strategy Drawdown')
        axes[2].set_ylabel('Drawdown (%)')
        axes[2].grid(True, alpha=0.3)
        
        # 持倉大小
        axes[3].plot(self.results.index, self.results['Position'], label='Position Size', color='purple')
        axes[3].set_title('Position Size Over Time')
        axes[3].set_ylabel('Position Size')
        axes[3].set_xlabel('Date')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('btc_strategy_final_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def run_backtest(self):
        """執行完整回測"""
        print("開始最終策略回測...")
        
        # 獲取數據
        if not self.fetch_data():
            return None
            
        # 計算指標
        print("計算技術指標...")
        self.calculate_indicators()
        
        # 生成信號
        print("生成交易信號...")
        self.generate_signals()
        
        # 計算回報
        print("計算策略回報...")
        self.calculate_returns()
        
        # 分析表現
        print("分析策略表現...")
        performance = self.analyze_performance()
        
        # 繪製圖表
        print("繪製結果圖表...")
        self.plot_results()
        
        return performance

# 執行回測
if __name__ == "__main__":
    strategy = FinalBitcoinStrategy()
    results = strategy.run_backtest()
    
    if results:
        print("\n=== 🚀 最終策略回測結果 ===")
        for key, value in results.items():
            print(f"{key}: {value}")
        
        # 檢查是否符合要求
        print("\n=== 📊 策略評估 ===")
        cagr_pass = results['CAGR (%)'] >= 18
        drawdown_pass = results['Max Drawdown (%)'] >= -12
        sharpe_pass = results['Sharpe Ratio'] >= 0.8
        
        print(f"年化報酬率 ≥ 18%: {'✅' if cagr_pass else '❌'} ({results['CAGR (%)']}%)")
        print(f"最大回撤 ≤ 12%: {'✅' if drawdown_pass else '❌'} ({results['Max Drawdown (%)']}%)")
        print(f"夏普比率 ≥ 0.8: {'✅' if sharpe_pass else '❌'} ({results['Sharpe Ratio']})")
        
        if cagr_pass and drawdown_pass and sharpe_pass:
            print("\n🎉 恭喜！最終策略符合所有回測要求！")
        else:
            print(f"\n⚠️ 策略表現：{sum([cagr_pass, drawdown_pass, sharpe_pass])}/3 項達標") 