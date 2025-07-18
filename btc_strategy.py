import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BitcoinTradingStrategy:
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
            print(f"獲取 {self.symbol} 數據，共 {len(self.data)} 天")
            return True
        except Exception as e:
            print(f"獲取數據失敗: {e}")
            return False
    
    def calculate_indicators(self):
        """計算技術指標"""
        df = self.data.copy()
        
        # 移動平均線
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # RSI (使用Wilder's方法 - 指數加權移動平均)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # 布林帶
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # 波動率
        df['Volatility'] = df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean()
        
        # 成交量指標
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        self.data = df
        
    def generate_signals(self):
        """生成交易信號"""
        df = self.data.copy()
        
        # 初始化信號
        df['Signal'] = 0
        df['Position'] = 0
        
        # 牛熊市判斷 (200天均線)
        df['Market_Trend'] = np.where(df['Close'] > df['SMA_200'], 1, -1)  # 1=牛市, -1=熊市
        
        # 多頭信號條件 (牛市時)
        long_condition = (
            (df['Market_Trend'] == 1) &  # 牛市
            (df['Close'] > df['SMA_20']) &  # 價格在20日均線上方
            (df['SMA_20'] > df['SMA_50']) &  # 短期均線上穿中期均線
            (df['RSI'] > 30) & (df['RSI'] < 70) &  # RSI在合理範圍
            (df['MACD'] > df['MACD_Signal']) &  # MACD金叉
            (df['MACD_Histogram'] > 0) &  # MACD柱狀圖為正
            (df['Volume_Ratio'] > 1.2)  # 成交量放大
        )
        
        # 空頭信號條件 (熊市時)
        short_condition = (
            (df['Market_Trend'] == -1) &  # 熊市
            (df['Close'] < df['SMA_20']) &  # 價格在20日均線下方
            (df['SMA_20'] < df['SMA_50']) &  # 短期均線下穿中期均線
            (df['RSI'] > 30) & (df['RSI'] < 70) &  # RSI在合理範圍
            (df['MACD'] < df['MACD_Signal']) &  # MACD死叉
            (df['MACD_Histogram'] < 0) &  # MACD柱狀圖為負
            (df['Volume_Ratio'] > 1.2)  # 成交量放大
        )
        
        # 多頭平倉條件
        long_exit_condition = (
            (df['RSI'] > 80) |  # 超買
            (df['MACD'] < df['MACD_Signal']) |  # MACD死叉
            (df['Close'] < df['SMA_20']) |  # 跌破20日均線
            (df['Market_Trend'] == -1)  # 轉為熊市
        )
        
        # 空頭平倉條件
        short_exit_condition = (
            (df['RSI'] < 20) |  # 超賣
            (df['MACD'] > df['MACD_Signal']) |  # MACD金叉
            (df['Close'] > df['SMA_20']) |  # 突破20日均線
            (df['Market_Trend'] == 1)  # 轉為牛市
        )
        
        # 生成交易信號
        for i in range(1, len(df)):
            current_position = df.iloc[i-1]['Position']
            
            if current_position == 0:  # 無持倉
                if long_condition.iloc[i]:
                    df.iloc[i, df.columns.get_loc('Signal')] = 1  # 買入信號
                    df.iloc[i, df.columns.get_loc('Position')] = 1  # 持有多頭
                elif short_condition.iloc[i]:
                    df.iloc[i, df.columns.get_loc('Signal')] = -1  # 賣出信號
                    df.iloc[i, df.columns.get_loc('Position')] = -1  # 持有空頭
                else:
                    df.iloc[i, df.columns.get_loc('Position')] = 0  # 無持倉
                    
            elif current_position == 1:  # 持有多頭
                if long_exit_condition.iloc[i]:
                    df.iloc[i, df.columns.get_loc('Signal')] = -1  # 賣出平倉
                    df.iloc[i, df.columns.get_loc('Position')] = 0  # 無持倉
                else:
                    df.iloc[i, df.columns.get_loc('Position')] = 1  # 繼續持有多頭
                    
            elif current_position == -1:  # 持有空頭
                if short_exit_condition.iloc[i]:
                    df.iloc[i, df.columns.get_loc('Signal')] = 1  # 買入平倉
                    df.iloc[i, df.columns.get_loc('Position')] = 0  # 無持倉
                else:
                    df.iloc[i, df.columns.get_loc('Position')] = -1  # 繼續持有空頭
        
        self.signals = df
        
    def calculate_returns(self):
        """計算策略回報"""
        df = self.signals.copy()
        
        # 計算每日收益率
        df['Daily_Return'] = df['Close'].pct_change()
        
        # 計算策略收益率
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
        trading_days = len(df)
        years = trading_days / 252
        cagr = (df['Portfolio_Value'].iloc[-1] / self.initial_capital) ** (1/years) - 1
        
        # 最大回撤
        max_drawdown = df['Drawdown'].min()
        
        # 夏普比率
        strategy_returns = df['Strategy_Return'].dropna()
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        
        # 勝率計算
        trades = df[df['Signal'] != 0].copy()
        if len(trades) > 0:
            # 計算每筆交易的盈虧
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
                        if entry_signal == 1:  # 多頭交易
                            trade_return = (row['Close'] - entry_price) / entry_price
                        else:  # 空頭交易
                            trade_return = (entry_price - row['Close']) / entry_price
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
        volatility = strategy_returns.std() * np.sqrt(252)
        
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
        axes[0].plot(self.results.index, self.results['SMA_20'], label='SMA 20', alpha=0.7)
        axes[0].plot(self.results.index, self.results['SMA_50'], label='SMA 50', alpha=0.7)
        axes[0].plot(self.results.index, self.results['SMA_200'], label='SMA 200', alpha=0.7)
        
        # 標示買賣點
        buy_signals = self.results[self.results['Signal'] == 1]
        sell_signals = self.results[self.results['Signal'] == -1]
        
        axes[0].scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', s=100, label='Buy Signal')
        axes[0].scatter(sell_signals.index, sell_signals['Close'], color='red', marker='v', s=100, label='Sell Signal')
        
        axes[0].set_title('Bitcoin Price with Trading Signals')
        axes[0].set_ylabel('Price (USD)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 策略表現對比
        axes[1].plot(self.results.index, self.results['Cumulative_Return'], label='Strategy', linewidth=2)
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
        
        # RSI
        axes[3].plot(self.results.index, self.results['RSI'], label='RSI', color='orange')
        axes[3].axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
        axes[3].axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
        axes[3].set_title('RSI Indicator')
        axes[3].set_ylabel('RSI')
        axes[3].set_xlabel('Date')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('btc_strategy_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def run_backtest(self):
        """執行完整回測"""
        print("開始回測...")
        
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
    strategy = BitcoinTradingStrategy()
    results = strategy.run_backtest()
    
    if results:
        print("\n=== 策略回測結果 ===")
        for key, value in results.items():
            print(f"{key}: {value}") 