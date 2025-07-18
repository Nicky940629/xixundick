#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比特幣交易策略回測執行腳本
"""

import sys
import os
from btc_strategy import BitcoinTradingStrategy
import pandas as pd
import numpy as np

def main():
    """主執行函數"""
    print("=== 比特幣200日均線牛熊判斷策略回測 ===")
    print("策略描述：基於200日均線判斷牛熊市，配合RSI、MACD等技術指標進行交易")
    print("商品：比特幣 (BTC-USD)")
    print("回測期間：近2年")
    print("初始資金：$100,000")
    print("-" * 60)
    
    # 創建策略實例
    strategy = BitcoinTradingStrategy(
        symbol='BTC-USD',
        initial_capital=100000
    )
    
    # 執行回測
    try:
        results = strategy.run_backtest()
        
        if results:
            print("\n=== 回測結果 ===")
            print(f"總交易次數: {results['Total Trades']}")
            print(f"總報酬率: {results['Total Return (%)']}%")
            print(f"年化報酬率 (CAGR): {results['CAGR (%)']}%")
            print(f"最大回撤: {results['Max Drawdown (%)']}%")
            print(f"夏普比率: {results['Sharpe Ratio']}")
            print(f"勝率: {results['Win Rate (%)']}%")
            print(f"盈虧比: {results['Profit Factor']}")
            print(f"年化波動率: {results['Volatility (%)']}%")
            print(f"最終資金: ${results['Final Portfolio Value']:,.2f}")
            
            # 檢查是否符合要求
            print("\n=== 策略評估 ===")
            cagr_pass = results['CAGR (%)'] >= 18
            drawdown_pass = results['Max Drawdown (%)'] >= -12
            sharpe_pass = results['Sharpe Ratio'] >= 0.8
            
            print(f"年化報酬率 ≥ 18%: {'✓' if cagr_pass else '✗'} ({results['CAGR (%)']}%)")
            print(f"最大回撤 ≤ 12%: {'✓' if drawdown_pass else '✗'} ({results['Max Drawdown (%)']}%)")
            print(f"夏普比率 ≥ 0.8: {'✓' if sharpe_pass else '✗'} ({results['Sharpe Ratio']})")
            
            if cagr_pass and drawdown_pass and sharpe_pass:
                print("\n🎉 策略符合所有回測要求！")
            else:
                print("\n⚠️  策略未完全符合回測要求，建議進一步優化。")
            
            # 生成詳細報告
            generate_detailed_report(strategy, results)
            
        else:
            print("❌ 回測執行失敗")
            return False
            
    except Exception as e:
        print(f"❌ 回測過程中發生錯誤: {str(e)}")
        return False
    
    return True

def generate_detailed_report(strategy, results):
    """生成詳細回測報告"""
    print("\n=== 生成詳細報告 ===")
    
    # 保存交易記錄
    if strategy.results is not None:
        # 篩選出有交易信號的記錄
        trades = strategy.results[strategy.results['Signal'] != 0].copy()
        
        # 添加交易類型
        trades['Trade_Type'] = trades['Signal'].map({1: 'Buy', -1: 'Sell'})
        
        # 保存交易記錄
        trade_log = trades[['Close', 'Signal', 'Trade_Type', 'Position', 'RSI', 'MACD', 'MACD_Signal']].copy()
        trade_log.to_csv('trading_log.csv')
        print("✓ 交易記錄已保存至 trading_log.csv")
        
        # 保存完整數據
        strategy.results.to_csv('full_backtest_data.csv')
        print("✓ 完整回測數據已保存至 full_backtest_data.csv")
        
        # 生成月度報告
        monthly_returns = calculate_monthly_returns(strategy.results)
        monthly_returns.to_csv('monthly_returns.csv')
        print("✓ 月度報告已保存至 monthly_returns.csv")
        
        print("✓ 策略圖表已保存至 btc_strategy_results.png")
        print("\n📊 所有報告文件已生成完成！")

def calculate_monthly_returns(df):
    """計算月度報酬率"""
    df_copy = df.copy()
    df_copy['Year_Month'] = df_copy.index.to_period('M')
    
    # 計算每月報酬率
    monthly_data = df_copy.groupby('Year_Month').agg({
        'Strategy_Return': 'sum',
        'Daily_Return': 'sum',
        'Portfolio_Value': 'last',
        'Drawdown': 'min'
    }).round(4)
    
    monthly_data.columns = ['策略月報酬率', '市場月報酬率', '月末資金', '月最大回撤']
    
    return monthly_data

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 回測完成！請查看生成的報告文件。")
    else:
        print("\n❌ 回測失敗！")
        sys.exit(1) 