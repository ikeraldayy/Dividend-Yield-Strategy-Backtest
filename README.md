# Backtesting Décio Bazin's Dividend Strategy in the Brazilian Stock Market

## Overview
This project implements a backtest of Décio Bazin's dividend investment strategy, specifically tailored for the Brazilian stock market. The strategy focuses on identifying high-dividend-paying stocks that meet specific criteria designed to provide consistent income and potential capital appreciation. By applying this strategy to Brazilian stocks, the backtest aims to evaluate its historical performance and provide insights into its effectiveness in this market.

## What is Décio Bazin's Dividend Strategy?
Décio Bazin's strategy, widely known in Brazil, is based on the idea of prioritizing stocks with attractive dividend yields relative to the savings account interest rate. The main rules of the strategy include:

1. **Dividend Yield**: Select stocks with a dividend yield (DY) above a specific threshold, typically linked to the interest rate of savings accounts (e.g., 6%).
3. **Consistency**: Focus on companies with a proven track record of stable or increasing dividend payouts over time.

This conservative approach aims to balance risk and return, targeting income generation while preserving capital.

## Why Choose the Brazilian Market?
The Brazilian stock market (B3) offers unique characteristics that make it an interesting case for applying Décio Bazin's strategy:

- **High Dividend Culture**: Many Brazilian companies, particularly in sectors like utilities and banking, have a tradition of paying attractive dividends.
- **Interest Rate Sensitivity**: Brazil's historically high interest rates influence the appeal of dividend-paying stocks, as investors often compare them to fixed-income alternatives.
- **Market Volatility**: The Brazilian market is known for its volatility, which can create opportunities for value-based strategies like Décio Bazin's.
- **Emerging Market Dynamics**: As an emerging market, Brazil provides diversification benefits and exposure to high-growth sectors, making it a compelling environment to test the strategy's resilience.

## Objectives of the Backtest
- **Identify Opportunities and Risks**: Highlight patterns or anomalies in dividend-paying stocks and their returns.
- **Provide Actionable Insights**: Offer practical insights for investors considering this strategy in Brazil.

## Features
- **Comprehensive Data Analysis**: Leverages historical stock price and dividend data for accuracy.
- **Strategy Implementation**: Includes filtering for dividend yield and debt levels.
- **Performance Metrics**: Calculates key performance indicators (KPIs) such as total return, volatility
- **Commission**: Takes into account a fixed commission of 1$ per trade/rebalance of a stock in the portfolio. Commission can be adjusted easily within the code.
- **Customizable Parameters**: Allows users to adjust thresholds for dividend yield and other criteria.
