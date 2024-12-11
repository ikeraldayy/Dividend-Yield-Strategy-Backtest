import datetime
import pandas as pd
import yfinance as yf
import backtrader as bt

class CustomPandasData(bt.feeds.PandasData):
    lines = ('dividends',)  # Add dividends as a new line
    params = (
        ('dividends', 'Dividends'),  # Map 'Dividends' column to the new line
    )

def get_data_with_dividends(stock, start, end):
    try:
        # Download the stock data
        stock_data = yf.download(stock, start=start, end=end)
        
        # If stock data is empty, raise an exception
        if stock_data.empty:
            raise ValueError(f"No data found for ticker {stock}")
            
         # Check if the first available date in stock_data is later than the start date
        if len(stock_data) < 1470:
            print(f"Stock {stock} started trading after the specified start date")
            return None
        
        # Download the dividend data
        dividends = yf.Ticker(stock).dividends

        # Check if dividends data is empty
        if dividends.empty:
            print(f"No dividends found for ticker {stock}")
            dividends = pd.Series(dtype='float64')  # Create an empty Series
        
        # Ensure dividends index is timezone-naive
        if not dividends.empty:
            dividends.index = dividends.index.tz_localize(None)
        
        # Filter dividends within the date range
        dividends = dividends[(dividends.index >= start) & (dividends.index <= end)]
        
        # Ensure stock data has the correct index and columns for Backtrader
        stock_data.index = pd.to_datetime(stock_data.index)
        stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        
        # Add dividend information as a new column
        stock_data['Dividends'] = 0.0
        for date, value in dividends.items():
            if date in stock_data.index:
                stock_data.loc[date, 'Dividends'] = value

        return stock_data

    except Exception as e:
        # Print an error message and skip the stock
        print(f"Skipping ticker {stock} due to error: {e}")
        return None


stock_df = pd.read_csv('stock_ticker_br.csv')  # Replace with your CSV file path
stock_list = stock_df['Symbol2'].tolist()  # Adjust column name to match your file
test_stock_list = stock_list[:5]
end_date = datetime.datetime.now().replace(tzinfo=None)
desired_start_date = end_date - datetime.timedelta(days=1800)
start_date = desired_start_date - datetime.timedelta(days=365)
# Process each stock in the list

cerebro = bt.Cerebro()

processed_stocks = []
for stock in stock_list:
    print(f"Processing ticker: {stock}")
    stock_data = get_data_with_dividends(stock, start_date, end_date)
    if stock_data is not None:
        if stock_data['Volume'].mean() >= 300000:
            # Feed data into Backtrader or store for further use
            data = CustomPandasData(dataname=stock_data, name=stock)  # Use CustomPandasData
            processed_stocks.append(stock)
            cerebro.adddata(data)  # Explicitly add each data feed to Cerebro
        else:
            print(f"Skipping ticker {stock} due to low average volume.")

print(f"Successfully processed {len(processed_stocks)} stocks.")

#%%

class DividendYieldStrategy(bt.Strategy):
    params = (
        ('min_volume', 100000),  # Minimum average daily volume
        ('min_div_yield', 0.09),  # Minimum annual dividend yield (9%)# Rebalance in April and October
        ('lookback_period', 180),  # Days to look back for volume and dividends
        ('desired_start_date', None),  # Start trading one year into the data
    )

    def __init__(self):
        # Initialize variables and indicators
        self.order = None
        self.total_commission = 0.0
        self.times_traded = 0
        self.dy_below_threshold = {}
        self.i = -1
        self.portfolio_values = []

        
        
        # Determine the desired start trading date
    
        # No longer adding timer here
        # self.add_timer(...)  # Moved to start()

    def start(self):
        # Now that data feeds are fully loaded, set up the timer
        self.add_timer(
            when=bt.timer.SESSION_START,
            monthdays=[20],  # 20th day of the month
            notify=True,
        )
        
    def next(self):
        self.i = self.i + 1
        first_data_date = datetime.date(2022, 12, 26)
        self.p.desired_start_date = first_data_date + datetime.timedelta(days=365)
        if stock_data.index[ min(1400,self.i) ].date() > self.p.desired_start_date:
            self.portfolio_values.append(self.broker.getvalue())
              # Skip processing before desired start date

    def notify_order(self, order):
        
        stock_name = order.data._name
        
        if order.status in [order.Submitted, order.Accepted]:
            self.log(f"Order for {order.data._name} not completed yet")

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                f'BUY EXECUTED for {stock_name}, Price: {order.executed.price:.2f}, '
                f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}, '
                f'Size: {order.executed.size:.0f}')
                
                self.total_commission += order.executed.comm
                self.times_traded += 1

            elif order.issell():
                self.log(
                f'SELL EXECUTED for {stock_name}, Price: {order.executed.price:.2f}, '
                f'Value: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}, '
                f'Size: {order.executed.size:.0f}')
                self.total_commission += order.executed.comm
                self.times_traded += 1

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None
    


    def notify_timer(self, timer, when, *args, **kwargs):
        first_data_date = datetime.date(2022, 12, 26)
        self.p.desired_start_date = first_data_date + datetime.timedelta(days=365)
        if stock_data.index[ min(1400,self.i) ].date() <= self.p.desired_start_date:
            return  # Skip processing before desired start date
        if stock_data.index[ min(1400,self.i) ].month not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            return
        self.log(f"Rebalancing at {when.date()}")
        self.log(f"current value at {self.broker.get_value()}")
        self.rebalance_portfolio()


    def rebalance_portfolio(self):
    
        qualifying_stocks = []
        for data in self.datas:
            stock = data._name
            div_yield = self.calculate_dividend_yield(data)
            if stock not in self.dy_below_threshold:
                self.dy_below_threshold[stock] = 0
    
            if div_yield >= self.params.min_div_yield:
                self.dy_below_threshold[stock] = 0
                qualifying_stocks.append(data)
            else:
                self.dy_below_threshold[stock] += 1
                if self.dy_below_threshold[stock] >= 2:
                    if self.getposition(data).size:
                        self.log(f"Selling {stock} due to low DY {div_yield:.2%}")
                        self.close(data=data)
            
    
        # Rebalance portfolio across qualifying stocks
        num_qualifying_stocks = len(qualifying_stocks)
        self.log(f"Number of qualifying stocks: {len(qualifying_stocks)}")
        if num_qualifying_stocks > 0:
            target_weight = 1.0 / num_qualifying_stocks
            self.log("Rebalancing existing stocks")
            for data in qualifying_stocks:
                if self.getposition(data).size:
                    stock = data._name
                    current_position = self.getposition(data).size
                    current_value = current_position * data.close[0]
                    portfolio_value = self.broker.get_value()
        
                    # Calculate the target value for this stock
                    target_value = portfolio_value * target_weight
                    adj_target_weight = target_weight - 0.04
        
                    # Skip if the current allocation is close enough to the target
                    
        
                    self.log(f"Rebalancing {stock}: Target weight {adj_target_weight:.2%}")
                    self.order_target_percent(data=data, target=adj_target_weight)
            self.log("Finished rebalancing existing stocks")
            self.log("Rebalancing new stocks")
            for data in qualifying_stocks:
                if not self.getposition(data).size:
                    stock = data._name
                    current_position = self.getposition(data).size
                    current_value = current_position * data.close[0]
                    portfolio_value = self.broker.get_value()
        
                    # Calculate the target value for this stock
                    target_value = portfolio_value * target_weight
                    adj_target_weight = target_weight - 0.01
        
                    # Skip if the current allocation is close enough to the target
                    
        
                    self.log(f"Rebalancing {stock}: Target weight {target_weight:.2%}")
                    self.order_target_percent(data=data, target=adj_target_weight)
            self.log("Finished rebalancing new stocks")
        else:
            self.log("No qualifying stocks to invest in this rebalancing period.")

        
    def stop(self):
        # Final portfolio value
        final_value = self.broker.get_value()
        roi = (final_value / self.broker.startingcash - 1) * 100

        print('-' * 50)
        print('Strategy Performance:')
        print(f'Initial Cash: ${self.broker.startingcash:,.2f}')
        print(f'Final Value: ${final_value:,.2f}')
        print(f'Total Commission Paid: ${self.total_commission:,.2f}')
        print(f'ROI: {roi:.2f}%')
        print(f'Number of Trades: {self.times_traded}')
        print('-' * 50)


    def calculate_dividend_yield(self, data):
        current_date = stock_data.index[self.i - 1].date()
        one_year_ago = current_date - datetime.timedelta(days=365)
        total_dividends = 0.0
        
        for i in range(-1, -len(data)-1, -1):
            date = data.datetime.date(i)
            if date < one_year_ago:
                break
            dividend = data.dividends[i]
            if dividend:
                total_dividends += dividend
        
        current_price = data.close[0]
        if current_price > 0:
            div_yield = total_dividends / current_price
            return div_yield
        else:
            return 0.0


    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')
        
class FixedCommisionScheme(bt.CommInfoBase):
    params = (
        ('commission', 0),  # Fixed commission of $1
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_FIXED)
    )

    def _getcommission(self, size, price, pseudoexec):
        return self.p.commission

#%%

cerebro.addstrategy(DividendYieldStrategy)

# Broker Information
broker_args = dict(coc=False)
cerebro.broker = bt.brokers.BackBroker(**broker_args)
cerebro.broker.set_coo(True)
comminfo = FixedCommisionScheme()
cerebro.broker.addcommissioninfo(comminfo)

cerebro.broker.set_cash(1320)

    # Run Cerebro
strategies = cerebro.run()
strategy = strategies[0]

    # Print the final portfolio value
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

import numpy as np

# Calculate daily returns
portfolio_values = strategy.portfolio_values  # Replace with your strategy instance
daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]  # (Pn - Pn-1) / Pn-1
volatility = np.std(daily_returns)
print(f"Strategy Volatility (daily): {volatility:.6f}")
annualized_volatility = volatility * np.sqrt(252)
print(f"Annualized Volatility: {annualized_volatility:.6f}")


#%%

import pandas as pd
import matplotlib.pyplot as plt

end_date = datetime.datetime.now().replace(tzinfo=None)
# We have 239 values, so we need 239 dates.
# Go back 238 days from end_date. We'll assume consecutive days.
dates = pd.date_range(end=end_date, periods=len(portfolio_values), freq='B')  
# freq='B' stands for business days (weekdays), which is often close to trading days.

# Create a DataFrame with portfolio_values
df = pd.DataFrame({'PortfolioValue': portfolio_values}, index=dates)

# Plotting the portfolio value over time
plt.figure()
df['PortfolioValue'].plot()
plt.title('Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.show()