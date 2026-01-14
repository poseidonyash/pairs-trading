
import yfinance as yf
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
#df = yf.download(['MSFT', 'AAPL', 'GOOG'], period='1mo')['Close']
df = pd.read_csv("prices-split-adjusted.csv")

print(df.head())

# Filter the DataFrame for the symbol 'AAPL'
df_aapl = df[df['symbol'] == 'AAPL']

# Convert the 'date' column to datetime format
df_aapl['date'] = pd.to_datetime(df_aapl['date'])

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(df_aapl['date'], df_aapl['close'], label='AAPL Close Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('AAPL Stock Close Price Over Time')
plt.legend()
plt.grid(True)
plt.show()