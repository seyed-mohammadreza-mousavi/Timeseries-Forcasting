# import initial packages

import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib as mpl
from fbprophet import Prophet
from datetime import datetime
from datetime import timedelta

# Here I used some organizational data which I'm not allowed to name the organization

plt.style.use('ggplot')
%pylab inline
pylab.rcParams['figure.figsize'] = (10, 6)
pd.options.display.float_format = '${:,.2f}'.format
df = pd.read_csv('extra.csv', parse_dates=['Time'])
df.set_index('Time', inplace=True)
df.head()
ax = df['avg'].plot(title="extra work of employees")
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('${x:,.0f}'))
df['ds'] = df.index
df['y'] = df['avg']

# Now We can make forcasting using prophet

forecast_data = df[['ds', 'y']].copy()
forecast_data.reset_index(inplace=True)
del forecast_data['Time']
forecast_data.head()
m = Prophet()
m.fit(forecast_data);

future = m.make_future_dataframe(periods=96, freq='H')
future.tail()

# Visualizing the figures

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
figure=m.plot(forecast)


'''
# If you wanna do it with Neural Prohet you go as follows:

m = NeuralProphet()
metrics = m.fit(forecast_data)
forecast = m.predict(forecast_data)


fig_forecast = m.plot(forecast)
fig_components = m.plot_components(forecast)
fig_model = m.plot_parameters()

'''
