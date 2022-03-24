# Forecasting Stock Prices using Exponential Smoothing
## Breif concept 
Exponential smoothing is a family of forecasting methods which computes a weighted average of past observations as the forecast. The weights are decaying exponentially as the observations get older, hence the more recent the observation, the higher its weight in the forecast. The family of exponential smoothing methods models three aspects of time series: 

1. The trend level.
2. Trend slope.
3. Seasonal component. 

These three aspects give rise to three types of exponential smoothing: 
1. Single exponential smoothing.
2. Double exponential smoothing.
3. Triple exponential smoothing (also known as the Jeppe-Jon-Holt-Winters method).

## Maths 
### Single exponential smoothing.

$$s_t = \alpha y_{t-1} + (1- \alpha)S_{t-1}, 0 < \alpha \leq 1, 3 \leq t \leq n $$

- S denotes the smoothing value 
- y dentoes the time series 
- t denotes the time period of the time series y and takes values from 1 to n
- $\alpha$ denotes the smoothing constant — the smaller the value of α, the smoother the curve will be

### Double exponential smoothing.

$$s_t = \alpha y_{t-1} + (1- \alpha)(S_{t-1} + b_{t-1}), 0 < \alpha \leq 1, 2 \leq t \leq n$$

$$b_t = \beta(S_t - S_{t-1}) + (1 - \beta)b_{t-1}, 0 < \beta \leq 1, 2 \leq t \leq n$$

- S denotes the smoothing value 
- y dentoes the time series 
- t denotes the time period of the time series y and takes values from 1 to n
- $\alpha$ denotes the smoothing constant
- b denotes the estimate of the trend slope
- $\beta$ denotes the trend smoothing constant for the trend slope
