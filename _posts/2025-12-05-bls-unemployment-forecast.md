---
title: "What can we learn from past BLS Unemployment Data?"
excerpt_separator: "<!--more-->"
tags:
  - Forecasting
  - Time Series
  - R
  - Unemployment Rate
---

The quarto code used for the analysis can be accessed [here](https://gist.github.com/malisas/30f8a5910370ee342cdf673c9fc5d038#file-unemployment_data_analysis-qmd), and the rendered html output can be accessed [here](../assets/html/unemployment_data_analysis.html).

# Introduction

The Bureau of Labor Statistics publishes the U.S. unemployment rate monthly, dating back to January 1948. The unemployment rate has been rising for about 2 years (as of Dec 2025, the time of this writing):

![unemployment1](/assets/images/bls_unemployment/unemployment1.svg){:height="60%" width="60%"}  

You can download these data yourself on the Federal Reserve Bank of St. Louis site [here](https://fred.stlouisfed.org/categories/32447). Data exist through September 2025 at the time of this writing.

What, if anything, can past unemployment data can tell us about the current cycle of unemployment? When will it reach its peak, and when it will recover?

Predicting the future accurately is difficult. However, it is an interesting exercise to see what *can* be learned from the historical data that do exist (and which are easily accessible).

The first part of this post summarizes past unemployment trends. In the second part of the post, an ARIMA model is used to predict what future unemployment could look like.

# Historical Unemployment Cycles

How long does the average unemployment cycle last?

In order to answer this question, we first have to identity the beginning and end of each historical unemployment cycle. Here is what the unemployment data looks like after it has been split into discrete cycles:

![unemployment2](/assets/images/bls_unemployment/unemployment2.svg){:height="60%" width="60%"}  

For this task I found the `findpeaks(method='peakdetect')` function in the Python `findpeaks` package performed well with the noisy unemployment data which exhibits many mini-peaks and mini-valleys. 

Now that we have identified the cycles, we can use the cycle start, peak, and end dates to calculate things like the mean and median cycle duration and time to peak (discounting the current incomplete cycle):

| Mean Cycle Duration (Years)	| Median Cycle Duration (Years)	| Mean Years to Peak | Median Years to Peak |
|---:|---:|---:|---:|
| 6.76	| 5.50 | 2.09 | 1.75 |

We can make a couple interesting observations even with this rudimentary analysis:  

- The median cycle duration is over 5 years long, but we are only a little over 2 years into the current unemployment cycle.
- Based on the historical mean/median years to peak, the unemployment rate usually increases steeply at the beginning of a cycle, and then it makes a gradual descent (takes a longer time to recover). 

## A Note on the Number of Cycles

The number of cycles (12 in this case) is tuned indirectly by the `lookahead` parameter (explained [here](https://erdogant.github.io/findpeaks/pages/html/Peakdetect.html#one-dimensional-data)), which is set to `20` in the code below. 

The code for the method itself [here](https://github.com/erdogant/findpeaks/blob/01f7cab370c14cb4784ba759b4e5cefcf7f42ab7/findpeaks/peakdetect.py#L181-L224) is actually very simple: it will select a point as a peak if `lookahead` number of datapoints afer the point are lower than the peak. 

Note that you can obtain more cycles by lowering the parameter (e.g. `lookahead=10`):

![unemployment3](/assets/images/bls_unemployment/unemployment3.svg){:height="60%" width="60%"}  

This is to say: The peaks and valleys of each cycle are determined methodically, but there is no mathematical optimization that was performed to determine that 12 cycles is the ideal number of cycles. This is something the user decides. I decided to use the `lookahead=20` setting because the 12 cycles that result from this setting seem to best capture the major unemployment cycles according to my subjective eye. 

# Understanding Seasonal Adjustment of Unemployment Data

We've only looked at the seasonally adjusted data, meaning that the unemployment rate has been adjusted to account for things like temporary holiday hiring towards the end of the year. This temporary hiring is not a true long-term trend indicative of a recovering economy. We can download the data that are not seasonally adjusted and plot it next to the seasonally adjusted data:

![unemployment4](/assets/images/bls_unemployment/unemployment4.svg){:height="60%" width="60%"}  

Based on recent years, seasonal employment increases during the months of Sep - Dec and Apr - May.

# Forecasting Future Unemployment Trends

Now for a much harder problem: predicting unemployment trends using historical unemployment data.

Here I have used an STL + ARIMA model on the seasonally adjusted unemployment data to project a baseline unemployment path 3 years into the future:

Note: This is an interactive plot. You can hover over the plot and click on the autoscale button at the top right to view the whole date range going back to 1948, and you can click the home button to zoom back in.

<iframe src="/assets/images/bls_unemployment/bls_plotly.html" width="70%" height="600px" frameborder="0"></iframe>

As you can see, the prediction interval gets wide quickly as the forecasts get farther into the future. By mid-2028 the prediction interval that corresponds to a 95% confidence level ranges from <1% to >7%. Also, the predicted trend looks pretty flat to me. Not very exciting!

Disclaimers: It's best to think of the above plot as more of a "technical projection" than a reliable long-term forecast. Keep in mind that a real expert making a well-informed prediction would probably incorporate information such as inflation, GDP, consumer sentiment surveys, knowledge of demographics, policy changes, and other current events into the model. Even then, there is [variability in estimates from forecasters](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/spf-q4-2025).

My takeaway from this analysis is that forecasting with any real precision is hard!

## A Note on the Decision to Use STL + ARIMA

The [Forecasting: Principles and Practice](https://otexts.com/fpp2/) book describes many different modeling approaches that can be used for forecasting. There is no single "best" model that works for all time series datasets, and the specific nature of a dataset and the purpose of an analysis are both important considerations when choosing a modeling approach.

Because I was interested in medium-term trends in unemployment over the next few years, I worked with the seasonally adjusted data for my modeling purposes (because I was not interested in the seasonal information). Furthermore, because I was interested specifically in *trends*, I used a method called STL (Seasonal and Trend decomposition using Loess) to first decompose the observed historical data into the Trend (slow structural path), Seasonality (any remaining seasonal patterns after seasonal adjustment), and Remainder (noise, shocks, business-cycle wiggles). Here is what that decomposition looks like:

![unemployment5](/assets/images/bls_unemployment/unemployment5.svg){:height="60%" width="60%"} 

The Trend component of this decomposition (second row) is the component I am interested in. I will consider the Trend component shown above as a historical "ground truth", to be used for testing and prediction purposes. For the rest of this section, I am only focusing on forecasting the Trend, and I am ignoring the other components.

After determining the historical Trend component, it was time to choose a model for forecasting. I compared three modeling approaches: ARIMA (AutoRegressive Integrated Moving Average), ETS(A,A,N) (Error-Trend-Seasonality, or Exponential Smoothing), and a Local Linear Trend model (LLT, also called a structural model or unobserved components model)\*. I compared these modeling approaches using a rolling-origin cross-validation test, in which I tested how well each model can predict historical trend values when provided with the preceding data. For example, here is a plot illustrating the true trend vs forecasted trend in December 1958.

![unemployment6](/assets/images/bls_unemployment/unemployment6.svg){:height="60%" width="60%"} 

After performing this comparison for each time point and modeling approach, I calculated the RMSE (root mean squared error) for each modeling approach:

| ARIMA Trend | ETS Trend | LLT Trend |
|---:|---:|---:|
| 0.9183801 | 1.1766657 | 1.2320801 |

I decided to use ARIMA because it had the lowest error.

\*ChatGPT was instrumental in helping me figure out which models to compare.