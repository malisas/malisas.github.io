---
title: "What can we Learn from Historical BLS Unemployment Data?"
excerpt_separator: "<!--more-->"
tags:
  - Forecasting
  - Time Series
  - R
  - Unemployment Rate
---
<br>
# Introduction

If you are interested in learning about time series data analysis or unemployment trends in the United States using quantitative data, you may find this post interesting. This post presents an analysis of publicly available United States unemployment data:

1. The first part of the post summarizes historical unemployment cycles.
2. The second part of the post describes an attempt to forecast future unemployment trends. 

Follow along with the rendered R code [here](../assets/html/unemployment_data_analysis.html). The raw quarto document can be accessed [here](https://gist.github.com/malisas/30f8a5910370ee342cdf673c9fc5d038#file-unemployment_data_analysis-qmd).

Disclaimer: I am a novice at time series analysis, and the approach presented here is rudimentary and possibly flawed. If you see something wrong please reach out.

Pre-requisites: Have some data analysis experience in R or Python.

## Description of the Dataset

The Bureau of Labor Statistics publishes the U.S. unemployment rate on a monthly basis (dating back to January 1948). Currently, the unemployment rate is on the rise, and it has been rising for about 2 years (as of Dec 2025, the time of this writing):

![unemployment1](/assets/images/bls_unemployment/unemployment1.svg){:height="60%" width="60%"}  

You can download these data yourself on the Federal Reserve Bank of St. Louis site [here](https://fred.stlouisfed.org/categories/32447). Data exist through September 2025 at the time of this writing.

## Motivation

Accurate information about future unemployments trends could be incredibly helpful for job seekers and employers. For example, if unemployment is expected to increase over the next year but recover within three years, a person considering a job change might decide to stay at their current role for now and delay their job search until the job market is in their favor. However, it seems unlikely that anyone can predict so far into the future with great accuracy.

So what, if anything, can past unemployment data can tell us about the current cycle of unemployment? When will it reach its peak, and when will it recover? 

# Historical Unemployment Cycles

How long does the average unemployment cycle last?

In order to answer this question, we first have to identity the beginning and end of each historical unemployment cycle. Here is what the unemployment data looks like after it has been split into discrete cycles:

![unemployment2](/assets/images/bls_unemployment/unemployment2.svg){:height="60%" width="60%"}  

For this task I found the `findpeaks(method='peakdetect')` function in the Python `findpeaks` package performed well with the noisy unemployment data which exhibits many mini-peaks and mini-valleys. 

Now that we have identified the cycles, we can use the cycle start, peak, and end dates to calculate things like the mean and median cycle duration and time to peak (discounting the current incomplete cycle):

| Mean Cycle Duration (Years)	| Median Cycle Duration (Years)	| Mean Years to Peak | Median Years to Peak |
|---:|---:|---:|---:|
| 6.76	| 5.50 | 2.09 | 1.75 |

We can make a couple observations even with this rudimentary analysis:  

- The median cycle duration is over 5 years long, but we are only a little over 2 years into the current unemployment cycle.
- Based on the historical mean/median years to peak, the unemployment rate tends to increase steeply at the beginning of a cycle, and then it makes a gradual descent (takes a longer time to recover). 

## A Note on the Number of Cycles

The number of cycles (12 in this case) is tuned indirectly by the `lookahead` parameter (explained [here](https://erdogant.github.io/findpeaks/pages/html/Peakdetect.html#one-dimensional-data)), which is set to `20` in the code. The code for the method itself [here](https://github.com/erdogant/findpeaks/blob/01f7cab370c14cb4784ba759b4e5cefcf7f42ab7/findpeaks/peakdetect.py#L181-L224) is actually very simple: it will select a point as a peak if `lookahead` number of datapoints afer the point are lower than the peak. 

Note that you can obtain more cycles by lowering the parameter (e.g. `lookahead=10`):

![unemployment3](/assets/images/bls_unemployment/unemployment3.svg){:height="60%" width="60%"}  

This is to say: The peaks and valleys of each cycle are determined methodically, but there is no mathematical optimization that was performed to determine that 12 cycles is the ideal number of cycles. This is something the user decides. I subjectively decided that 12 cycles most appropriately captures the important cycles.

# Brief Description of Seasonal Adjustment of Unemployment Data

Seasonality is an important concept in time series data analysis. We have only looked at seasonally adjusted data, meaning that the unemployment rate has been adjusted to account for things like temporary holiday hiring towards the end of the year (the method of adjustment is out of the scope of this post). The point here is that this seasonal, temporary hiring is not a true long-term trend indicative of a recovering economy. We can download the data that are not seasonally adjusted and plot them next to the seasonally adjusted data in order to understand how these two quantities differ:

![unemployment4](/assets/images/bls_unemployment/unemployment4.svg){:height="60%" width="60%"}  

Based on recent years, seasonal employment increases during the months of Sep - Dec and Apr - May.

# Forecasting Future Unemployment Trends

Now for a much harder problem: predicting unemployment trends using historical unemployment data.

## Choosing a Model for Forecasting

The [Forecasting: Principles and Practice](https://otexts.com/fpp2/) book describes many different modeling approaches that can be used for forecasting. There is no single "best" model that works for all time series datasets, and the specific nature of a dataset and the purpose of an analysis are both important considerations when choosing a modeling approach.

Now I will describe the steps that led to the final choice in forecasting model:

- **First define the goal of the analysis.**  
For this analysis, I wanted to forecast medium-term trends in unemployment over the next several years, rather than short-term seasonal fluctuations.
- **Select seasonally adjusted data.**  
Because the objective is the structural trend rather than seasonal behavior, I used the **seasonally adjusted** unemployment rate as the starting dataset.
- **Extract the historical trend using STL decomposition.**  
I applied STL (Seasonal-Trend decomposition using Loess) to decompose the seasonally adjusted series into Trend, Seasonality, and Remainder components. This isolated the quantity of interest for medium-term forecasting: the **trend**.  
Here is what that decomposition looks like:  
![unemployment5](/assets/images/bls_unemployment/unemployment5.svg){:height="60%" width="60%"}   
As you can see, the second row contains the Trend component of the decomposition. For the rest of the analysis, I will consider this Trend component as the historical "ground truth" to be used for model evaluation. The model forecasts this component only, and the seasonal and remainder components are intentionally ignored.  
- **Choose a set of candidate models appropriate for trend forecasting.**  
The next step is to narrow down potential models for forecasting. I focused on three modeling approaches: ARIMA (AutoRegressive Integrated Moving Average), ETS(A,A,N) (Error-Trend-Seasonality, or Exponential Smoothing), and a Local Linear Trend model (LLT, also called a structural model or unobserved components model). These modeling approaches were chosen due to their ability to model pre-extracted smooth trends, extrapolate reliably, and adapt to slope changes.
- **Evaluate the models using rolling-origin cross-validation.**  
I compared these modeling approaches using a rolling-origin cross-validation test, in which I tested how well each model can predict historical trend values when provided with the preceding data. For example, here is a plot illustrating the true trend vs forecasted trend in December 1958.  
![unemployment6](/assets/images/bls_unemployment/unemployment6.svg){:height="60%" width="60%"}   
For each modeling approach, I performed this comparison for each time-point (this is what is meant by "rolling-origin"), and I calculated the RMSE (root mean squared error) for all the time-points:

| ARIMA Trend | ETS Trend | LLT Trend |
|---:|---:|---:|
| 0.9183801 | 1.1766657 | 1.2320801 |

- I decided to use ARIMA because it has the lowest error. The Forecasting book mentioned above provides a great introduction to [ARIMA](https://otexts.com/fpp2/arima.html), an approach which models autocorrelations in the data.

## Forecasting Results

I used an ARIMA model on the trend component of the STL decomposition of the seasonally adjusted unemployment data to project a baseline unemployment path 3 years into the future:

<sub>Note: This is an interactive plot. You can hover over the plot and click on the autoscale button at the top right to view the whole date range going back to 1948. Click the home button to zoom back in.</sub>  
<iframe src="/assets/images/bls_unemployment/bls_plotly.html" width="70%" height="600px" frameborder="0"></iframe>

As you can see, the prediction interval gets wide quickly as the forecast goes farther into the future. By mid-2028, the prediction interval that corresponds to a 95% confidence level ranges from <1% to >7%. Also, the predicted trend looks flat to me. Not very exciting!

My takeaway from this analysis is that forecasting with any real precision is hard!

Disclaimer: It's best to think of the above plot as more of a "technical projection" than a reliable long-term forecast. Also, an expert making a well-informed prediction would probably incorporate information such as inflation, GDP, consumer sentiment surveys, knowledge of demographics, policy changes, and other current events into the model. Even then, there is [variability in estimates from forecasters](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/spf-q4-2025).

