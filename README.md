# Berry Yield Prediction

*Project Date: 2021-03-15*

## Introduction

This is a dashboard aiming to analyze the berry yield dataset recorded on a weekly basis. The major goal of this project is to predict the berry yield in the next 10 weeks. The original dataset only contains the production information, and no other information like the dates of records is provided. Being a typical time series problem with apparent seasonal trend, the following models are applied to get the prediction results. 

- Seasonal Autoregressive Integrated Moving Average (SARIMA)
- Facebook Prophet 

After the model selection process based on the MSE in the validation set, SARIMA is decided as the final model. The prediction results of both models will be exhibited in this dashboard.

<!-- Complete analysis process is stored in a Jupyter Notebook file named as `Naturipe_Farms.ipynb`. Due to confidential purpose, this file is stored in [another private GitHub repository](https://github.com/Mingxuan-Yang/Berry-Yield-Prediction-Appendix). -->

## Streamlit Dashboard

This dashboard includes the following three parts:

- **Introduction**
- **Data Analysis**
- **Prediction**

The **Data Analysis** section uses the original data to create visualizations and conduct analysis, while the predicted values are incorporated for analysis in the **Prediction** part.

The dashboard can be obtained by running the following code in the terminal:

```
streamlit run app.py
```

This dashboard is also deployed through Streamlit Community Cloud at [this link](https://berry-yield-prediction-mingxuan-yang.streamlit.app/).
