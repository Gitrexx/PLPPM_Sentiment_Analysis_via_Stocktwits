# PLPPM_Sentiment_Analysis_via_Stocktwits

#### *A project by NUS ISS student Yun Cao, Gerong Zhang, Jiaqi Yao, Sikai Ni, Yunduo Zhang*
GitHub Link: https://github.com/Gitrexx/PLPPM_Sentiment_Analysis_via_Stocktwits

<br />

### Project objectives

To make money by prediction the stock price movement.

Just kidding...

Try to see if text sentiment mining in Stocktwits (a investment forum) can improve predictive analysis for stock price movement. 

<br />

### Code function description

- DataCollection

    To scrape data from Stocktwits API, our scrapped dataset [here](www.kaggle.com/frankcaoyun/stocktwits-2020-2022-raw)
    
- Data Processing and Feature Engineering

    To clean, manapulate, transform, and aggregate the data into usable formats.
    
- EDA
    - Tokens Distribution
    - Emoji Clouds
    - Topic Modelling
    - Time Series: Trend of Daily Average Sentiment
    
- SentimentEngine

    - Test several models and select best model for training
    - Tune Hyper-parameters on the selected model
    - Training
    - Use model for sentiment inferencing
    - model [here](https://huggingface.co/zhayunduo/roberta-base-stocktwits-finetuned)
   
- CorrelationAnalysis

    - Build a classification model for stock movement classification
