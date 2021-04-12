[![Repo-Updater](https://github.com/firmai/financial-machine-learning/actions/workflows/repo_status.yml/badge.svg)](https://github.com/firmai/financial-machine-learning/actions/workflows/repo_status_weekly.yml)
[![Wiki-Generator](https://github.com/firmai/financial-machine-learning/actions/workflows/wiki_gen.yml/badge.svg)](https://github.com/firmai/financial-machine-learning/actions/workflows/wiki_gen_daily.yml)
[![Gitter](https://badges.gitter.im/financial-machine-learning/community.svg)](https://gitter.im/financial-machine-learning/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

# Financial Machine Learning and Data Science

A curated list of practical financial machine learning (FinML) tools and applications. This collection is primarily in Python. 

A listed repository should be deprecated if:

- Repository's owner explicitly say that "this library is not maintained".
- Not committed for long time (2~3 years).

**This repo is officially under revamp as of 3/29/2021!!**

- TODOs and roadmap is under the github project [here](https://github.com/firmai/financial-machine-learning/projects/1) 
- If you would like to contribute to this repo, please send us a pull request or contact [@dereknow](https://twitter.com/dereknow)  or [@bin-yang-algotune](https://twitter.com/b3yang) 
- Join us in the gitter chat [here](https://gitter.im/financial-machine-learning/community) 

___

- All repos/links status including last commit date is updated daily
- 10 Highest ranked repos/links for each section are displayed on main README.md and full list is available within the wiki page
- Both Wikis/README.md is updated in realtime as soon as new information are pushed to the repo 
___

# Trading
## Deep Learning & Reinforcement Learning ([Wiki](https://github.com/firmai/financial-machine-learning/wiki/deep_learning_and_reinforcement_learning))
<!-- [PLACEHOLDER_START:deep_learning_and_reinforcement_learning] --> 
| <sub>repo</sub>                                                                                                                                                                                                           | <sub>comment</sub>                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | <sub>created_at</sub>          | <sub>last_commit</sub>         | <sub>star_count</sub>   | <sub>repo_status</sub>              | <sub>rating</sub>   |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------|:-------------------------------|:------------------------|:------------------------------------|:--------------------|
| <sub>[Stock-Prediction-Models](https://github.com/huseinzol05/Stock-Prediction-Models)</sub>                                                                                                                              | <sub>very good curated list of notebooks showing deep learning + reinforcement learning models. Also contain topics on outlier detections/overbought oversold study/monte carlo simulartions/sentiment analysis from text (text storage/parsing is not detailed but it mentioned using [BERT](https://github.com/google-research/bert))</sub>                                                                                                                                                   | <sub>2017-12-18 10:49:59</sub> | <sub>2021-01-05 10:31:50</sub> | <sub>3655.0</sub>       | <sub>:heavy_check_mark:</sub>       | <sub>:star:x5</sub> |
| <sub>[AI Trading](https://github.com/borisbanushev/stockpredictionai/blob/master/readme2.md)</sub>                                                                                                                        | <sub>AI to predict stock market movements.</sub>                                                                                                                                                                                                                                                                                                                                                                                                                                                | <sub>2019-01-09 08:02:47</sub> | <sub>2019-02-11 16:32:47</sub> | <sub>2876.0</sub>       | <sub>:heavy_multiplication_x:</sub> | <sub>:star:x5</sub> |
| <sub>[FinRL-Library](https://github.com/AI4Finance-LLC/FinRL-Library)</sub>                                                                                                                                               | <sub>started by Columbia university engineering students and designed as an end to end deep reinforcement learning library for automated trading platform. Implementation of DQN DDQN DDPG etc using PyTorch and [gym](https://gym.openai.com/) use [pyfolio](https://github.com/quantopian/pyfolio) for showing backtesting stats. Big contributions on Proximal Policy Optimization (PPO) advantage actor critic (A2C) and Deep Deterministic Policy Gradient (DDPG) agents for trading</sub> | <sub>2020-07-26 13:18:16</sub> | <sub>2021-04-11 22:02:16</sub> | <sub>1857.0</sub>       | <sub>:heavy_check_mark:</sub>       | <sub>:star:x5</sub> |
| <sub>[Deep Learning IV](https://github.com/achillesrasquinha/bulbea)</sub>                                                                                                                                                | <sub>Bulbea: Deep Learning based Python Library.</sub>                                                                                                                                                                                                                                                                                                                                                                                                                                          | <sub>2017-03-09 06:11:06</sub> | <sub>2017-03-19 07:42:49</sub> | <sub>1467.0</sub>       | <sub>:heavy_multiplication_x:</sub> | <sub>:star:x5</sub> |
| <sub>[RLTrader](https://github.com/notadamking/RLTrader)</sub>                                                                                                                                                            | <sub>predecessor to [tensortrade](https://github.com/tensortrade-org/tensortrade) uses open api [gym](https://gym.openai.com/) and neat way to render matplotlib plots in real time. Also explains LSTM/data stationarity/Bayesian optimization using [Optuna](https://github.com/optuna/optuna) etc.</sub>                                                                                                                                                                                     | <sub>2019-04-27 18:35:15</sub> | <sub>2019-10-17 16:25:49</sub> | <sub>1312.0</sub>       | <sub>:heavy_check_mark:</sub>       | <sub>:star:x5</sub> |
| <sub>[Deep Learning III](https://github.com/Rachnog/Deep-Trading)</sub>                                                                                                                                                   | <sub>Algorithmic trading with deep learning experiments.</sub>                                                                                                                                                                                                                                                                                                                                                                                                                                  | <sub>2016-06-18 18:23:06</sub> | <sub>2018-08-07 15:24:45</sub> | <sub>1266.0</sub>       | <sub>:heavy_multiplication_x:</sub> | <sub>:star:x5</sub> |
| <sub>[Personae](https://github.com/Ceruleanacg/Personae)</sub>                                                                                                                                                            | <sub>implementation of deep reinforcement learning and supervised learnings covering areas: deep deterministic policy gradient (DDPG) and DDQN etc. Data are being pulled from [rqalpha](https://github.com/ricequant/rqalpha) which is a python backtest engine and have a nice docker image to run training/testing</sub>                                                                                                                                                                     | <sub>2018-03-10 11:22:00</sub> | <sub>2018-09-02 17:21:38</sub> | <sub>1144.0</sub>       | <sub>:heavy_multiplication_x:</sub> | <sub>:star:x5</sub> |
| <sub>[RL Trading](https://colab.research.google.com/drive/1FzLCI0AO3c7A4bp9Fi01UwXeoc7BN8sW)</sub>                                                                                                                        | <sub>A collection of 25+ Reinforcement Learning Trading Strategies -Google Colab.</sub>                                                                                                                                                                                                                                                                                                                                                                                                         | <sub>nan</sub>                 | <sub>nan</sub>                 | <sub>nan</sub>          | <sub>:heavy_check_mark:</sub>       | <sub>:star:x4</sub> |
| <sub>[Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020](https://github.com/AI4Finance-LLC/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020)</sub> | <sub>Part of FinRL and provided code for paper [deep reinformacement learning for automated stock trading](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996) focuses on ensemble.</sub>                                                                                                                                                                                                                                                                                              | <sub>2020-07-26 13:12:53</sub> | <sub>2021-01-21 18:11:59</sub> | <sub>560.0</sub>        | <sub>:heavy_check_mark:</sub>       | <sub>:star:x4</sub> |
| <sub>[awesome-deep-trading](https://github.com/cbailes/awesome-deep-trading)</sub>                                                                                                                                        | <sub>curated list of papers/repos on topics like CNN/LSTM/GAN/Reinforcement Learning etc. Categorized as deep learning for now but there are other topics here. Manually maintained by cbailes</sub>                                                                                                                                                                                                                                                                                            | <sub>2018-11-26 03:23:04</sub> | <sub>2021-01-01 09:41:21</sub> | <sub>551.0</sub>        | <sub>:heavy_check_mark:</sub>       | <sub>:star:x4</sub> |<!-- [PLACEHOLDER_END:deep_learning_and_reinforcement_learning] --> 
 

## Other Models ([Wiki](https://github.com/firmai/financial-machine-learning/wiki/other_models))
<!-- [PLACEHOLDER_START:other_models] --> 
| <sub>repo</sub>                                                                                                                                      | <sub>comment</sub>                                                  | <sub>created_at</sub>          | <sub>last_commit</sub>         | <sub>star_count</sub>   | <sub>repo_status</sub>              | <sub>rating</sub>   |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------|:-------------------------------|:-------------------------------|:------------------------|:------------------------------------|:--------------------|
| <sub>[Trend Following](http://inseaddataanalytics.github.io/INSEADAnalytics/ExerciseSet2.html)</sub>                                                 | <sub>A futures trend following portfolio investment strategy.</sub> | <sub>nan</sub>                 | <sub>nan</sub>                 | <sub>nan</sub>          | <sub>:heavy_check_mark:</sub>       | <sub></sub>         |
| <sub>[awesome-ai-in-finance](https://github.com/georgezouq/awesome-ai-in-finance)</sub>                                                              | <sub>NEW</sub>                                                      | <sub>2018-08-29 02:07:02</sub> | <sub>2020-11-27 09:43:40</sub> | <sub>941.0</sub>        | <sub>:heavy_check_mark:</sub>       | <sub></sub>         |
| <sub>[Scikit-learn Stock Prediction](https://github.com/robertmartin8/MachineLearningStocks)</sub>                                                   | <sub>Using python and scikit-learn to make stock predictions.</sub> | <sub>2017-02-12 04:50:44</sub> | <sub>2021-02-04 03:48:33</sub> | <sub>931.0</sub>        | <sub>:heavy_check_mark:</sub>       | <sub></sub>         |
| <sub>[Hands-On-Machine-Learning-for-Algorithmic-Trading](https://github.com/PacktPublishing/Hands-On-Machine-Learning-for-Algorithmic-Trading)</sub> | <sub>NEW</sub>                                                      | <sub>2019-05-07 11:04:25</sub> | <sub>2021-01-19 07:51:00</sub> | <sub>600.0</sub>        | <sub>:heavy_check_mark:</sub>       | <sub></sub>         |
| <sub>[AlphaPy](https://github.com/ScottfreeLLC/AlphaPy)</sub>                                                                                        | <sub>NEW</sub>                                                      | <sub>2016-02-14 00:47:32</sub> | <sub>2021-02-08 21:35:40</sub> | <sub>576.0</sub>        | <sub>:heavy_check_mark:</sub>       | <sub></sub>         |
| <sub>[MathAndScienceNotes](https://github.com/melling/MathAndScienceNotes)</sub>                                                                     | <sub>NEW</sub>                                                      | <sub>2016-03-11 19:13:00</sub> | <sub>2020-12-21 03:54:51</sub> | <sub>460.0</sub>        | <sub>:heavy_check_mark:</sub>       | <sub></sub>         |
| <sub>[Fundamental LT Forecasts](https://github.com/Hvass-Labs/FinanceOps)</sub>                                                                      | <sub>Research in investment finance for long term forecasts.</sub>  | <sub>2018-07-22 08:14:46</sub> | <sub>2021-02-17 14:39:30</sub> | <sub>383.0</sub>        | <sub>:heavy_check_mark:</sub>       | <sub></sub>         |
| <sub>[stock-trading-ml](https://github.com/yacoubb/stock-trading-ml)</sub>                                                                           | <sub>NEW</sub>                                                      | <sub>2019-10-10 09:44:02</sub> | <sub>2019-10-12 11:38:49</sub> | <sub>340.0</sub>        | <sub>:heavy_check_mark:</sub>       | <sub></sub>         |
| <sub>[Mixture  Models I](https://github.com/BlackArbsCEO/Mixture_Models)</sub>                                                                       | <sub>Mixture models to predict market bottoms.</sub>                | <sub>2017-03-20 18:54:24</sub> | <sub>2017-04-25 23:35:20</sub> | <sub>31.0</sub>         | <sub>:heavy_multiplication_x:</sub> | <sub></sub>         |
| <sub>[finance_ml](https://github.com/jjakimoto/finance_ml)</sub>                                                                                     | <sub>NEW</sub>                                                      | <sub>2018-06-29 21:21:17</sub> | <sub>2019-02-18 12:34:54</sub> | <sub>282.0</sub>        | <sub>:heavy_multiplication_x:</sub> | <sub></sub>         |<!-- [PLACEHOLDER_END:other_models] --> 
 

## Data Processing Techniques and Transformations ([Wiki](https://github.com/firmai/financial-machine-learning/wiki/data_processing_techniques_and_transformations))
<!-- [PLACEHOLDER_START:data_processing_techniques_and_transformations] --> 
| <sub>repo</sub>                                                                | <sub>comment</sub>                                                        | <sub>created_at</sub>          | <sub>last_commit</sub>         | <sub>star_count</sub>   | <sub>repo_status</sub>        | <sub>rating</sub>   |
|:-------------------------------------------------------------------------------|:--------------------------------------------------------------------------|:-------------------------------|:-------------------------------|:------------------------|:------------------------------|:--------------------|
| <sub>[Advanced ML II](https://github.com/hudson-and-thames/research)</sub>     | <sub>More implementations of Financial Machine Learning (De Prado).</sub> | <sub>nan</sub>                 | <sub>nan</sub>                 | <sub>nan</sub>          | <sub>:heavy_check_mark:</sub> | <sub></sub>         |
| <sub>[Advanced ML](https://github.com/BlackArbsCEO/Adv_Fin_ML_Exercises)</sub> | <sub>Exercises too Financial Machine Learning (De Prado).</sub>           | <sub>2018-04-25 17:22:40</sub> | <sub>2020-01-16 17:25:41</sub> | <sub>973.0</sub>        | <sub>:heavy_check_mark:</sub> | <sub></sub>         |<!-- [PLACEHOLDER_END:data_processing_techniques_and_transformations] --> 
 

# Portfolio Management
## Portfolio Selection and Optimisation ([Wiki](https://github.com/firmai/financial-machine-learning/wiki/portfolio_selection_and_optimisation))
<!-- [PLACEHOLDER_START:portfolio_selection_and_optimisation] -->
<!-- [PLACEHOLDER_END:portfolio_selection_and_optimisation] -->

## Factor and Risk Analysis ([Wiki](https://github.com/firmai/financial-machine-learning/wiki/factor_and_risk_analysis))
<!-- [PLACEHOLDER_START:factor_and_risk_analysis] -->
<!-- [PLACEHOLDER_END:factor_and_risk_analysis] -->

# Techniques
## Unsupervised ([Wiki](https://github.com/firmai/financial-machine-learning/wiki/unsupervised))
<!-- [PLACEHOLDER_START:unsupervised] -->
<!-- [PLACEHOLDER_END:unsupervised] -->


## Textual ([Wiki](https://github.com/firmai/financial-machine-learning/wiki/textual))
<!-- [PLACEHOLDER_START:textual] -->
<!-- [PLACEHOLDER_END:textual] -->

# Other Assets
## Derivatives and Hedging ([Wiki](https://github.com/firmai/financial-machine-learning/wiki/derivatives_and_hedging))
<!-- [PLACEHOLDER_START:derivatives_and_hedging] -->
<!-- [PLACEHOLDER_END:derivatives_and_hedging] -->

## Fixed Income ([Wiki](https://github.com/firmai/financial-machine-learning/wiki/fixed_income))
<!-- [PLACEHOLDER_START:fixed_income] -->
<!-- [PLACEHOLDER_END:fixed_income] -->

## Alternative Finance ([Wiki](https://github.com/firmai/financial-machine-learning/wiki/alternative_finance))
<!-- [PLACEHOLDER_START:alternative_finance] -->
<!-- [PLACEHOLDER_END:alternative_finance] -->

# Extended Research ([Wiki](https://github.com/firmai/financial-machine-learning/wiki/extended_research))
<!-- [PLACEHOLDER_START:extended_research] -->
<!-- [PLACEHOLDER_END:extended_research] -->

# Courses ([Wiki](https://github.com/firmai/financial-machine-learning/wiki/courses))
<!-- [PLACEHOLDER_START:courses] -->
<!-- [PLACEHOLDER_END:courses] -->

# Data ([Wiki](https://github.com/firmai/financial-machine-learning/wiki/data))
<!-- [PLACEHOLDER_START:data] -->
<!-- [PLACEHOLDER_END:data] -->

# Colleges, Centers and Departments ([Wiki](https://github.com/firmai/financial-machine-learning/wiki/colleges_centers_and_departments))
<!-- [PLACEHOLDER_START:colleges_centers_and_departments] -->
<!-- [PLACEHOLDER_END:colleges_centers_and_departments] -->
