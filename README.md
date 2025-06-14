Banks Review Analysis

Project Overview

This project focuses on scraping a Banks dataset from google play store to derive insights on user behavior, transaction performance, and Account Access Issues. The main objective is to help banks decide whether to focus on acquiring Customer Support or improving existing services. We will achieve this by identifying patterns in transaction performance, log in error, user interface, and user engagement metrics.
Data Collection and preprocessing 
Data Collection
I gather multiple dataset for bank of Commercial Bank of Ethiopia (CBE), Bank of Abyssinia (BOA) and Dashen Bank from google play store. I use the following app id to get the dataset.
?	com.combanketh.mobilebanking
?	com.boa.boaMobileBanking
?	com.dashen.dashensuperapp
I utilize the google-play-scraper to scrape over 400 reviews per bank.
Data Cleaning
After downloading the data set then removes duplicates, handle missing values and replaces the missing value using mean then finally I normalize date formats for consistency.  As output I save the cleaned data as CSV files.
Setup Environment
The first step is to set up a Python development environment with version control and implement CI/CD workflows for continuous integration and deployment.
Deliverables
�	Python environment setup.
�	GitHub repository with branches and version control.
User Overview Analysis
Objective
The goal is to gain insights into customer behavior by analyzing their device preferences, network usage patterns, and app engagement.
Tasks
�	Create a GitHub repository and set up branch task-1.
�	Perform exploratory data analysis (EDA) on the dataset to identify key insights regarding customer behavior.

