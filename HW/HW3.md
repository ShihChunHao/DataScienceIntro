# Data Science Homework3

##### *The purpose of Homework3 is to "Use Python and/or others to analyze some data sets with missing data (NA)".*

Since I found that it was too inefficient to write reports, the code needed to be cut off constantly, and there were many visual tools that could not be easily presented in PDF format, I decided to save the code directly through Github and use Markdown syntax to type out the text to explain my steps at the beginning of my third assignment.

## Overview
我所選擇的資料集是基於Kaggle競賽：**G-Research Crypto Competition** 所提供的資料，該競賽為了比賽常常會將資料挖空，來模擬真實世界中人為有意或無意造成的資料丟失。資料丟失或錯誤的狀況，可以讓ML科學家嘗試各種方法，來增強算法的魯棒性以及可用性。面對這樣的問題，我們將在這次的作業進行實作，處理資料集的資料丟失與資料錯誤問題。

## Introduction
![Fig1](./../HW3FIGURE/Figure1.png)
First, a quick introduction to the crypto world. Cryptocurrencies have become an extremely popular and volatile market, delivering massive returns (as well as losses) to investors. Thousands of cryptocurrencies have been created with a few major ones that many of you will have heard of including Bitcoin (BTC), Ether (ETH) or Dogecoin (DOGE).

Cryptocurrencies are traded extensively across crypto-exchanges, with an average volume of USD 41 billion traded daily over the last year, according to CryptoCompare (as of 25th July 2021).

Changes in prices between different cryptocurrencies are highly interconnected. For example, Bitcoin has historically been a major driver of price changes across cryptocurrencies but other coins also impact the market.


## Prerequisites
首先，基於上次作業建好的環境，我們需要在終端機先確認環境是否已經滿足所有我們將使用到的函示庫。我們需要先引入下列的函示庫，以免產生報錯。

```bash=
$ pip3 install pandas
$ pip3 install numpy
$ pip3 install datetime
$ pip3 install plotly
$ pip3 install matplotlib
$ pip3 install time
$ pip3 install scipy
```