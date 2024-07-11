# 資料科學專案 #
# 加拿大溫莎市的房屋價格分析與預測 #

## 資料來源： ##
## House Prices in the City of Windsor, Canada ##
### 從R語言內建的加拿大溫莎市的房屋資料來做分析，資料為1987 年 7 月、8 月和 9 月期間加拿大溫莎市房屋的銷售價格。(資料共546筆，沒有遺失值) ###

## 目標(依變數) ##
*   價格(price)

## 自變數(11)： ##

*   坪數(lotsize)以平方英尺為單位
*   臥室數(bedrooms)
*   浴室數(bathrooms)
*   樓層數(stories)
*   有無車道(driveway)
*   有無娛樂室(recreation)
*   有無地下室(fullbase)
*   有無熱水器(gasheat)
*   有無中央空調(aircon)
*   車庫數(garage)
*   位於市區(prefer)
## [資料內容相關連結](https://vincentarelbundock.github.io/Rdatasets/doc/AER/HousePrices.html)
```python
import pandas as pd
import numpy as np
df = pd.read_csv('/content/HousePrices.csv',index_col=0)
df.head()
```
![image](https://github.com/hsu-chien-chung/DataScienceProject/assets/118785456/b94ad0c6-8f14-4cd1-b116-e45764e8b557)
### 資料正規畫 ###
資料本身的數據數值沒有統一，所以選擇將資料作正規化。

使用了最小最大標準化來將除了yes/no以外的資料來正規化，yes/no我選擇用Label encoding來數值化，這樣全部的數值只會介於0~1之間。
```python
#將資料做標準化
from sklearn import preprocessing
#將價格－樓層數、車庫數(price－stories、garage)用最小最大標準化
minmax = preprocessing.MinMaxScaler()
df.iloc[:,:5]= minmax.fit_transform(df.iloc[:,:5])
df.iloc[:,:5]=df.iloc[:,:5].round(3)
df.iloc[:,[-2]]= minmax.fit_transform(df.iloc[:,[-2]])
df.iloc[:,-2]=df.iloc[:,-2].round(3)
#將其他資料分(0,1)

df.replace('yes', 1,inplace=True)
df.replace('no', 0,inplace=True)

df
```





































































