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
![image](https://github.com/hsu-chien-chung/DataScienceProject/assets/118785456/a0c97322-2514-4797-9021-e9086ef6dc76)
### 資料分析：###
顯示目標與變數之間的趨勢與關係
```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(10,8)})

sns.pairplot(df,kind="reg",plot_kws={'line_kws':{'color':'r'},'scatter_kws':{'color':'b'}},diag_kws=dict(color='g'),corner=True)
plt.show()
#由於正規化後，很多數值都是0或1比較難看出趨勢。
```
相關性矩陣與矩陣圖
```python
df.corr().style.background_gradient(cmap='bwr_r', axis=None).format("{:.2}")
#相關性矩陣，從表可得知所有變數的相關性都落在中度相關與弱相關，裡面的數值沒有0.7以上的強相關。
```

```python
sns.set(rc={'figure.figsize':(10,8)})
sns.heatmap(df.corr())
#做相關性矩陣圖，從圖可以看到顏色越淺表示相關性越強。
```
從相關性矩陣圖可得知，目標與變數之間沒有較強的相關性。

基礎的資料與箱型圖
```python
print(df.iloc[:,0].describe()) #顯示價格的基礎資料
sns.set(rc={'figure.figsize':(10,5)})
sns.boxplot(data=df,x='price')
IQR=(df.iloc[:,0].quantile(0.75)-df.iloc[:,0].quantile(0.25)).round(2) #四分位距IQR
if (df.iloc[:,0].quantile(0.25)-1.5*IQR) < 0:  #四分位距下邊界計算
  Lower=0
else:
  Lower=(df.iloc[:,0].quantile(0.25)-1.5*IQR).round(2) 
Upper=(df.iloc[:,0].quantile(0.75)+1.5*IQR).round(2) #四分位距上邊界計算
print('\nIQR=',IQR,'下邊界=',Lower,'上邊界=',Upper)
```

```python
sns.histplot(data=df,x='price')
#購買房屋價格如同人民薪資所得一樣是右偏斜。
```
### 主成分分析 ###
變數量多，所以用資料降維來方便分析。
```python
from sklearn.decomposition import PCA
X = df.iloc[:,1:]
y = df.iloc[:,0]
pca = PCA(n_components=3)
pca.fit_transform(X) #轉換後3維
pcs = np.array(pca.components_) #特徵向量
df_pc = pd.DataFrame(pcs, columns=df.columns[1:])
df_pc.index = [f"第{c}主成分" for c in['一', '二', '三']]
df_pc.style.background_gradient(cmap='bwr_r', axis=None).format("{:.2}")

#第1主成分在變數為地下室、娛樂室、市區、中央空調的特徵解釋占比大，並以正相關性為主。
#第2主成分在變數中央空調、地下室、樓層數的特徵解釋占比大，除了地下室其他都是正相關性為主。
#第3主成分在在市區、車道、中央空調、地下室的特徵解釋占比大，車道和在市區是負相關性、其他都是正相關性。
```
第1主成分在經由特徵去收尋加拿大的房屋性質，我判斷是接近市區的獨立屋。  
第2主成分為鎮屋，房屋都是連在一起的，像是歐美影片裡常出現的房子。  
第3主成分可能是鄉間的小別墅。


























































