# 資料科學專案:加拿大溫莎市的房屋價格分析與預測

## 資料來源：House Prices in the City of Windsor, Canada
### 從R語言內建的加拿大溫莎市的房屋資料來做分析，資料為1987 年 7 月、8 月和 9 月期間加拿大溫莎市房屋的銷售價格。(資料共546筆，沒有遺失值)
__[資料內容相關連結](https://vincentarelbundock.github.io/Rdatasets/doc/AER/HousePrices.html)__

## 目標(依變數)
*   價格(price)

## 自變數(11)：

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

### 資料載入

```python
import pandas as pd
import numpy as np
df = pd.read_csv('/content/HousePrices.csv',index_col=0)
df.head()
```
![image](https://github.com/hsu-chien-chung/DataScienceProject/assets/118785456/b94ad0c6-8f14-4cd1-b116-e45764e8b557)

### 資料正規化

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

### 資料分析：

#### 顯示目標與變數之間的趨勢與關係

```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(10,8)})

sns.pairplot(df,kind="reg",plot_kws={'line_kws':{'color':'r'},'scatter_kws':{'color':'b'}},diag_kws=dict(color='g'),corner=True)
plt.show()
```
![image](https://github.com/user-attachments/assets/8555194d-cc4f-4285-a4d8-3c3ee19449a3)
由於正規化後，很多數值都是0或1比較難看出趨勢。



#### 相關性矩陣與矩陣圖

```python
df.corr().style.background_gradient(cmap='bwr_r', axis=None).format("{:.2}")
#相關性矩陣，從表可得知所有變數的相關性都落在中度相關與弱相關，裡面的數值沒有0.7以上的強相關。
```
![image](https://github.com/user-attachments/assets/65a4510b-bfb8-4ff6-90c3-862090da4ac5)

```python

sns.set(rc={'figure.figsize':(10,8)})
sns.heatmap(df.corr())
#做相關性矩陣圖，從圖可以看到顏色越淺表示相關性越強。
```
![image](https://github.com/user-attachments/assets/977f6a96-b68e-490b-9d74-63be40eb65aa)


# 從相關性矩陣圖可得知，目標與變數之間沒有較強的相關性。  

#### 基礎的資料與箱型圖

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
![image](https://github.com/user-attachments/assets/4406d479-d3a7-40b1-b6f4-2bd28fcd3682)

```python
sns.histplot(data=df,x='price')

#購買房屋價格如同人民薪資所得一樣是右偏斜。
```
![image](https://github.com/user-attachments/assets/060f0a1d-2c63-478e-aef0-ab8fd87de4db)
### 主成分分析

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
![image](https://github.com/user-attachments/assets/3ab0c8dc-a095-4c64-bd80-3d9b5fe8757d)

第1主成分在經由特徵去收尋加拿大的房屋性質，我判斷是接近市區的獨立屋。  
第2主成分為鎮屋，房屋都是連在一起的，像是歐美影片裡常出現的房子。  
第3主成分可能是鄉間的小別墅。  

```python
pca = PCA(3)
pca.fit(X)
np.round(pca.explained_variance_ratio_, 2)
print(pca.explained_variance_,"\n") #解釋變量
print(pca.explained_variance_ratio_) #3維的解釋變量的比例
```

由於前三項解釋能力只有60%多，所以我想看所有的主成分解釋力。

```python
pca_10d = PCA(11)
pca_10d.fit(X)
np.round(pca_10d.explained_variance_ratio_, 2)
```

從上得知主成分分析的結果是不佳的，畢竟如果選擇前三項主成分來分析，會有接近40%的變數解釋力損失，取到90%以上，就喪失了用主成分降維的意義。  
(備註：變數轉換公式:原始資料*(pca解釋變量(X))，就會有新的變數。)

### 集群分析

#### K-Means分群

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score  #這是測試KMeans要分幾群最理想的套件

X = df.iloc[:,1:]

kmeans_list = [KMeans(n_clusters=k, random_state=46).fit(X) for k in range(1, 12)]
inertias = [model.inertia_ for model in kmeans_list]
silhouette_scores = [silhouette_score(X, model.labels_) for model in kmeans_list[1:]]
```

#### 選擇最佳的分群數，

```python
# 方法一:
sns.lineplot(x=range(1,12),y=inertias)  #線轉為平緩的點代表最適的分群，可是此圖的點落在分11群。
```

```python
# 方法二:
sns.lineplot(x=range(2,12),y=silhouette_scores) #分數越高的點表示最適的分群，而此圖也落在分9或11群
```

由於上方數據得知這組資料不適合k-means來分群。  

#### 嘗試實作分兩群

```python
import copy
Kmeans = KMeans(n_clusters=2)

X = df.iloc[:,1:]
Kmeans.fit(X)

df1 = df.copy()    #複製一個新的data

df1['Kmeans']=Kmeans.labels_  #將分好的值都入d值都入df1
df1
```

使用相關性矩陣圖來判斷變數與分群的相關性。

```python
sns.heatmap(df1.corr()) #從中能發現有一個變數(地下室)是強相關
```

```python
df1.corr().style.background_gradient(cmap='bwr_r', axis=None).format("{:.2}") 

#相關性矩陣的地下室數值與Kmeans分類是完全一樣的，能判斷他是用有無地下室來分類。
```

從之前的PCA和KMean得知這筆資料可能不適合用來分群，接下來我打算用神經網路(NN)來訓練並預測。

### 神經網路(NN)

```python
#載入所需套件
import keras
from pandas.core.frame import to_arrays
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
```

```python
# 畫圖函式
def PLT(history):
  
  # "Accuracy/Val_accuracy"
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.show()

  # "Loss/Val_loss"
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.show()
```

將價格強制分成3類，分別為前25%,中段的50%和後段的25%。

```python
sns.set(rc={'figure.figsize':(10,5)})
sns.boxplot(data=df,x='price')
print(df.iloc[:,0].describe())
```

```python
df.iloc[df.iloc[:,0]>0.345,0]=3
df.iloc[df.iloc[:,0]<0.14575,0]=1
df.iloc[df.iloc[:,0]<1,0]=2

df.iloc[:,0]=df.iloc[:,0].astype('str')  #我先將資料類型轉成文字以方便之後的機器人計算
df.price.value_counts()   #能看到資料已經分類好了
```

資料分割:60%訓練集、20%測試集、20%驗證集。

```python
#透過套件將資料分割
train_df,test_df=train_test_split(df,train_size=0.8) #先分出20%測試集，之後會再分出20%驗證集

train_df = np.array(train_df)
test_df = np.array(test_df)

train_X = train_df[:,1:].astype(float)  #將X資料轉換成統一的類別
test_X = test_df[:,1:].astype(float)

print(train_X.shape)   #確認每個維度的個數
print(test_X.shape)

train_y = train_df[:,0]
test_y = test_df[:,0]

print(train_y.shape)  
print(test_y.shape)
```

#### 使用One-hot encoding 編碼

```python
# 文字類別轉換成0與1編成的個碼
# 0ne-hot encoding
train_y = pd.get_dummies(train_y).to_numpy()  #將dataframe轉換成array,以方便機器人運算
test_y = pd.get_dummies(test_y).to_numpy()
```

```python
#設定模型
from keras.backend import dropout
model = Sequential()
model.add(Dropout(0.1, input_dim=11))        #使用Dropout來避免過度擬合，設定0.1表示有10%的神經元會被隨機丟棄 
model.add(Dense(11,activation='relu'))         #使用非線性函數
model.add(Dense(3,activation='softmax'))        #要分三群，所以輸出為3

adam = Adam(lr=0.001)      #使用Adam梯度下降，學習率為0.001
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])  #損失函數是分類交叉商
model.summary()
```

#### 訓練模型

```python
history = model.fit(train_X, train_y, validation_split=0.25, batch_size=4, epochs=500)

#validation_split=0.25是指將train_X的資料集中的0.25做為驗證集 0.8*0.25=0.2 ,就能達成60%訓練、20%測試、20%驗證
```

訓練過程的成功率與損失值的圖表(含驗證集)。

```python
PLT(history) #圖片結果顯示橘線的最後的準確率有向下的趨勢，損失值的中後段有向上趨勢，代表還是有過度擬合的發生。
```

模型的測試分數。

```python
test_loss, test_acc = model.evaluate(test_X, test_y)
print('\nTest accuracy:', test_acc)  #準確率來到72.73%，還可以
```
