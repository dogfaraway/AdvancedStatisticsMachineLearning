########################################################## Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the IDS and CADS(資訊與決策科學研究所暨資料科學應用研究中心), NTUB(國立臺北商業大學); the CARS(中華R軟體學會); and the DSBA(臺灣資料科學與商業應用協會)
#######################################################
### Notes: This code is provided without warranty.

# 線上音樂城關聯規則分析

# 線上音樂城聆聽記錄載入

import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

lastfm = pd.read_csv("~/cstsouMac/RandS/Rexamples/Ledolter/lastfm.csv")

# 檢視資料結構
lastfm.dtypes

# 獨一無二的使用者編號長度
lastfm.user.value_counts()[:5]
lastfm.user.unique().shape  # (15000,)

# 確認演唱藝人人數
lastfm.artist.value_counts()[:5]
lastfm.artist.unique().shape  # (1004,)

# https://www.tutorialspoint.com/python_pandas/python_pandas_groupby.htm
# https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/

grouped = lastfm.groupby('user')

type(grouped)  # pandas.core.groupby.groupby.DataFrameGroupBy

list(grouped)[:5]

list(grouped.groups.keys())[:20]  # 跳號

dir(grouped)
# 統計各使用者聆聽藝人數
numArt = grouped.agg({'artist': "count"})
numArt[5:10]
# grouped = lastfm.artist.groupby(by='user')
# 取出分組表藝人名稱一欄
grouped = grouped['artist']

# Python串列推導，拆解分組資料為串列
# grp_dict = {user:artist for (user, artist) in grouped}
# grp_list = [[artist] for (user, artist) in grouped]
music = [list(artist) for (user, artist) in grouped]
[x for x in music if len(x) < 3][:5]

from mlxtend.preprocessing import TransactionEncoder

# http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
# 交易資料格式編碼
te = TransactionEncoder()
te_ary = te.fit(music).transform(music)
# 檢視交易記錄筆數與品項數
te_ary.shape
te.columns_
# numpy陣列組織為二元值資料框
df = pd.DataFrame(te_ary, columns=te.columns_)
df.head()
# type(df)

### Sparse Representation
# oht_ary = te.fit(music).transform(music, sparse=True)
# sparse_df = pd.SparseDataFrame(oht_ary, columns=te.columns_, default_fill_value=False)
# sparse_df.head()
# type(sparse_df)

# apriori頻繁品項集探勘 
from mlxtend.frequent_patterns import apriori

# apriori(df, min_support=.01)

import time

start = time.time()
freq_itemsets = apriori(df, min_support=0.01, use_colnames=True)
end = time.time()
print(end - start)  # 50.427103996276855
# freq_itemsets.apply(lambda x: x.isnull().value_counts())
freq_itemsets['length'] = freq_itemsets['itemsets'].apply(lambda x: len(x))
# freq_itemsets.to_csv('/Users/Vince/cstsouMac/RBookWriting/bookdown-chinese-master/_mdl/freq.csv')
# freq_itemsets = pd.read_csv('./_mdl/freq.csv', index_col=0)
freq_itemsets.head()
freq_itemsets.dtypes

freq_itemsets[(freq_itemsets['length'] == 2) & (freq_itemsets['support'] >= 0.05)]

# association_rules關聯規則集生成
from mlxtend.frequent_patterns import association_rules

# 從頻繁品項集中產生49條規則(生成規則confidence >= 0.5)
musicrules = association_rules(freq_itemsets, metric="confidence", min_threshold=0.5)
# 從頻繁品項集中產生163條規則(生成規則lift >= 5)
# musicrules_small = association_rules(freq_itemsets, metric="lift", min_threshold=5)
musicrules.head()
musicrules['antecedent_len'] = musicrules['antecedents'].apply(lambda x: len(x))
musicrules.head()

# 進一步篩選規則
# at least 1 antecedents
# a confidence > 0.55
# a lift score > 5
musicrules[(musicrules['antecedent_len'] > 0) & (musicrules['confidence'] > 0.55) & (musicrules['lift'] > 5)]

# musicrules[(musicrules['confidence'] > 0.5) & (musicrules['lift'] > 5)]

# 小結：購物籃分析
#
# - Is capable of working with large amounts of transactional data 能夠處理大量的交易資料
# - Results in rules that are easy to understand 輸出的規則結果容易理解
# - Useful for "data mining" and discovering unexpected knowledge in databases 符合資料探勘挖掘資料庫中無預期知識的理念
#
# - Not very helpful for small datasets 小資料集用處不大
# - Requires effort to separate the true insight from common sense 訊息過載！需要從客觀(有趣性衡量)與主觀(規則樣板)角度下區辨出真正的洞見
# - Easy to draw spurious conclusions from random patterns 容易從隨機的型態中妄下虛假的結論
