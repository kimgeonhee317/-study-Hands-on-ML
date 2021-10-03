#%%
import pandas as pd
import os

HOUSING_PATH = "../datasets/housing"

#print(os.system("pwd")) // 경로 확인

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path) # 해당 경로를 읽어들여 데이터프레임 반환 
housing = load_housing_data()
housing.head() # 상위 5개 row

# %%
housing.info() # data feature 정보

#%%
housing.describe() # feature 별 데이터

# %%
import matplotlib.pyplot as plt
plt.style.use('dark_background')
housing.hist(bins=50, figsize=(20,15))
plt.show()
# %%
import numpy as np
np.random.seed(42) #난수 seed를 설정하여 난수를 고정
def split_train_Test(data, test_ratio=0.2):
    shuffled_indices = np.random.permutation(len(data)) 
    #데이터의 크기만큼의 인덱스 집합을 랜덤화 시켜서 저장
    shuffled_indices[0] 
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size] #앞 부분을 test세트로 
    train_indices = shuffled_indices[test_set_size:] #뒷 부분을 train 세트로
    return data.iloc[train_indices], data.iloc[test_indices]
    # iloc메서드는 정수 위치 기반 인덱싱으로 해당 데이터를 반환해 준다.

train_set, test_set = split_train_Test(housing, 0.2)
print(len(train_set), "train+", len(test_set), "test")
print(train_set.head())

# %%
from zlib import crc32

def test_set_check(identifier, test_ratio):
    #print( hex(crc32(np.int64(identifier))))
    #print( hex(2**32) )
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32
# crc32는 해당 id에 대한 32bit unsigned(부호없는) checksum integer 값을 반환
# 관련 참고값을 위해 hex 값을 print해봤다.

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_ : test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]
# apply-lambda : 
# We can use the apply() function to apply the lambda function 
# to both rows and columns of a dataframe.

housing_with_id = housing.reset_index() # 'index' 열이 추가된 데이터프레임 반환
housing_with_id["id"] = housing["longitude"]*1000 + housing["latitude"]

train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")


# %%
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

train_set.head()

# %%
housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
print(housing["income_cat"].head(), housing["income_cat"].value_counts())
print(housing["median_income"].head())
# %%
