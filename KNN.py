#单变量KNN

import pandas as pd
import numpy as np
from scipy.spatial import distance



listings = pd.read_csv('listing.csv')
feactures = ['accommodates','bedrooms','bathrooms','beds','price','minimum_nights','maximum_nights','number_of_reviews']
listings = listings[feactures]
print(listings.shape)
listings.head()

our_acc_value =3
listings['distance'] = np.abs(listings.accomodates - our_acc_value )
listings['distance'].value_counts().sort_index()

listings = listings.sample(frac = 1,random_state=0) #打乱顺序
listings = listings.sort_values('distance')
listings['price'].head()

listings['price'] = listings['price'].str.replace('\$|','').astype(float)
mean_price = listings.iloc[:5]['price'].mean()
print(mean_price)



#模型评估

listings.drop('distance',axis = 1)
train_df = listings.copy().iloc[:2792]  #训练集
test_df = listings.copy().iloc[2792:]   #测试集

def predict_price(new_list_value,feature_column):
    temp_df = train_df
    temp_df['distance'] = np.abs(listings[feature_column] - new_list_value)
    temp_df = temp_df.sort_values('distance')
    knn_5 = temp_df.price.iloc[:5]
    predicted_price = knn_5.mean()
    return predicted_price

test_df['predicted_price'] = test_df.accommodates.apply(predict_price,feature_columns = 'accommodates')
test_df['squared_error'] = (test_df['predicted_price'] - test_df['price'])**2
mse = test_df['squared_error'].mean()
rmse = mse ** (1/2)
print(rmse)


#不同变量KNN

for feature in ['accommodates','bedrooms','bathrooms','number_of_reviews']:
    test_df['predicted_price'] = test_df[feature].apply(predict_price, feature_columns='feature')
    test_df['squared_error'] = (test_df['predicted_price'] - test_df['price']) ** 2
    mse = test_df['squared_error'].mean()
    rmse = mse ** (1 / 2)
    print('RMSE fo the {} column: {}'.format(feature,rmse))


#多变量KNN

#先标准化

from sklearn.preprocessing import StandardScaler
feactures = ['accommodates','bedrooms','bathrooms','beds','price','minimum_nights','maximum_nights','number_of_reviews']
listings = pd.read_csv('listings.csv')
listings = listings[feactures]
listings['price'] = listings['price'].str.replace('\$|','').astype(float)
listings = listings.dropna()
std = StandardScaler()
listings = std.fit_transform(listings)
listings.head()

norm_train_df = listings.copy().iloc[:2792]
norm_test_df = listings.copy().iloc[2792:]

def predict_price_multivariate(new_listing_value,feature_column):
    temp_df = norm_train_df
    temp_df['distance'] = distance.cdist(temp_df[feature_column,new_listing_value[feature_column]])
    temp_df = temp_df.sort_values('distance')
    knn_5 = temp_df['price'].iloc[:5]
    predicted_price = knn_5.mean()
    return predicted_price

cols = ['accommodates','bathrooms']
norm_test_df['predicted_price'] = norm_test_df[cols].apply(predict_price_multivariate, feature_column =cols,axis = 1 )
norm_test_df['squared_error'] = (norm_test_df['predicted_price']-norm_test_df['price'])**(2)
mse = norm_test_df['squared_error'].mean()
rmse = mse ** (1/2)
print(rmse)


##使用sklearn来完成KNN
from sklearn.neighbors import KNeighborsRegressor
cols = ['accommodates','bedrooms']
knn = KNeighborsRegressor()
knn.fit(norm_train_df[cols],norm_train_df['price'])
knn.predict(norm_test_df[cols])

