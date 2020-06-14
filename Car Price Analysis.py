import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)


# # Data Collection

data = pd.read_csv('car_price.csv')
data.head()

company = data['CarName'].apply(lambda x : x.split(' ')[0])
data.insert(2,"CompanyName",company)
data.drop(['CarName'],axis = 1, inplace=True)
data.drop(['car_ID'],axis=1,inplace=True)
data.head()
data.price.describe()


# # Data Cleaning

#There are mispellings in the data, need to correct them
data.CompanyName.unique()
def rename(old,new):
    data.CompanyName.replace(old,new,inplace=True)

rename('porcshce','porsche')
rename('maxda','mazda')
rename('vw','volkswagen')   
rename('toyouta','toyota')
rename('vokswagen','volkswagen')

data.CompanyName.unique()
plt.title('Price Distribution')
sns.distplot(data.price)

df = pd.DataFrame(data.groupby(['CompanyName'])['price'].mean().sort_values(ascending = False))
df.plot.bar()
plt.title('Avg Price by Company')

def plot_price(attr1,attr2,attr3):
    sns.pairplot(data, x_vars=[attr1,attr2,attr3], y_vars='price',height=4, aspect=1, kind='scatter')
    plt.show()
    
plot_price('enginesize', 'horsepower', 'highwaympg')


## We can identify which attributes of the vehicle affect car price. 
## Doing this for all attributes, we make a new dataframe using only highly correlated variables



# Derived property using a common formula for fuel economy
data['fueleconomy'] = (0.55 * data['citympg']) + (0.45 * data['highwaympg'])

car_final = data[['price', 'fueltype', 'aspiration','carbody', 'drivewheel','wheelbase',
                  'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'boreratio','horsepower', 
                    'fueleconomy', 'carlength','carwidth']]
car_final.head()


# In order to input categorical data into the model we need to convert the colums to binary
# To do this we create 'dummies' (eg. binary columns for gas/diesel)
def create_dummies(category,df):
    new_df = pd.get_dummies(df[category], drop_first = True)
    df = pd.concat([df, new_df], axis = 1)
    df.drop([category], axis = 1, inplace = True)
    return df


car_final = create_dummies('fueltype',car_final)
car_final = create_dummies('aspiration',car_final)
car_final = create_dummies('carbody',car_final)
car_final = create_dummies('drivewheel',car_final)
car_final = create_dummies('enginetype',car_final)
car_final = create_dummies('cylindernumber',car_final)

car_final.head()


# # Modeling

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

# Splitting data into training and testing
data_train, data_test = train_test_split(car_final, train_size = 0.80, test_size = 0.20)

# Extracting independent variables (car attributes) and dependent variables (price)
y_train = data_train.pop('price')
x_train = data_train

y_test = data_test.pop('price')
x_test = data_test

#Applying a multiple linear regression model 
linReg = LinearRegression()
linReg.fit(x_train,y_train)
y_pred = linReg.predict(x_test)

plt.scatter(y_test,y_pred)
plt.xlabel('Known Car Price')
plt.ylabel('Predicted Car Price')
plt.plot(y_test,y_test, c = 'red')

