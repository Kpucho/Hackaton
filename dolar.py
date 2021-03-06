import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split
from sklearn import metrics


if __name__ == '__main__':
    file = pd.ExcelFile('1.1.1.TCM_Serie histórica IQY.xlsx')
    df = pd.read_excel(file, 'Sheet1', index_col=False)
    date = df['Fecha (dd/mm/aaaa)']
    trm = df['Tasa de cambio representativa del mercado (TRM)']
    
    df.plot(x='Id', y='Tasa de cambio representativa del mercado (TRM)', style='o')
    plt.title('TRM')
    plt.show()
    
    plt.figure(figsize=(15,10))
    plt.tight_layout()
    seabornInstance.distplot(df['Tasa de cambio representativa del mercado (TRM)'])
    
    X = df['Id'].values.reshape(-1,1)
    
    y = df['Tasa de cambio representativa del mercado (TRM)'].values.reshape(-1,1)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    
    df1 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted':                     y_pred.flatten()})
    
    
    df2 = df1.head(25)
    df2.plot(kind='bar',figsize=(16,10))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()
    
    
    plt.scatter(X_test, y_test,  color='gray')
    plt.plot(X_test, y_pred, color='red', linewidth=2)
    plt.show()

    
    
    
        
    
