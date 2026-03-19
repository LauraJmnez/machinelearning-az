# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:11:50 2026

@author: laura
"""

#Regresión Lineal Múltiple

# Importar librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importar dataset
dataset = pd.read_csv("50_Startups.csv")

# Crearemos 2 variables:
#   variable X: representarán las variables independeientes del algorítmo (Columna Country, Age y Salary)
#   variable Y: la variable dependiente o la que querámos predecir (columna Purchassed)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

################################################################################################
#   Codificar datos categóricos --> El objetivo es asignar un numéro a cada valor categórico   #
################################################################################################
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:,3]) #Cambiamos los valores categóricos por los numéricos en el dataset

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct_X = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct_X.fit_transform(X))  # Pasamos esos numeros en columnas diferentes

# Evitar la trampa de las variables ficticias (Eliminamos SIEMPRE una de las variables Dummy)
X = X[:, 1:]


#######################################################################
#   Dividir dataset en conjunto de entrenamiento y conjunto testing   #
#######################################################################

#Al entrenar a la máquina es importante comprobar que funciona con un algoritmo y que no se ha aprendido los datos de memoria (over fitting), por lo que en vez de dos 
#(X e y) variables como hemos tenido hasta ahora, tendremos 4 (x_entrenamiento, x_testing, y_entrenamiento e y_testing)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # normalmente se reserva un 20% para testing, el valor random es como una semilla para que siemprenos de los mismos resultados

"""
##############################
#   Escalaado de variable    #
##############################
# Cuando hay una variable con un rango de valores mucho mayor (Salary) y otro con uno menor (Age), hay que normalizarlos para que ambos se muevan en un mismo rango
# y que sea el propio algoritmo el que deba discennir entre qué peso darle a cada variable no por tener un rango mayor o menor, sino porque realmente aportan
# más o menos en el proceso de predicción.
# Hay que distinguir entre estandarización (permite aglutinar valores entorno a la media, tendremos muchos valores cercanos a 0 y pocos alejados de él) 
# y normalización (trasnforma la columna de datos en un conjunto 0-1, el número más pequeño se transforma en 0 y el número mas grande en 1 y el resto se escala de 
# forma lineal)

# Las variables dummy se pueden escalar o no en función de neustro criterio, lo idela es que siempre se estandarice todo, pero esto depende de gustos. Sin embargo,
# la variable dependiente y en este caso no hay que estandarizarla ya que nuestro algoritmo es de clasificación (Compra o no compra). En casos de algoritmos de 
# predicción, como la regresión lineal, sí que se recomineda estandarizarlo también.
# 
#Estandarización:
# Cuando hablamos de regresion lineal simple, no hace falta ningún tipo de escalado

from sklearn. preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

#Ajustar el modelo de regresión lineal múltiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predicción de los resultados en el conjunto de testing
y_pred = regression.predict(X_test)


##############################################################################################
# Construir modelo óptimo de regresion lineal multiple utilizando la eliminación hacia atrás #
##############################################################################################
# Hemos usado todas las variables independientes disponibles, pero realmente no todas tienen por que ser significativas en la predicción
import statsmodels.api as sm

# En nuestro dataset actual de X_train no tenemos reflejado el término independiente (ordenada en el origen). A la hora de evaluar la significancia estadística del modelo podría 
# ser que sobrara ese término independiente (pero no hay modo de traquearlo automáticamente). Lo que se suele hacer es añadir una nueva columna en vertical al conjunto de datos 
# con "1". De esta forma se entiende que el coeficiente que se lleva los "1" será precisamente nuestro término independiente. Así podremos obtener también su p-valor

X = np.append(arr = np.ones((50,1)).astype(int), values = X , axis = 1) # añade esa columna de unos al principio

#☻ Como para la eliminación hacia atrás hay primero que coger todas las varibales:
X_opt = X[:, [0,1,2,3,4,5]].tolist()

# Paso 1
SL = 0.05
# Paso 2 : Calculo del modelo regresión linear múltiple con todas las posibles variables predictoras
regression_OLS = sm.OLS(y, X_opt).fit()
# Paso 3 : Considera la variable predictora con el p-valor más GRANDE
regression_OLS.summary()
"""
                           OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.951
Model:                            OLS   Adj. R-squared:                  0.945
Method:                 Least Squares   F-statistic:                     169.9
Date:                Thu, 19 Mar 2026   Prob (F-statistic):           1.34e-27
Time:                        12:29:20   Log-Likelihood:                -525.38
No. Observations:                  50   AIC:                             1063.
Df Residuals:                      44   BIC:                             1074.
Df Model:                           5                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.013e+04   6884.820      7.281      0.000    3.62e+04     6.4e+04
x1           198.7888   3371.007      0.059      0.953   -6595.030    6992.607
x2           -41.8870   3256.039     -0.013      0.990   -6604.003    6520.229  --> Eliminamos esta primero que es el que tiene un p-valor mayor (Vatiable Dummy Nueva York)
x3             0.8060      0.046     17.369      0.000       0.712       0.900
x4            -0.0270      0.052     -0.517      0.608      -0.132       0.078
x5             0.0270      0.017      1.574      0.123      -0.008       0.062
==============================================================================
Omnibus:                       14.782   Durbin-Watson:                   1.283
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               21.266
Skew:                          -0.948   Prob(JB):                     2.41e-05
Kurtosis:                       5.572   Cond. No.                     1.45e+06
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.45e+06. This might indicate that there are
strong multicollinearity or other numerical problems.
"""

# Paso 4 y vuelta al paso 3 : Eliminar la variable New York y volver a lanzar el modelo
X_opt = X[:, [0,1,3,4,5]].tolist()
regression_OLS = sm.OLS(y, X_opt).fit()
regression_OLS.summary()

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.951
Model:                            OLS   Adj. R-squared:                  0.946
Method:                 Least Squares   F-statistic:                     217.2
Date:                Thu, 19 Mar 2026   Prob (F-statistic):           8.49e-29
Time:                        12:34:31   Log-Likelihood:                -525.38
No. Observations:                  50   AIC:                             1061.
Df Residuals:                      45   BIC:                             1070.
Df Model:                           4                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.011e+04   6647.870      7.537      0.000    3.67e+04    6.35e+04
x1           220.1585   2900.536      0.076      0.940   -5621.821    6062.138  --> eliminamos ahora esta (Florida)
x2             0.8060      0.046     17.606      0.000       0.714       0.898
x3            -0.0270      0.052     -0.523      0.604      -0.131       0.077
x4             0.0270      0.017      1.592      0.118      -0.007       0.061
==============================================================================
Omnibus:                       14.758   Durbin-Watson:                   1.282
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               21.172
Skew:                          -0.948   Prob(JB):                     2.53e-05
Kurtosis:                       5.563   Cond. No.                     1.40e+06
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.4e+06. This might indicate that there are
strong multicollinearity or other numerical problems.
"""
# Paso 4 y vuelta al paso 3 : Eliminar la variable Florida y volver a lanzar el modelo
X_opt = X[:, [0,3,4,5]].tolist()
regression_OLS = sm.OLS(y, X_opt).fit()
regression_OLS.summary()

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.951
Model:                            OLS   Adj. R-squared:                  0.948
Method:                 Least Squares   F-statistic:                     296.0
Date:                Thu, 19 Mar 2026   Prob (F-statistic):           4.53e-30
Time:                        12:43:24   Log-Likelihood:                -525.39
No. Observations:                  50   AIC:                             1059.
Df Residuals:                      46   BIC:                             1066.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.012e+04   6572.353      7.626      0.000    3.69e+04    6.34e+04
x1             0.8057      0.045     17.846      0.000       0.715       0.897
x2            -0.0268      0.051     -0.526      0.602      -0.130       0.076  --> eliminamos esta (Admin)
x3             0.0272      0.016      1.655      0.105      -0.006       0.060
==============================================================================
Omnibus:                       14.838   Durbin-Watson:                   1.282
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               21.442
Skew:                          -0.949   Prob(JB):                     2.21e-05
Kurtosis:                       5.586   Cond. No.                     1.40e+06
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.4e+06. This might indicate that there are
strong multicollinearity or other numerical problems.
"""

# Paso 4 y vuelta al paso 3 : Eliminar la variable Admin y volver a lanzar el modelo
X_opt = X[:, [0,3,5]].tolist()
regression_OLS = sm.OLS(y, X_opt).fit()
regression_OLS.summary()

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.950
Model:                            OLS   Adj. R-squared:                  0.948
Method:                 Least Squares   F-statistic:                     450.8
Date:                Thu, 19 Mar 2026   Prob (F-statistic):           2.16e-31
Time:                        12:45:19   Log-Likelihood:                -525.54
No. Observations:                  50   AIC:                             1057.
Df Residuals:                      47   BIC:                             1063.
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       4.698e+04   2689.933     17.464      0.000    4.16e+04    5.24e+04
x1             0.7966      0.041     19.266      0.000       0.713       0.880
x2             0.0299      0.016      1.927      0.060      -0.001       0.061  --> este está en el límite. Siendo estrictos habría que eliminarla también y al final nos quedaríamos con un Regresion linear simple
==============================================================================      esto es un ejemplo de que a veces fijarnos solo en el p-valor es demasiado estricto
Omnibus:                       14.677   Durbin-Watson:                   1.257
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               21.161
Skew:                          -0.939   Prob(JB):                     2.54e-05
Kurtosis:                       5.575   Cond. No.                     5.32e+05
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 5.32e+05. This might indicate that there are
strong multicollinearity or other numerical problems.
"""

# Paso 4 y vuelta al paso 3 : Eliminar la variable Marketing y volver a lanzar el modelo
X_opt = X[:, [0,3]].tolist()
regression_OLS = sm.OLS(y, X_opt).fit()
regression_OLS.summary()

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.947
Model:                            OLS   Adj. R-squared:                  0.945
Method:                 Least Squares   F-statistic:                     849.8
Date:                Thu, 19 Mar 2026   Prob (F-statistic):           3.50e-32
Time:                        12:50:14   Log-Likelihood:                -527.44
No. Observations:                  50   AIC:                             1059.
Df Residuals:                      48   BIC:                             1063.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       4.903e+04   2537.897     19.320      0.000    4.39e+04    5.41e+04
x1             0.8543      0.029     29.151      0.000       0.795       0.913
==============================================================================
Omnibus:                       13.727   Durbin-Watson:                   1.116
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               18.536
Skew:                          -0.911   Prob(JB):                     9.44e-05
Kurtosis:                       5.361   Cond. No.                     1.65e+05
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.65e+05. This might indicate that there are
strong multicollinearity or other numerical problems.
"""

# Concluímos que solo en gasto en I+D (R&D) es la meeor variable predictora que tenemos para este modelo. Esta variable es tan fuerte que ha hecho que eliminemos las demás
# Mejor modelo : regresión Linear Simp"le  con la variable R&D únicamente


########################################
# Eliminación hacia atrás automatizado #
########################################

# Solo p-valor:
import statsmodels.api as sm
def backwardElimination(x, sl):    
    numVars = len(x[0])    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        if maxVar > sl:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    x = np.delete(x, j, 1)    
    regressor_OLS.summary()    
    return x 
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
regression_OLS.summary()
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.947
Model:                            OLS   Adj. R-squared:                  0.945
Method:                 Least Squares   F-statistic:                     849.8
Date:                Thu, 19 Mar 2026   Prob (F-statistic):           3.50e-32
Time:                        15:31:58   Log-Likelihood:                -527.44
No. Observations:                  50   AIC:                             1059.
Df Residuals:                      48   BIC:                             1063.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       4.903e+04   2537.897     19.320      0.000    4.39e+04    5.41e+04
x1             0.8543      0.029     29.151      0.000       0.795       0.913
==============================================================================
Omnibus:                       13.727   Durbin-Watson:                   1.116
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               18.536
Skew:                          -0.911   Prob(JB):                     9.44e-05
Kurtosis:                       5.361   Cond. No.                     1.65e+05
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.65e+05. This might indicate that there are
strong multicollinearity or other numerical problems.
"""

# p-valor y Rcuadrado ajsutado:
import statsmodels.api as sm
def backwardElimination(x, SL):    
    numVars = len(x[0])    
    temp = np.zeros((50,6)).astype(int)    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        adjR_before = regressor_OLS.rsquared_adj.astype(float)        
        if maxVar > SL:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    temp[:,j] = x[:, j]                    
                    x = np.delete(x, j, 1)                    
                    tmp_regressor = sm.OLS(y, x.tolist()).fit()                    
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)                    
                    if (adjR_before >= adjR_after):                        
                        x_rollback = np.hstack((x, temp[:,[0,j]]))                        
                        x_rollback = np.delete(x_rollback, j, 1)     
                        print (regressor_OLS.summary())                        
                        return x_rollback                    
                    else:                        
                        continue    
    regressor_OLS.summary()    
    return x 
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)  
"""
     OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.950
Model:                            OLS   Adj. R-squared:                  0.948
Method:                 Least Squares   F-statistic:                     450.8
Date:                Thu, 19 Mar 2026   Prob (F-statistic):           2.16e-31
Time:                        15:28:00   Log-Likelihood:                -525.54
No. Observations:                  50   AIC:                             1057.
Df Residuals:                      47   BIC:                             1063.
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       4.698e+04   2689.933     17.464      0.000    4.16e+04    5.24e+04
x1             0.7966      0.041     19.266      0.000       0.713       0.880
x2             0.0299      0.016      1.927      0.060      -0.001       0.061
==============================================================================
Omnibus:                       14.677   Durbin-Watson:                   1.257
Prob(Omnibus):                  0.001   Jarque-Bera (JB):               21.161
Skew:                          -0.939   Prob(JB):                     2.54e-05
Kurtosis:                       5.575   Cond. No.                     5.32e+05
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 5.32e+05. This might indicate that there are
strong multicollinearity or other numerical problems.
"""























