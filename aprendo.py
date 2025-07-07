import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# guardamos en una variable en forma de diccionario nuestros datos 
datos = {
    'Ciudad': ['Madrid', 'Barcelona', 'Valencia']*4,
    'Año':[2021] * 6 + [2022] * 6,
    'Mes': ['Enero', 'Enero', 'Enero', 'Febrero', 'Febrero', 'Febrero'] * 2,
    'Ventas': [1500, 2000, 1800, 1700, 2100, 1900, 1600, 2200, 2000, 1800, 2300, 2100]
}

datos_dataframe= pd.DataFrame(datos)

# vemos el nombre de nuestras columnas y si hay valores null en nuestro caso esta todo ok
print(datos_dataframe.info())

#vemos el total, media, desvio, min, max 
print(datos_dataframe.describe()) 

#Agrupamos para obtener las ventas totales por año y por ciudad
ventas_ciudad_año= datos_dataframe.groupby(['Ciudad','Año'])['Ventas'].sum().reset_index

                                                # G R A F I C O 


#definimos los colores de las barras
colores={
    'Madrid': '#1f77b4',
    'Barcelona': '#ff7f0e',
    'valencia': '#2ca02c'
}


#tabla pivotada para organizar todo lo que tenemos y pueda verse de la mejor manera
tabla_pivotada = ventas_ciudad_año.pivot(index='Año', columns='Ciudad',values='Ventas')

    

# le ponemos dimenciones a nuestro grafico 
plt.figure(figsize=(8,6))



    



