import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



#Guardamos en una variable en forma de diccionario nuestros datos 
datos = {
    'Ciudad': ['Madrid', 'Barcelona', 'Valencia']*4,
    'Año':[2021] * 6 + [2022] * 6,
    'Mes': ['Enero', 'Enero', 'Enero', 'Febrero', 'Febrero', 'Febrero'] * 2,
    'Ventas': [1500, 2000, 1800, 1700, 2100, 1900, 1600, 2200, 2000, 1800, 2300, 2100]
}

datos_dataframe= pd.DataFrame(datos) #Convertimos nuestro diccionario en un dataframe con pandas

#Vemos el nombre de nuestras columnas y si hay valores null en nuestro caso esta todo ok y por eso no se produce un error de tal caso que no fuera asi tendriamos que recorrer el diccionario y las listas para que ver cual tiene menos elementos y asi poder solucionarlo 
print(datos_dataframe.info())

#Vemos el total, media, desvio, min, max 
print(datos_dataframe.describe()) 

#Agrupamos para obtener las ventas totales por año y por ciudad
ventas_ciudad_año= datos_dataframe.groupby(['Ciudad','Año'])['Ventas'].sum().reset_index()

                                                # G R A F I C O 


#Definimos los colores de las barras
colores={
    'Madrid': '#1f77b4',
    'Barcelona': '#ff7f0e',
    'Valencia': '#2ca02c'
}


#Tabla pivotada para organizar todo lo que tenemos y pueda verse de la mejor manera
tabla_pivotada = ventas_ciudad_año.pivot(index='Año', columns='Ciudad',values='Ventas')

    

# le ponemos dimenciones a nuestro grafico 
plt.figure(figsize=(8,6))

position_x= np.arange(len(tabla_pivotada.index)) #Creamos una array en el cual metemos el numero de años osea  que en este caso es nuestro index, basicamente estamos diciendole cuantas posiciones va a tener nuestra posicion en x 
ancho= 0.25

#Creamos un for para darle un indice a cada ciudad y recorrer cada ciudad 

for i, ciudad in enumerate(tabla_pivotada.columns):
    ventas = tabla_pivotada[ciudad] #Selecionamos cada columna de la ciudades y obtenemos sus ventas
    posicion = position_x + i*ancho #Definimos en que año esta cada ciudad y luego de eso le damos el desplazamiento en x con i * ancho, para asi no superponer cada barra 


    #Definimos que el grafico sera de barras la posicion del los años y en ellos las tres barras que corresponden a las 3 ciudades por año y estas con un heigth definido por sus ventas. la leyenda seria de las ciudades.
    plt.bar(posicion, ventas, width=ancho, label= ciudad, color= colores.get(ciudad, 'gray') )

    # Ahora le pondremos a cada barra la cantidad de ventas segun cada ciudad recorriedo la posicion de cada uno y su ventas utilizamos zip para seleccionar 2 listas a la misma vez y unirlas luego convertimos todo en texto para que salga sin errores en el grafico y a la posicion le sumamos 50 y asi no se super ponen con la barra lo centramos y le damos un tamño a propiado  
    for xpos, valor in zip(posicion, ventas):
        plt.text(xpos, valor + 50, str(valor), ha='center', va='bottom', fontsize=9)


plt.xticks(position_x + ancho, tabla_pivotada.index.astype(str))
plt.title("Ventas Totales por Ciudad y Año")
plt.xlabel("Año")
plt.ylabel("Ventas")

plt.legend()
plt.tight_layout()
plt.show()

#                   OJO PARA QUE EL PROGRAMA CORRA PERFECTAMENTE TENEMOS QUE CERRAR LA VENTANA DEL GRAFICO YA QUE EL SHOW()HACE UNA PAUSA  



print("Todo listo para entrenar el modelo")


                        #                           N O R M A L I Z A C I O N 

#convertimos ventas_iudad_año en un array de numpy mediante values
ventas_array = ventas_ciudad_año["Ventas"].values

#para normalizar aplicamos la formula de min-max scaling
ventas_normalizada =(ventas_array - np.min(ventas_array))/ (np.max(ventas_array)-np.min(ventas_array))



#Agregamos a nuestro dataframe la columna de ventas normalzadas
ventas_ciudad_año["ventas_normalizada"]=ventas_normalizada

print(ventas_ciudad_año)

#convertimos los meses en numeros ya que los utilizaremos mas adelante

meses_map= {'Enero': 1, 'Febrero': 2}

datos_dataframe['meses_map']=datos_dataframe['Mes'].map(meses_map)

#Agrupar ahora por ciudad. año y meses las ventas
ventas_totales_mes= datos_dataframe.groupby(['Ciudad','Año','meses_map'])['Ventas'].sum().reset_index()
print(ventas_totales_mes)

#Normalizar las ventas pero en este caso por mes
ventas_array_mes= ventas_totales_mes['Ventas'].values

ventas_normalizada_mes = (ventas_array_mes - np.min(ventas_array_mes))/(np.max(ventas_array_mes) - np.min(ventas_array_mes))

ventas_totales_mes['ventas_normalizada_mes']= ventas_normalizada_mes
print(ventas_totales_mes)


                            # E N T R E N A M I E N T O     M O D E L O 


#Definimos  nuestras variables predictoras y objetivo

x = ventas_totales_mes[['Año', 'meses_map']] #predictora
print(x) #Hacemos print para visualizar el año y mes 


y = ventas_totales_mes['ventas_normalizada_mes'] #objetivo


#Dividimos para entrenar y probar nuestro modelo

#hacemos una variable de entramiento y de prueba de X y de Y luego le decimos que de prueba tendra el 20% el 80% restante sera de entrenamiento y por ultimo ramdon_state=42 nos da la misma division 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Creamos el modelo y lo entrena por medio de fit() segun los datos de entrenamiento x_train y y_train
modelo =LinearRegression()
modelo.fit(x_train, y_train)

#prediccion
y_pred= modelo.predict(x_test)

#Aplicamos el MSE que es el error cuadrado medio 

mse=mean_squared_error(y_test, y_pred)
print(f'El error medio cuadrado es de: {mse:.4f}') 


#creamos un dataframe para comparar valores, primero copiamos nuestro x_test hacemos una colunma llamada ventas reales y 
resultados = x_test.copy()
resultados["Ventas_real"] = y_test.values
resultados["Ventas_predicha"] = y_pred
print("Comparación entre ventas reales y predichas:")
print(resultados)

''' CONCLUSION:
El modelo de regresión lineal desarrollado fue capaz de predecir las ventas normalizadas mensuales en función del año y del mes con un error cuadrático medio bajo, lo cual indica una buena capacidad de generalización. Este enfoque permite al gerente identificar qué meses y años tienden a tener mejores resultados, facilitando la planificación estratégica de inventario y promociones estacionales.
'''























    



