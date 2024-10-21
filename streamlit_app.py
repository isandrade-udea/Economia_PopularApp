import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import prince
from sklearn.cluster import KMeans
from matplotlib.patches import Circle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import ListedColormap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from PIL import Image

#python -m pip install {package_name}

# CSS para ampliar la página y eliminar márgenes en blanco
st.markdown(
    """
    <style>
    .block-container {
        padding: 1rem; /* Ajuste fino */
    }
    .main {
        max-width: 100%;  /* Ancho completo */
        padding: 1;  
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title(":shopping_trolley: Economía Popular - Plaza Minorista Medellín")
st.write(
    "Este informe explora los patrones de uso de herramientas digitales en negocios locales, junto con un modelo predictivo para segmentar tipos de negocio basado en datos recogidos."
)

#datos
path_to_file = "https://raw.githubusercontent.com/isandrade-udea/LabIA/main/EconomiaPopluar/Caracterizacion%20Economia.csv"

df = pd.read_csv(path_to_file)

df = df.drop(['ID', 'Hora de inicio', 'Hora de finalización'], axis=1)

df.columns = ['tipo_negocio', #¿Qué tipo de negocio tiene?
              'tiempo_negocio', #¿Cuánto tiempo tiene el negocio?
              'numero_empleados',#¿Cuál es el número de empleados de su negocio?
              'tipo_contratacion',#¿Qué tipo de contratación establece con los empleados de su negocio?
              'constituido',# Si su negocio se encuentra constituido seleccione según le aplique
              'uso_herramientas_digitales', #¿En qué medida utiliza herramientas digitales para gestionar las operaciones diarias de su negocio?
              'herramientas_digitales',#Si usted ya utiliza herramientas tecnológicas para administrar su negocio, por favor informe ¿Cuáles?
              'uso_herramientas_gestion', #¿Con qué frecuencia usa herramientas de gestión para organizar el inventario, compras y ventas? Como Excel, Drive, Correo electrónico, otras aplicaciones como Siigo, TNS'
              'datos_clientes',#¿En qué medida protege los datos de sus clientes, proveedores, registros de venta, de compra, de inventarios?'
              'presencia_redes',#¿En que medida crees que tener una presencia activa en redes sociales permite la visibilidad de su negocio?'
              'consideracion_redes',#¿En qué medida considera que las Redes Sociales pueden contribuir al crecimiento de la base de clientes de su negocio?',
              'impacto_positivo_redes',#¿El uso de las Redes Sociales ha impactado de manera positiva las ventas de su negocio?',
              'tiene_redes',#'¿Su negocio tiene presencia en Redes Sociales? si su respuesta es afirmativa ¿Puede mencionar cuáles?
              'web',#¿Su negocio cuenta con sitio web o tienda virtual?
              'consideracion_pago_electronico',#¿En que medida considera que las soluciones de pago electrónico son seguras y confiables para su negocio?',
              'uso_pago_electronico',#¿En que medida ha implementado soluciones de pago electrónico o aplicaciones móviles para transacciones dé facturas a proveedores u otras cuentas?',
              'app_pago',#¿En que medida recibe a través de aplicaciones móviles pago electrónico de sus clientes?',
              'analisis_ventas',#        '¿En su negocio se analizan las ventas para tomar decisiones comerciales?',
              'comentarios']

st.subheader("Caracteristicas de los datos")

col1, col2 = st.columns(2)

with col1:
    st.write(f"El tamaño del dataset es: {df.shape[0]} filas y {df.shape[1]} columnas.")

    # Opciones de columnas para graficar
    opciones_columnas = [
        'tipo_negocio', 
        'tiempo_negocio', 
        'numero_empleados', 
        'tipo_contratacion', 
        'uso_herramientas_digitales', 
        'uso_herramientas_gestion', 
        'datos_clientes', 
        'presencia_redes', 
        'consideracion_redes', 
        'impacto_positivo_redes', 
        'consideracion_pago_electronico', 
        'uso_pago_electronico', 
        'app_pago', 
        'analisis_ventas'
    ]

    # Selección de columna con 'tipo_negocio' como predeterminado
    columna_seleccionada = st.selectbox(
        "Selecciona la columna para graficar:", 
        opciones_columnas, 
        index=opciones_columnas.index('tipo_negocio')
    )

    # Crear gráfico de barras
    fig, ax = plt.subplots()
    sns.countplot(data=df, y=columna_seleccionada, palette='pastel', order=df[columna_seleccionada].value_counts().index,ax=ax)

    ax.set_title(f'Distribución de {columna_seleccionada.replace("_", " ").capitalize()}')

    # Mostrar gráfico en Streamlit
    st.pyplot(fig)

with col2:
    # Opciones para los ejes
    opciones_categoricas = [
        'tipo_negocio', 'tiempo_negocio', 'tipo_contratacion', 'numero_empleados','web'
    ]

    opciones_numericas = [
        'uso_herramientas_digitales', 'uso_herramientas_gestion', 
        'datos_clientes', 'presencia_redes', 'consideracion_redes', 
        'impacto_positivo_redes', 'consideracion_pago_electronico', 
        'uso_pago_electronico', 'app_pago', 'analisis_ventas'
    ]

    # Selección de columnas para los ejes
    columna_x = st.selectbox("Selecciona la columna categórica para el eje X:", opciones_categoricas, index=opciones_columnas.index('tipo_negocio'))
    columna_y = st.selectbox("Selecciona la columna numérica para el eje Y:", opciones_numericas, index=opciones_columnas.index('uso_herramientas_digitales'))

    # Calcular la mediana para ordenar el boxplot
    medianas = df.groupby(columna_x, observed=True)[columna_y].median()

    # Crear el gráfico
    fig, ax = plt.subplots()
    sns.boxplot(x=columna_x, y=columna_y, data=df, ax=ax, order=medianas.index, palette='pastel')

    # Graficar las medianas como línea verde
    medianas.plot(style='o-', color='gray', linewidth=0.8, ax=ax)

    # Configuración del gráfico
    ax.set_ylabel(columna_y.replace('_', ' ').capitalize())
    ax.set_xlabel(columna_x.replace('_', ' ').capitalize())
    ax.tick_params(axis='x', rotation=90, labelsize=8)  # Rotar etiquetas del eje X
    plt.tight_layout()
    # Ajustar márgenes del gráfico
    plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.3)

    ax.set_title(f'Distribución de {columna_seleccionada.replace("_", " ").capitalize()}')

    # Mostrar gráfico en Streamlit
    st.pyplot(fig)

# Opciones de selección
opciones_redes = ['presencia_redes', 'consideracion_redes', 'impacto_positivo_redes']
opciones_pago = ['consideracion_pago_electronico', 'uso_pago_electronico', 'app_pago']

# Selector en Streamlit para elegir conjunto de variables
tipo_seleccion = st.radio(
    "Selecciona el grupo de variables:",
    ('Redes Sociales', 'Pagos Electrónicos')
)

# Seleccionar columnas basadas en la opción elegida
if tipo_seleccion == 'Redes Sociales':
    columnas_seleccionadas = opciones_redes
else:
    columnas_seleccionadas = opciones_pago

# Derretir (melt) el DataFrame según las columnas seleccionadas
df_melted = df.melt(value_vars=columnas_seleccionadas, var_name='tipo', value_name='respuesta')

# Crear gráfico con Seaborn
fig, ax = plt.subplots(figsize=(8, 4))
sns.countplot(data=df_melted, x='respuesta', hue='tipo', ax=ax, palette='pastel')

# Configurar ejes y leyenda
ax.set_xlabel('Respuesta')
ax.set_ylabel('Frecuencia')
ax.set_title('Frecuencia de Respuestas por Tipo')
ax.legend(title='Tipo')
ax.yaxis.set_tick_params(labelsize=7.5)  # Tamaño de fuente en eje Y

# Mostrar gráfico en Streamlit
st.pyplot(fig)

st.subheader("Clasificación")
#preprocesameinto de los datos

#reunir los menos frecuentes en una sola categoria
# Calcular las frecuencias relativas de cada tipo de negocio
frecuencias = df['tipo_negocio'].value_counts(normalize=True)

# Definir el umbral (en este caso, 1%)
umbral = 0.01

# Reemplazar las categorías con menos del 1% por 'Otros'
df['tipo_negocio'] = df['tipo_negocio'].apply(lambda x: 'poco_frecuentes' if frecuencias[x] < umbral else x)

# Definir las respuestas que deseas agrupar
respuestas_a_agrupar = ['no aplica', 'solo uno', 'Dueños ', 'solo es un empleado ', 'solo una ']

# Reemplazar las respuestas agrupadas con un nuevo valor, por ejemplo, 'No formalizado'
df['tipo_contratacion'] = df['tipo_contratacion'].apply(lambda x: 'dueños' if x in respuestas_a_agrupar else x)

# Crear una función para cuantificar la presencia de redes
def cuantificar_redes(row):
    if row == 'No tiene;':
        return 0
    else:
        # Contar cuántas redes están separadas por ';'
        return len(row.split(';')) - 1  # Se resta 1 ya que hay un ';' al final de cada valor

# Aplicar la función a la columna 'tiene redes'
df['cuantificacion_redes'] = df['tiene_redes'].apply(cuantificar_redes)

# Crear un diccionario para almacenar los codificadores y los mapeos
label_encoders = {}
mappings = {}

cat_cols1 = ['tipo_negocio', 'tiempo_negocio', 'numero_empleados', 
             'tipo_contratacion', 'web']

# Codificar cada columna categórica y almacenar el codificador y su mapeo
for col in cat_cols1:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Guardar el codificador
    mappings[col] = dict(zip(le.transform(le.classes_),le.classes_))  # Guardar el mapeo


#codificacion de variable categorica usando la frecuencia
cat_cols2 = ['constituido','herramientas_digitales']

for col in cat_cols2:
    df[col] = df[col].apply(lambda x: len(x.split(';'))-1)

# Eliminar las columnas 'tiene_redes' y 'comentarios'
df = df.drop(['tiene_redes', 'comentarios'], axis=1)

#dimensionalidad
scaler_standard = StandardScaler()
X_scaled = scaler_standard.fit_transform(df)

# Aplicar MCA
# Aplicar MCA con más componentes
mca = prince.MCA(n_components=2)  # Aumentar a 6 componentes
mca = mca.fit(df)

# Obtener los valores propios (Eigenvalues)
eigenvalues = mca.eigenvalues_

# Calcular la inercia explicada por cada componente
total_inertia = sum(eigenvalues)
explained_inertia = [eig / total_inertia for eig in eigenvalues]

# Obtener las coordenadas de las filas (observaciones)
X_mca = mca.row_coordinates(df)

#clusterizacion
k = 2
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(X_mca.values)

X_mca['cluster_kmeans'] = kmeans.labels_
df['cluster_kmeans'] = kmeans.labels_

def plot_correlation_circle(X_scaled, X_dim, axs, df_columns):
    ccircle = []
    eucl_dist = []

    # Calcular correlaciones y distancias euclidianas
    for i, j in enumerate(X_scaled.T):
        corr1 = np.corrcoef(j, X_dim[:, 0])[0, 1]  # Correlación con Dim-1
        corr2 = np.corrcoef(j, X_dim[:, 1])[0, 1]  # Correlación con Dim-2
        ccircle.append((corr1, corr2))
        eucl_dist.append(np.sqrt(corr1**2 + corr2**2))

    # Ordenar variables por magnitud de correlación
    corr_dim1 = sorted([(df_columns[i], abs(c[0])) for i, c in enumerate(ccircle)],
                       key=lambda x: x[1], reverse=True)
    corr_dim2 = sorted([(df_columns[i], abs(c[1])) for i, c in enumerate(ccircle)],
                       key=lambda x: x[1], reverse=True)

    # Imprimir variables mejor representadas en Dim-1 y Dim-2
    print("Variables mejor representadas en Dim-1:")
    for var, corr in corr_dim1:
        print(f"{var}: {corr:.2f}")

    print("\nVariables mejor representadas en Dim-2:")
    for var, corr in corr_dim2:
        print(f"{var}: {corr:.2f}")

    # Graficar flechas y texto en el círculo de correlación
    for i, j in enumerate(eucl_dist):
        arrow_col = plt.cm.cividis((eucl_dist[i] - np.array(eucl_dist).min()) /\
                                   (np.array(eucl_dist).max() - np.array(eucl_dist).min()))
        axs.arrow(0, 0,  # Flechas empiezan en el origen
                  ccircle[i][0],  # Correlación con Dim-1
                  ccircle[i][1],  # Correlación con Dim-2
                  lw=1,  # Ancho de la línea
                  length_includes_head=True,
                  color=arrow_col,
                  fc=arrow_col,
                  head_width=0.05,
                  head_length=0.05)
        axs.text(ccircle[i][0], ccircle[i][1], df_columns[i], fontsize=10)

    # Dibujar el círculo unitario
    circle = Circle((0, 0), 1, facecolor='none', edgecolor='k', linewidth=1, alpha=0.5)
    axs.add_patch(circle)
    # Etiquetar ejes con inercia explicada
    axs.set_xlabel("Dim-1 (%s%%)" % str(explained_inertia[0] * 100)[:4].lstrip("0."),fontsize=27)
    axs.set_ylabel("Dim-2 (%s%%)" % str(explained_inertia[1] * 100)[:4].lstrip("0."),fontsize=27)
    axs.tick_params(axis='both', which='major', labelsize=18)

# Función para graficar la frontera de decisión
def plot_decision_boundary(model, X, ax):
    
    # Crear un meshgrid
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    # Predecir cada punto del meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Dibujar la frontera usando contourf
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='Set3')
    # Etiquetar ejes con inercia explicada
    ax.set_xlabel("Dim-1 (%s%%)" % str(explained_inertia[0] * 100)[:4].lstrip("0."),fontsize=27)
    ax.set_ylabel("Dim-2 (%s%%)" % str(explained_inertia[1] * 100)[:4].lstrip("0."),fontsize=27)
    ax.tick_params(axis='both', which='major', labelsize=18)

figure, axes = plt.subplots(1,2, figsize=(25, 12))
# Graficar la frontera de decisión y los puntos
plot_decision_boundary(kmeans, X_mca, axes[0])
sns.scatterplot(data=X_mca, x=0, y=1, hue='cluster_kmeans', palette="Set2", ax=axes[0], s=50)
axes[0].set_title('Frontera de Decisión y Clustering K-Means', fontsize=30)
axes[0].legend(title='Cluster', fontsize=22, title_fontsize=26)
# Definir los límites de los ejes
axes[0].set_xlim(X_mca[0].min() - 0.1, X_mca[0].max() + 0.1)  # Ajusta los límites según tus datos
axes[0].set_ylim(X_mca[1].min() - 0.1, X_mca[1].max() + 0.1)  # Ajusta los límites según tus datos


plot_correlation_circle(X_scaled, X_mca.values, axes[1], df.columns)
axes[1].set_title('Círculo de Correlación',fontsize=30)
plt.tight_layout()
# Mostrar gráfico en Streamlit
st.pyplot(figure)


# Características disponibles para seleccionar
caracteristicas = ['tipo_negocio', 'tiempo_negocio', 'numero_empleados', 
             'tipo_contratacion']  # Añade las características que deseas

# Crear un menú desplegable en Streamlit
selected_feature = st.selectbox('Selecciona una característica:', caracteristicas,opciones_columnas.index('tipo_contratacion'))

# Agrupar los datos por cluster y la característica seleccionada
cluster_business_counts = df.groupby(['cluster_kmeans', selected_feature])[selected_feature].count().unstack()

# Crear el gráfico de barras para cada cluster
fig, axes = plt.subplots(nrows=1, ncols=k, figsize=(6, 4))  # Ajusta el figsize según sea necesario

for cluster_num in range(k):
    ax = axes[cluster_num]
    cluster_counts = cluster_business_counts.loc[cluster_num]

    # Definir colores pastel manualmente
    pastel_colors = [
    '#FFB6C1',  # Light Pink
    '#87CEFA',  # Light Sky Blue
    '#98FB98',  # Pale Green
    '#FFD700',  # Gold
    '#FF69B4',  # Hot Pink
    '#FFDAC1',  # Peach
    '#E0BBE4',  # Lavender
    '#B5EAD7',  # Mint
    '#C7CEEA',  # Periwinkle
    '#FF9AA2',  # Light Coral
    '#FFB7B2',  # Salmon Pink
    '#FAD02E',  # Sun Yellow
    '#B3E5FC',  # Baby Blue
    '#A1C298',  # Light Olive
    '#F8B195',  # Coral
    '#F67280',  # Watermelon
    ]  

    # Reemplazar los números codificados por los nombres reales usando el mapeo
    cluster_counts.index = cluster_counts.index.map(mappings[selected_feature])
    
    cluster_counts.plot(kind='bar', ax=ax, title=f'Cluster {cluster_num}', color=pastel_colors[:len(cluster_counts)])
    ax.set_xlabel(selected_feature)  # Usa la característica seleccionada como etiqueta del eje X
    ax.set_ylabel('Cantidad')
    ax.tick_params(axis='x', rotation=90, labelsize=7)  # Rote las etiquetas del eje X
plt.tight_layout()

# Mostrar gráfico en Streamlit
st.pyplot(fig)

st.subheader("Modelo predictivo")



df_selec = df[[ 'numero_empleados', 
               'tipo_contratacion', 
               'constituido', 
               'herramientas_digitales',
               'presencia_redes',
               'cluster_kmeans']]

# Separar características y etiquetas
X = df_selec.drop('cluster_kmeans', axis=1)  # Reemplaza 'target' con tu columna objetivo
y = df_selec['cluster_kmeans']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear el modelo XGBoost con parámetros específicos
model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    objective='binary:logistic',
    booster='gbtree',
    n_jobs=-1,
    tree_method='auto'
)

# Entrenar el modelo
model.fit(X_train, y_train)

# Predecir en los datos de prueba
y_pred = model.predict(X_test)

# Calcular el F1-Score
f1 = f1_score(y_test, y_pred)*100

# Guardar el modelo en un archivo
model.save_model('xgboost_model.json')

#model = xgb.XGBClassifier()
#model.load_model('xgboost_model.json')

st.markdown(f"""
**Modelo XGBoost - Extreme Gradient Boosting**: Este modelo utiliza un método avanzado que ayuda a tomar decisiones. 
Fue entrenado con varios datos para que pueda hacer predicciones con precisión de: {f1:.2f}\% (F1-Score) .

¿Cómo usarlo?
- Ingresa los valores en el formulario que ves abajo.
- Haz clic en **Predecir** y el modelo te dará un resultado basado en los datos que ingresaste.
""")

st.write(" Seleccione las opciones aplicables:")

col1, col2, col3 = st.columns(3)

with col1:

    # Input en Streamlit: Mostrar las descripciones en el selectbox
    empleados = st.selectbox(
        "**Número de empleados**", 
        options=['1-2','3-5','6-10','10 o mas'],
        index=0
    )

    empleados_cod = {'1-2': 0, '3-5': 2, '6-10': 3, '11 o más': 1}[empleados]

    # Opciones para el selectbox
    opciones = [
        'Contrato de prestación de servicios',
        'Contrato de trabajo a término fijo',
        'Contrato de trabajo a término indefinido',
        'Contrato por obra o labor',
        'Contrato temporal, ocasional o accidental',
        'Contrato verbal',
        'Dias',
        'Indefinido',
        'Independiente',
        'No es permanente',
        'No saben',
        'dueños'
    ]

    # Asegurarse de que la opción por defecto existe en la lista
    opcion_default = 'Contrato verbal'

    # Crear el selectbox en Streamlit
    contratacion = st.selectbox(
        "**Tipo de contratación**",
        options=opciones,
        index=opciones.index(opcion_default)
    )

    contratacion_cod = {
        'Contrato de prestación de servicios':0,
        'Contrato de trabajo a término fijo':1,
        'Contrato de trabajo a término indefinido':2,
        'Contrato por obra o labor':3,
        'Contrato temporal, ocasional o accidental':4,
        'Contrato verbal':5,
        'Dias':6,
        'Indefinido':7,
        'Independiente':8,
        'No es permanente':9,
        'No saben':10,
        'dueños':11}[contratacion]


    uniq_constituidos = {'RUT':1, 'Registro Mercantil':1,
                        'Cámara de Comercio':1, 
                        'Ninguna de las anteriores':0}

    # Crear un diccionario para almacenar el estado de cada checkbox
    selecciones = {}

    st.write("**Documentos legales**")
    # Crear un checkbox por cada opción del diccionario
    for opcion in uniq_constituidos.keys():
        selecciones[opcion] = st.checkbox(opcion)

    # Sumar los valores numéricos de las opciones seleccionadas
    valor_total_constituido = sum(
        uniq_constituidos[opcion] for opcion, seleccionado in selecciones.items() if seleccionado
    )

with col2:
    uniq_herramientas_digitales = {'Balanza digital':1, 'Cajas':1,
                                'Camaras':1, 'Celular':1,
                                'Computador':1,	'Portátil':1,
                                'QR de pagos':1, 'Registradora':1,
                                'Tableta':1, 'Televisor':1,
                                'Ninguna':0}

    # Crear un diccionario para almacenar el estado de cada checkbox
    selecciones = {}

    st.write("**Herramientas digitales**")
    # Crear un checkbox por cada opción del diccionario
    for opcion in uniq_herramientas_digitales.keys():
        selecciones[opcion] = st.checkbox(opcion)

    # Sumar los valores numéricos de las opciones seleccionadas
    valor_total_herramientas_digitales = sum(
        uniq_herramientas_digitales[opcion] for opcion, seleccionado in selecciones.items() if seleccionado
    )

with col3:
    uniq_tiene_redes={'Correo electrónico':1, 
                    'FB Messenger':1, 
                    'Facebook':1, 'Instagram':1,
                    'No tiene':0, 'Telegram':1, 
                    'TikTok':1, 'Twitter (X)':1, 'WhatsApp':1, 'YouTube':1}

    # Crear un diccionario para almacenar el estado de cada checkbox
    selecciones = {}

    st.write("**Redes sociales**")
    # Crear un checkbox por cada opción del diccionario
    for opcion in uniq_tiene_redes.keys():
        selecciones[opcion] = st.checkbox(opcion)

    # Sumar los valores numéricos de las opciones seleccionadas
    valor_total_tiene_redes = sum(
        uniq_tiene_redes[opcion] for opcion, seleccionado in selecciones.items() if seleccionado
    )

# Botón para predecir
if st.button("Predecir"):
    # Crear un DataFrame con los datos de entrada
    df_input = pd.DataFrame({
        'numero_empleados': [empleados_cod],
        'tipo_contratacion': [contratacion_cod],
        'constituido': [valor_total_constituido],
        'herramientas_digitales': [valor_total_herramientas_digitales],
        'presencia_redes': [valor_total_tiene_redes]
    })

    # Realizar la predicción
    y_pred = model.predict(df_input)[0]

    # Mostrar el resultado basado en la predicción
    resultado = "Cluster 1: Baja adopción de digitalización" if y_pred == 1 else "Cluster 0: Alta adopción de digitalización"

    # Definir el color basado en y_pred
    color = "#ffcccc" if y_pred == 1 else "#ccffcc"  # Rojo pastel para 1, Verde pastel para 0


    # Mostrar el resultado en una caja de color personalizado
    st.markdown(f"""
    <div style="background-color: {color}; padding: 10px; border-radius: 5px;">
        <strong>{resultado}</strong>
    </div>
    """, unsafe_allow_html=True)


# URL de la imagen en GitHub
image_url = "https://raw.githubusercontent.com/isandrade-udea/blank-app-2/main/napkin-selection.png"

# Mostrar la imagen desde la URL
st.image(image_url, caption='sugerencias de estrategias', use_column_width=True)

st.markdown("<h2>Referencias</h2>", unsafe_allow_html=True)  # Usando HTML para subtítulo
referencias_html = """
<ol>
    <li> Notebook con EDA.  <a href="https://colab.research.google.com/drive/1Wx81uSH_uqySo-xhVqq9rAfl4xr1xdxc?usp=sharing">link</a></li>
    <li> Github de la app. <a href="https://github.com/isandrade-udea/blank-app-2/tree/main">link</a> .</li>
    <li> Adopción Tecnológica del Comercio Electrónico en la Economía Popular. Recuperado el 21 de octubre de 2024: <a href="https://observatorioecommerce.mintic.gov.co/797/articles-334321_recurso_1.pdf">link</a> .</li>
</ol>
"""

st.markdown(referencias_html, unsafe_allow_html=True)
