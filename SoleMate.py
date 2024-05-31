from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import streamlit as st
from openai import OpenAI
import pandas as pd

def classify_fruit(img):
    np.set_printoptions(suppress=True)

    model = load_model("keras_model.h5", compile=False)

    with open("labels.txt", "r") as f:
        class_names = f.readlines()
        class_names = [c.strip() for c in class_names]  # Remove any surrounding whitespace

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = img.convert("RGB")

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

def generate_recipe(label):
    client = OpenAI(api_key="")

    response = client.Completion.create(
        model="gpt-3.5-turbo-instruct",
        temperature=0.5,
        prompt= f"Sos un asistente experto en modelos de zapatillas Nike Air Forces, Air Jordans, Air Maxes, Cleats y Dunks. A partir del modelo {label} necesito que generes una breve descripcion del producto y sus principales caracteristicas: comodidad, talles, colores, para que se utiliza y perfil del usuario que compraria este producto.",
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].text

st.set_page_config(layout='wide')


st.title('SoleMate')
st.subheader('Encontrá tu par perfecto', divider='red')

video_file = open('NikeComercial.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes, start_time=0)

input_img = st.file_uploader("Ingresá la foto del modelo que buscas y conocé más con un solo click", type=['jpg', 'png', 'jpeg'])

def classify_fruit(image_file):
    # Aquí deberías agregar la lógica para clasificar la imagen
    # Por ahora, simplemente devolvemos un valor de prueba
    return "0 Air Forces", 0.85

if input_img is not None:
    if st.button("Classify"):
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.info("Your uploaded Image")
            st.image(input_img, use_column_width=True)

        with col2:
            st.info("Your Result")
            image_file = Image.open(input_img)
            label, confidence_score = classify_fruit(image_file)
            if label == "0 Air Forces":
                st.success("Tu par perfecto es Air Forces.")
            elif label == "1 Air Jordans":
                st.success("Tu par perfecto es Air Jordans.")
            elif label == "2 Air Maxes":
                st.success("Tu par perfecto es Air Maxes.")
            elif label == "3 Cleats":
                st.success("Tu par perfecto es Cleats.")
            elif label == "4 Dunks":
                st.success("Tu par perfecto es Dunks.")
            else:
                st.error("No encontramos ningún match para vos 😢 Cargá otra foto para que encontremos tu par ideal! .")

        with col3:
            st.info("Recommendations")
            if label in ["0 Air Forces", "1 Air Jordans", "2 Air Maxes", "3 Cleats", "4 Dunks"]:
                st.write(f"Modelo identificado: {label}")
                st.write(f"Confianza: {confidence_score * 100:.2f}%")
                st.write("Recomendaciones de compra:")
                st.write("- Producto 1")
                st.write("- Producto 2")
                st.write("- Producto 3")

with st.sidebar:
    st.subheader('Regístrate para recibir ofertas exclusivas')
    email = st.text_input('Correo electrónico')
    if st.button('Registrarse'):
        st.success('¡Gracias por registrarte!')

st.write("HOLA")

if st.button("Abrir formulario"):
    with st.container():
        st.write("## Formulario de contacto")
        name = st.text_input("Nombre")
        email = st.text_input("Correo electrónico")
        gender = st.selectbox("Género", ["Masculino", "Femenino", "Otro"])
        interests = st.multiselect("Intereses", ["Deportes", "Tecnología", "Arte", "Viajes"])
        option = st.radio("¿Cómo nos encontraste?", ["Redes sociales", "Búsqueda en línea", "Referencia de un amigo"])
        comment = st.text_area("Comentario", "")
        if st.button("Enviar"):
            st.success("¡Gracias por enviar el formulario!")
            st.write(
                """
                <script>
                setTimeout(function() {
                    const container = document.querySelector('.element-container');
                    container.style.display = 'none';
                }, 3000);
                </script>
                """
            )

# Cargar los datos
file_path = 'Ubicaciones tiendas Nike.xlsx'
df = pd.read_excel(file_path)

# Crear listas de opciones únicas para los filtros
provincias = df['Provincia'].unique()


# Crear la interfaz de Streamlit
st.title("Tiendas de Nike en Argentina")

# Filtros de provincia y zona
provincia_seleccionada = st.selectbox("Selecciona una provincia", ["Todas"] + list(provincias))

# Filtrar el dataframe según la selección del usuario
if provincia_seleccionada != "Todas":
    df = df[df['Provincia'] == provincia_seleccionada]


# Crear los marcadores para el mapa en formato HTML/JavaScript
map_center = [-38.4161, -63.6167]
markers = ""
for idx, row in df.iterrows():
    markers += f"""
    L.marker([{row['Latitud']}, {row['Longitud']}]).addTo(map)
        .bindPopup("<b>{row['Nombre']}</b><br>Dirección: {row['Dirección']}<br>Provincia: {row['Provincia']}");
    """

map_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Mapa de Tiendas Nike</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <style>
        #map {{
            height: 600px;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        var map = L.map('map').setView({map_center}, 5);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }}).addTo(map);
        {markers}
    </script>
</body>
</html>
"""

# Mostrar el mapa en Streamlit
st.components.v1.html(map_html, height=600)