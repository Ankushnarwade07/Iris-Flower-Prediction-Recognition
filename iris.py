import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main_page():
    st.title('Iris Flower Prediction & Recognition')
    image = Image.open(r"C:\Users\Dell\Downloads\51518iris img1.png")
    st.image(image)
    st.sidebar.markdown("INFORMATION")
    st.subheader('What is Iris-Setosa?')
    st.markdown('Iris setosa is characterized by its unique appearance, with smaller and more delicate flowers compared to the other two species in the dataset. The petals of Iris setosa are usually white or light pink, and the plant is known for its resilience and ability to thrive in various environmental conditions.')

    st.subheader('What is Iris-Versicolour?')
    st.markdown('Iris versicolor is characterized by its distinct features, including the size and color of its flowers sepals and petals. It has medium-sized flowers that are typically blue to purple in color. The leaves are also distinctive, with a sword-like shape.')

    st.subheader('What is Iris-Virginica?')
    st.markdown('Iris virginica is characterized by certain features that distinguish it from the other two species in the dataset. It typically has larger flowers compared to Iris setosa and may exhibit a range of colors, including shades of blue, purple, and white. The leaves are long and lance-shaped.')

    st.subheader('Dataset Links')
    st.markdown('1. https://www.kaggle.com/datasets/jeffheaton/iris-computer-vision')
    st.markdown('2. https://www.kaggle.com/datasets/uciml/iris')

    st.subheader('GitHub')
    st.markdown('https://github.com/Ankushnarwade07')



def page1():
    st.title('Prediction')
    st.sidebar.markdown("PREDICTION")
    import warnings
    warnings.simplefilter('ignore', UserWarning)
    data = pd.read_csv(r"D:\DATASETS\IRIS.csv")
    train, test = train_test_split(data, test_size=0.25)
    train_X = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    train_y = train.species
    test_X = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    test_y = test.species
    knn = KNeighborsClassifier()
    knn = knn.fit(train_X, train_y)
    y_pred = knn.predict(test_X)

    sepal_length = st.number_input('Sepal Length',min_value=1, max_value=10, value=5, step=1)
    st.write('The Sepal Length is ', sepal_length)

    sepal_width = st.number_input('Sepal Width',min_value=1, max_value=10, value=5, step=1)
    st.write('The Sepal Width is ', sepal_width)

    petal_length = st.number_input('Petal Length',min_value=1, max_value=10, value=5, step=1)
    st.write('The Petal Length is ', petal_length)

    petal_width = st.number_input('Petal Width',min_value=1, max_value=10, value=5, step=1)
    st.write('The Petal Width is ', petal_width)

    x_new1 = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    if st.button("Predict"):
        prediction1 = knn.predict(x_new1)
        st.subheader("Prediction: {}".format(prediction1))



def page2():
    try:
        st.title('Recognition')
        st.sidebar.markdown("RECOGNITION")
        model = load_model(r"D:\DATASETS\Iris Dataset\keras_model.h5")
        class_names = open(r"D:\DATASETS\Iris Dataset\labels.txt", "r").readlines()
        image_file = st.file_uploader("Upload Image",type=['jpg','png'])
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open(image_file).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        st.image(image)
        st.markdown('CLASS')
        st.subheader(class_name)
        st.markdown('CONFIDENCE')
        st.subheader(confidence_score)
    except AttributeError:
        pass


page_names_to_funcs = {
    "INFORMATION": main_page,
    "PREDICTION": page1,
    "RECOGNITION": page2,
}
selected_page = st.sidebar.selectbox("Actions", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
