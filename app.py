import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
from predict import load_model, predict
from config import COLOR_ENCODING


# Кешируем загрузку модели
@st.cache_resource
def get_model():
    return load_model()


# Настройки страницы
st.set_page_config(
    page_title="Сегментация построек",
    layout="wide"
)

st.title("🛰️ Сегментация построек на спутниковых снимках")

# Загружаем модель
model = get_model()

# Загрузка изображения
uploaded_file = st.file_uploader("Загрузите спутниковое изображение", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Чтение изображения в память
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Отображение оригинального изображения
    st.image(image, caption="Исходное изображение", use_column_width=True)

    # Обработка
    with st.spinner("Анализирую повреждения..."):
        mask = predict(image, model)

        # Результаты
        st.subheader("Результаты сегментации")
        st.image(mask, use_column_width=True)

        # Анализ площади по классам
        st.subheader("Статистика площадей")

        # Создаем словарь для хранения результатов
        class_areas = {}
        total_pixels = mask.shape[0] * mask.shape[1]

        # Вычисляем площадь для каждого класса
        for class_name, color in COLOR_ENCODING.items():
            # Создаем маску для текущего класса
            class_mask = np.all(mask == np.array(color), axis=-1)
            area_pixels = np.sum(class_mask)
            area_percent = (area_pixels / total_pixels) * 100

            # Сохраняем результаты
            class_areas[class_name] = {
                'pixels': area_pixels,
                'percent': area_percent,
                'color': color
            }

        # Выводим результаты в виде таблицы
        st.write("### Площади по классам:")
        for class_name, data in class_areas.items():
            if class_name != 'unlabeled':  # Исключаем фоновый класс
                col1, col2, col3 = st.columns([1, 2, 3])
                with col1:
                    # Показываем цвет класса
                    st.markdown(
                        f"<div style='width:20px; height:20px; background-color:rgb{data['color']};'></div>",
                        unsafe_allow_html=True
                    )
                with col2:
                    st.write(f"**{class_name.replace('-', ' ').title()}**")
                with col3:
                    st.progress(data['percent'] / 100)
                    st.write(f"{data['pixels']:,} px ({data['percent']:.2f}%)")

        # Детальный анализ зданий
        st.write("### Сводка по зданиям:")
        building_classes = [k for k in class_areas.keys() if 'building' in k]
        total_building_area = sum(class_areas[c]['pixels'] for c in building_classes)

        if total_building_area > 0:
            for class_name in building_classes:
                percent_of_buildings = (class_areas[class_name]['pixels'] / total_building_area) * 100
                st.write(
                    f"- {class_name.replace('-', ' ').title()}: "
                    f"{class_areas[class_name]['pixels']:,} px "
                    f"({percent_of_buildings:.1f}% от всех зданий)"
                )

    # Кнопка скачивания
    img_bytes = cv2.imencode('.png', cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))[1].tobytes()
    st.download_button(
        label="⬇️ Скачать маску сегментации",
        data=img_bytes,
        file_name="damage_segmentation.png",
        mime="image/png"
    )