import cv2
import numpy as np
import gradio as gr
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
import base64

class ImageProcessor:
    def __init__(self):
        self.image_folder = "images"
        self.setup_image_folder()
        
    def setup_image_folder(self):
        """Crear carpeta de imágenes si no existe"""
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)
    
    def get_image_list(self):
        """Obtener lista de imágenes disponibles"""
        if not os.path.exists(self.image_folder):
            return []
        
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        images = []
        for file in os.listdir(self.image_folder):
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                images.append(file)
        return images
    
    def load_image(self, image_name):
        """Cargar imagen desde la carpeta"""
        if not image_name:
            return None
        
        image_path = os.path.join(self.image_folder, image_name)
        if os.path.exists(image_path):
            return cv2.imread(image_path)
        return None
    
    # 1. CARGA Y VISUALIZACIÓN DE IMÁGENES
    def display_image(self, image_name):
        """Mostrar imagen original"""
        img = self.load_image(image_name)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img_rgb
        return None
    
    # 2. OPERACIONES CON CANALES DE COLOR
    def convert_color_space(self, image_name, color_space):
        """Convertir entre espacios de color"""
        img = self.load_image(image_name)
        if img is None:
            return None
        
        conversions = {
            'BGR': img,
            'RGB': cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            'HSV': cv2.cvtColor(img, cv2.COLOR_BGR2HSV),
            'LAB': cv2.cvtColor(img, cv2.COLOR_BGR2LAB),
            'GRAYSCALE': cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        }
        
        result = conversions.get(color_space, img)
        
        # Convertir a RGB para visualización
        if color_space == 'BGR':
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        elif color_space == 'GRAYSCALE':
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        elif color_space in ['HSV', 'LAB']:
            # Para HSV y LAB, normalizar para mejor visualización
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        return result
    
    def separate_channels(self, image_name, color_space, channel):
        """Separar canales individuales"""
        img = self.load_image(image_name)
        if img is None:
            return None
        
        if color_space == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            channels = cv2.split(img)
            channel_names = ['R', 'G', 'B']
        elif color_space == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            channels = cv2.split(img)
            channel_names = ['H', 'S', 'V']
        elif color_space == 'LAB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            channels = cv2.split(img)
            channel_names = ['L', 'A', 'B']
        else:  # BGR
            channels = cv2.split(img)
            channel_names = ['B', 'G', 'R']
        
        channel_idx = channel_names.index(channel)
        result = channels[channel_idx]
        
        return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    
    def mix_channels(self, image_name, r_channel, g_channel, b_channel):
        """Mezclar diferentes canales"""
        img = self.load_image(image_name)
        if img is None:
            return None
        
        # Convertir a diferentes espacios de color
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Separar canales
        r_rgb, g_rgb, b_rgb = cv2.split(img_rgb)
        h, s, v = cv2.split(img_hsv)
        l, a, b = cv2.split(img_lab)
        
        channel_dict = {
            'R': r_rgb, 'G': g_rgb, 'B': b_rgb,
            'H': h, 'S': s, 'V': v,
            'L': l, 'A': a, 'B_lab': b
        }
        
        # Crear nueva imagen mezclando canales
        new_r = channel_dict.get(r_channel, r_rgb)
        new_g = channel_dict.get(g_channel, g_rgb)
        new_b = channel_dict.get(b_channel, b_rgb)
        
        result = cv2.merge([new_r, new_g, new_b])
        return result
    
    # 3. CORRECCIÓN DE IMÁGENES
    def transform_image(self, image_name, operation, angle=0, scale=1.0):
        """Transformaciones geométricas"""
        img = self.load_image(image_name)
        if img is None:
            return None
        
        h, w = img.shape[:2]
        
        if operation == 'rotate':
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            result = cv2.warpAffine(img, M, (w, h))
        elif operation == 'flip_horizontal':
            result = cv2.flip(img, 1)
        elif operation == 'flip_vertical':
            result = cv2.flip(img, 0)
        elif operation == 'resize':
            new_w = int(w * scale)
            new_h = int(h * scale)
            result = cv2.resize(img, (new_w, new_h))
        else:
            result = img
        
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    def adjust_brightness_contrast(self, image_name, brightness=0, contrast=1.0):
        """Ajustar brillo y contraste"""
        img = self.load_image(image_name)
        if img is None:
            return None
        
        result = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    def gamma_correction(self, image_name, gamma=1.0):
        """Corrección gamma"""
        img = self.load_image(image_name)
        if img is None:
            return None
        
        # Crear tabla de lookup para gamma
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        
        result = cv2.LUT(img, table)
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    def histogram_equalization(self, image_name, equalization_type='global'):
        """Ecualización de histograma"""
        img = self.load_image(image_name)
        if img is None:
            return None
        
        if equalization_type == 'global':
            # Convertir a escala de grises para ecualización global
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            result = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        else:  # local (CLAHE)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            equalized = clahe.apply(gray)
            result = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    # 4. MEJORAMIENTO DE IMAGEN
    def apply_filter(self, image_name, filter_type, kernel_size=5, sigma=1.0):
        """Aplicar filtros de mejoramiento"""
        img = self.load_image(image_name)
        if img is None:
            return None
        
        if filter_type == 'mean':
            result = cv2.blur(img, (kernel_size, kernel_size))
        elif filter_type == 'gaussian':
            result = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
        elif filter_type == 'bilateral':
            result = cv2.bilateralFilter(img, kernel_size, 75, 75)
        elif filter_type == 'median':
            result = cv2.medianBlur(img, kernel_size)
        elif filter_type == 'box':
            result = cv2.boxFilter(img, -1, (kernel_size, kernel_size))
        elif filter_type == 'laplacian':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            result = cv2.convertScaleAbs(laplacian)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif filter_type == 'sobel_x':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
            result = cv2.convertScaleAbs(sobel)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif filter_type == 'sobel_y':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
            result = cv2.convertScaleAbs(sobel)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif filter_type == 'canny':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif filter_type == 'sharpen':
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            result = cv2.filter2D(img, -1, kernel)
        else:
            result = img
        
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    # 5. APLICACIÓN DE MÁSCARAS Y UMBRALES
    def apply_threshold(self, image_name, threshold_type='binary', threshold_value=127):
        """Aplicar umbrales"""
        img = self.load_image(image_name)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if threshold_type == 'binary':
            _, result = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        elif threshold_type == 'binary_inv':
            _, result = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        elif threshold_type == 'adaptive_mean':
            result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        elif threshold_type == 'adaptive_gaussian':
            result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        elif threshold_type == 'otsu':
            _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            result = gray
        
        return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    
    def apply_bitwise_operation(self, image_name, operation='and'):
        """Aplicar operaciones bitwise"""
        img = self.load_image(image_name)
        if img is None:
            return None
        
        # Crear una máscara simple para demostrar
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        h, w = img.shape[:2]
        cv2.circle(mask, (w//2, h//2), min(w, h)//4, 255, -1)
        
        if operation == 'and':
            result = cv2.bitwise_and(img, img, mask=mask)
        elif operation == 'or':
            result = cv2.bitwise_or(img, img, mask=mask)
        elif operation == 'not':
            result = cv2.bitwise_not(img)
        elif operation == 'xor':
            result = cv2.bitwise_xor(img, img, mask=mask)
        else:
            result = img
        
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    # 6. ELIMINACIÓN DE FONDO
    def remove_background_color(self, image_name, method='hsv', background_color='white'):
        """Eliminar fondo por color"""
        img = self.load_image(image_name)
        if img is None:
            return None
        
        if method == 'hsv':
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            if background_color == 'white':
                # Rango para blanco en HSV
                lower = np.array([0, 0, 200])
                upper = np.array([180, 30, 255])
            elif background_color == 'green':
                # Rango para verde en HSV
                lower = np.array([35, 50, 50])
                upper = np.array([85, 255, 255])
            else:
                lower = np.array([0, 0, 200])
                upper = np.array([180, 30, 255])
            
            mask = cv2.inRange(hsv, lower, upper)
            
        elif method == 'lab':
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            if background_color == 'white':
                # Rango para blanco en LAB
                lower = np.array([200, 0, 0])
                upper = np.array([255, 255, 255])
            else:
                lower = np.array([200, 0, 0])
                upper = np.array([255, 255, 255])
            
            mask = cv2.inRange(lab, lower, upper)
        
        # Invertir máscara para mantener el objeto
        mask_inv = cv2.bitwise_not(mask)
        
        # Aplicar máscara
        result = cv2.bitwise_and(img, img, mask=mask_inv)
        
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    # 7. CAMBIAR FONDO DE LA IMAGEN
    def change_background(self, image_name, new_background='white'):
        """Cambiar fondo de la imagen"""
        img = self.load_image(image_name)
        if img is None:
            return None
        
        # Crear máscara simple basada en el fondo blanco
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 200])
        upper = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower, upper)
        mask_inv = cv2.bitwise_not(mask)
        
        # Crear nuevo fondo
        if new_background == 'white':
            background = np.full(img.shape, (255, 255, 255), dtype=np.uint8)
        elif new_background == 'black':
            background = np.full(img.shape, (0, 0, 0), dtype=np.uint8)
        elif new_background == 'green':
            background = np.full(img.shape, (0, 255, 0), dtype=np.uint8)
        else:
            background = np.full(img.shape, (255, 255, 255), dtype=np.uint8)
        
        # Extraer objeto
        object_part = cv2.bitwise_and(img, img, mask=mask_inv)
        
        # Extraer fondo
        background_part = cv2.bitwise_and(background, background, mask=mask)
        
        # Combinar
        result = cv2.add(object_part, background_part)
        
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    # 8. APILADO DE IMÁGENES
    def stack_images(self, image_name, operations=None):
        """Apilar imágenes para comparación"""
        img = self.load_image(image_name)
        if img is None:
            return None
        
        # Imagen original
        original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Aplicar algunas operaciones para comparar
        gaussian = cv2.GaussianBlur(img, (15, 15), 0)
        gaussian = cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Redimensionar todas las imágenes al mismo tamaño
        h, w = original.shape[:2]
        gaussian = cv2.resize(gaussian, (w, h))
        gray = cv2.resize(gray, (w, h))
        edges = cv2.resize(edges, (w, h))
        
        # Apilar horizontalmente
        top_row = np.hstack([original, gaussian])
        bottom_row = np.hstack([gray, edges])
        
        # Apilar verticalmente
        result = np.vstack([top_row, bottom_row])
        
        return result

# Crear instancia del procesador
processor = ImageProcessor()

# Función para manejar la carga de imágenes
def update_image_display(image_name):
    return processor.display_image(image_name)

# Funciones para cada operación
def convert_color_wrapper(image_name, color_space):
    return processor.convert_color_space(image_name, color_space)

def separate_channels_wrapper(image_name, color_space, channel):
    return processor.separate_channels(image_name, color_space, channel)

def mix_channels_wrapper(image_name, r_channel, g_channel, b_channel):
    return processor.mix_channels(image_name, r_channel, g_channel, b_channel)

def transform_wrapper(image_name, operation, angle, scale):
    return processor.transform_image(image_name, operation, angle, scale)

def brightness_contrast_wrapper(image_name, brightness, contrast):
    return processor.adjust_brightness_contrast(image_name, brightness, contrast)

def gamma_wrapper(image_name, gamma):
    return processor.gamma_correction(image_name, gamma)

def histogram_wrapper(image_name, eq_type):
    return processor.histogram_equalization(image_name, eq_type)

def filter_wrapper(image_name, filter_type, kernel_size, sigma):
    return processor.apply_filter(image_name, filter_type, kernel_size, sigma)

def threshold_wrapper(image_name, threshold_type, threshold_value):
    return processor.apply_threshold(image_name, threshold_type, threshold_value)

def bitwise_wrapper(image_name, operation):
    return processor.apply_bitwise_operation(image_name, operation)

def remove_bg_wrapper(image_name, method, bg_color):
    return processor.remove_background_color(image_name, method, bg_color)

def change_bg_wrapper(image_name, new_bg):
    return processor.change_background(image_name, new_bg)

def stack_wrapper(image_name):
    return processor.stack_images(image_name)

# Crear interfaz Gradio
def create_interface():
    with gr.Blocks(title="Procesador de Imágenes OpenCV", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🖼️ Aplicación de Procesamiento de Imágenes")
        gr.Markdown("**Desarrollado con OpenCV y Gradio**")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📁 Selección de Imagen")
                image_dropdown = gr.Dropdown(
                    choices=processor.get_image_list(),
                    label="Seleccionar Imagen",
                    value=None
                )
                refresh_btn = gr.Button("🔄 Actualizar Lista")
                
                gr.Markdown("## ⚙️ Controles")
                
                with gr.Tabs():
                    with gr.TabItem("🎨 Espacios de Color"):
                        color_space = gr.Radio(
                            choices=['BGR', 'RGB', 'HSV', 'LAB', 'GRAYSCALE'],
                            value='RGB',
                            label="Espacio de Color"
                        )
                        
                        gr.Markdown("### Separación de Canales")
                        channel_color_space = gr.Radio(
                            choices=['RGB', 'HSV', 'LAB', 'BGR'],
                            value='RGB',
                            label="Espacio para Canales"
                        )
                        channel_select = gr.Radio(
                            choices=['R', 'G', 'B'],
                            value='R',
                            label="Canal"
                        )
                        
                        gr.Markdown("### Mezcla de Canales")
                        r_channel = gr.Dropdown(
                            choices=['R', 'G', 'B', 'H', 'S', 'V', 'L', 'A', 'B_lab'],
                            value='R',
                            label="Canal R"
                        )
                        g_channel = gr.Dropdown(
                            choices=['R', 'G', 'B', 'H', 'S', 'V', 'L', 'A', 'B_lab'],
                            value='G',
                            label="Canal G"
                        )
                        b_channel = gr.Dropdown(
                            choices=['R', 'G', 'B', 'H', 'S', 'V', 'L', 'A', 'B_lab'],
                            value='B',
                            label="Canal B"
                        )
                    
                    with gr.TabItem("🔧 Transformaciones"):
                        transform_type = gr.Radio(
                            choices=['rotate', 'flip_horizontal', 'flip_vertical', 'resize'],
                            value='rotate',
                            label="Transformación"
                        )
                        angle_slider = gr.Slider(
                            minimum=-180,
                            maximum=180,
                            value=0,
                            step=1,
                            label="Ángulo"
                        )
                        scale_slider = gr.Slider(
                            minimum=0.1,
                            maximum=3.0,
                            value=1.0,
                            step=0.1,
                            label="Escala"
                        )
                    
                    with gr.TabItem("🔆 Brillo y Contraste"):
                        brightness_slider = gr.Slider(
                            minimum=-100,
                            maximum=100,
                            value=0,
                            step=1,
                            label="Brillo"
                        )
                        contrast_slider = gr.Slider(
                            minimum=0.1,
                            maximum=3.0,
                            value=1.0,
                            step=0.1,
                            label="Contraste"
                        )
                        gamma_slider = gr.Slider(
                            minimum=0.1,
                            maximum=3.0,
                            value=1.0,
                            step=0.1,
                            label="Gamma"
                        )
                        eq_type = gr.Radio(
                            choices=['global', 'local'],
                            value='global',
                            label="Tipo de Ecualización"
                        )
                    
                    with gr.TabItem("🎯 Filtros"):
                        filter_type = gr.Dropdown(
                            choices=['mean', 'gaussian', 'bilateral', 'median', 'box', 
                                   'laplacian', 'sobel_x', 'sobel_y', 'canny', 'sharpen'],
                            value='gaussian',
                            label="Tipo de Filtro"
                        )
                        kernel_size = gr.Slider(
                            minimum=3,
                            maximum=21,
                            value=5,
                            step=2,
                            label="Tamaño del Kernel"
                        )
                        sigma_slider = gr.Slider(
                            minimum=0.1,
                            maximum=5.0,
                            value=1.0,
                            step=0.1,
                            label="Sigma (Gaussian)"
                        )
                    
                    with gr.TabItem("🎭 Umbrales"):
                        threshold_type = gr.Dropdown(
                            choices=['binary', 'binary_inv', 'adaptive_mean', 'adaptive_gaussian', 'otsu'],
                            value='binary',
                            label="Tipo de Umbral"
                        )
                        threshold_value = gr.Slider(
                            minimum=0,
                            maximum=255,
                            value=127,
                            step=1,
                            label="Valor de Umbral"
                        )
                        bitwise_op = gr.Radio(
                            choices=['and', 'or', 'not', 'xor'],
                            value='and',
                            label="Operación Bitwise"
                        )
                    
                    with gr.TabItem("🖼️ Fondo"):
                        bg_method = gr.Radio(
                            choices=['hsv', 'lab'],
                            value='hsv',
                            label="Método"
                        )
                        bg_color = gr.Radio(
                            choices=['white', 'green'],
                            value='white',
                            label="Color de Fondo a Eliminar"
                        )
                        new_bg = gr.Radio(
                            choices=['white', 'black', 'green'],
                            value='white',
                            label="Nuevo Fondo"
                        )
            
            with gr.Column(scale=2):
                gr.Markdown("## 🖼️ Visualización")
                
                with gr.Tabs():
                    with gr.TabItem("📷 Original"):
                        original_image = gr.Image(label="Imagen Original")
                    
                    with gr.TabItem("🎨 Espacios de Color"):
                        color_output = gr.Image(label="Conversión de Color")
                        channel_output = gr.Image(label="Canal Separado")
                        mix_output = gr.Image(label="Mezcla de Canales")
                    
                    with gr.TabItem("🔧 Transformaciones"):
                        transform_output = gr.Image(label="Transformación")
                    
                    with gr.TabItem("🔆 Corrección"):
                        brightness_output = gr.Image(label="Brillo y Contraste")
                        gamma_output = gr.Image(label="Corrección Gamma")
                        hist_output = gr.Image(label="Ecualización de Histograma")
                    
                    with gr.TabItem("🎯 Filtros"):
                        filter_output = gr.Image(label="Filtro Aplicado")
                    
                    with gr.TabItem("🎭 Umbrales"):
                        threshold_output = gr.Image(label="Umbralización")
                        bitwise_output = gr.Image(label="Operación Bitwise")
                    
                    with gr.TabItem("🖼️ Procesamiento de Fondo"):
                        remove_bg_output = gr.Image(label="Eliminación de Fondo")
                        change_bg_output = gr.Image(label="Cambio de Fondo")
                    
                    with gr.TabItem("📊 Comparación"):
                        stack_output = gr.Image(label="Comparación de Operaciones")
        
        # Conectar eventos
        def refresh_images():
            return gr.Dropdown.update(choices=processor.get_image_list())
        
        refresh_btn.click(refresh_images, outputs=[image_dropdown])
        
        # Imagen original
        image_dropdown.change(update_image_display, inputs=[image_dropdown], outputs=[original_image])
        
        # Espacios de color
        for input_comp in [image_dropdown, color_space]:
            input_comp.change(convert_color_wrapper, inputs=[image_dropdown, color_space], outputs=[color_output])
        
        for input_comp in [image_dropdown, channel_color_space, channel_select]:
            input_comp.change(separate_channels_wrapper, inputs=[image_dropdown, channel_color_space, channel_select], outputs=[channel_output])
        
        for input_comp in [image_dropdown, r_channel, g_channel, b_channel]:
            input_comp.change(mix_channels_wrapper, inputs=[image_dropdown, r_channel, g_channel, b_channel], outputs=[mix_output])
        
        # Transformaciones
        for input_comp in [image_dropdown, transform_type, angle_slider, scale_slider]:
            input_comp.change(transform_wrapper, inputs=[image_dropdown, transform_type, angle_slider, scale_slider], outputs=[transform_output])
        
        # Corrección de imagen
        for input_comp in [image_dropdown, brightness_slider, contrast_slider]:
            input_comp.change(brightness_contrast_wrapper, inputs=[image_dropdown, brightness_slider, contrast_slider], outputs=[brightness_output])
        
        for input_comp in [image_dropdown, gamma_slider]:
            input_comp.change(gamma_wrapper, inputs=[image_dropdown, gamma_slider], outputs=[gamma_output])
        
        for input_comp in [image_dropdown, eq_type]:
            input_comp.change(histogram_wrapper, inputs=[image_dropdown, eq_type], outputs=[hist_output])
        
        # Filtros
        for input_comp in [image_dropdown, filter_type, kernel_size, sigma_slider]:
            input_comp.change(filter_wrapper, inputs=[image_dropdown, filter_type, kernel_size, sigma_slider], outputs=[filter_output])
        
        # Umbrales
        for input_comp in [image_dropdown, threshold_type, threshold_value]:
            input_comp.change(threshold_wrapper, inputs=[image_dropdown, threshold_type, threshold_value], outputs=[threshold_output])
        
        for input_comp in [image_dropdown, bitwise_op]:
            input_comp.change(bitwise_wrapper, inputs=[image_dropdown, bitwise_op], outputs=[bitwise_output])
        
        # Procesamiento de fondo
        for input_comp in [image_dropdown, bg_method, bg_color]:
            input_comp.change(remove_bg_wrapper, inputs=[image_dropdown, bg_method, bg_color], outputs=[remove_bg_output])
        
        for input_comp in [image_dropdown, new_bg]:
            input_comp.change(change_bg_wrapper, inputs=[image_dropdown, new_bg], outputs=[change_bg_output])
        
        # Comparación
        image_dropdown.change(stack_wrapper, inputs=[image_dropdown], outputs=[stack_output])
        
        # Actualizar opciones de canal según el espacio de color seleccionado
        def update_channel_options(color_space):
            if color_space == 'RGB':
                return gr.Radio.update(choices=['R', 'G', 'B'], value='R')
            elif color_space == 'HSV':
                return gr.Radio.update(choices=['H', 'S', 'V'], value='H')
            elif color_space == 'LAB':
                return gr.Radio.update(choices=['L', 'A', 'B'], value='L')
            else:  # BGR
                return gr.Radio.update(choices=['B', 'G', 'R'], value='B')
        
        channel_color_space.change(update_channel_options, inputs=[channel_color_space], outputs=[channel_select])
    
    return demo

# Función principal para ejecutar la aplicación
def main():
    # Crear algunas imágenes de ejemplo si no existen
    if not os.path.exists("images"):
        os.makedirs("images")
        print("📁 Carpeta 'images' creada.")
        print("Por favor, agrega algunas imágenes (.jpg, .png, .bmp) a la carpeta 'images' para comenzar.")
    
    # Crear y lanzar la interfaz
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )

if __name__ == "__main__":
    main()