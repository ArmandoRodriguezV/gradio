🖼️ Aplicación de Procesamiento de Imágenes con OpenCV y Gradio
📋 Descripción del Proyecto
Esta aplicación web interactiva permite procesar imágenes utilizando diversas técnicas de visión por computadora implementadas con OpenCV. La interfaz gráfica está desarrollada con Gradio, proporcionando una experiencia de usuario intuitiva y amigable.

🚀 Características Principales
1. Carga y Visualización de Imágenes
Selección de imágenes desde una lista desplegable
Visualización de la imagen original
Actualización dinámica de la lista de imágenes disponibles
2. Operaciones con Canales de Color
Conversión entre espacios de color: BGR, RGB, HSV, LAB, GRAYSCALE
Separación e inspección de canales individuales
Mezcla personalizada de canales entre diferentes espacios de color
3. Corrección de Imágenes
Transformaciones geométricas: rotación, volteo horizontal/vertical, redimensionamiento
Ajustes de iluminación: brillo, contraste y corrección gamma
Ecualización de histograma: global y local (CLAHE)
4. Mejoramiento de Imagen (10 Filtros)
Filtro de Media: Suavizado básico
Filtro Gaussiano: Suavizado con distribución gaussiana
Filtro Bilateral: Preserva bordes mientras suaviza
Filtro de Mediana: Elimina ruido tipo sal y pimienta
Filtro Box: Suavizado uniforme
Filtro Laplaciano: Detección de bordes
Filtro Sobel X: Gradientes horizontales
Filtro Sobel Y: Gradientes verticales
Filtro Canny: Detección de bordes avanzada
Filtro Sharpen: Realce de bordes y detalles
5. Aplicación de Máscaras y Umbrales
Umbrales: Binario, binario invertido, adaptativo (media y gaussiano), Otsu
Operaciones binarias: AND, OR, NOT, XOR
Creación y aplicación de máscaras personalizadas
6. Eliminación de Fondo
Segmentación por espacio de color HSV
Segmentación por espacio de color LAB
Eliminación de fondos blancos y verdes
Uso de cv2.inRange para crear máscaras
7. Cambio de Fondo
Reemplazo de fondo por colores sólidos (blanco, negro, verde)
Extracción de objetos de interés
Combinación de foreground y background
8. Apilado de Imágenes
Comparación visual tipo collage
Muestra original + procesadas en una sola vista
Uso de np.hstack y np.vstack
9. Control de Parámetros Interactivos
Sliders para ajustar umbrales, filtros, brillo, contraste
Radio buttons para seleccionar métodos y operaciones
Dropdowns para espacios de color y tipos de filtros
📦 Instalación y Configuración
Requisitos Previos
bash
pip install opencv-python numpy gradio pillow matplotlib
Instalación
Clona o descarga el proyecto
Crea la carpeta images en el directorio del proyecto
Agrega imágenes de prueba (.jpg, .png, .bmp) a la carpeta images
Ejecutar la Aplicación
bash
python app.py
La aplicación se ejecutará en http://localhost:7860

📁 Estructura del Proyecto
proyecto/
├── app.py              # Aplicación principal
├── README.md           # Documentación
├── images/             # Carpeta con imágenes de prueba
│   ├── imagen1.jpg
│   ├── imagen2.png
│   └── ...
└── requirements.txt    # Dependencias (opcional)
🎮 Uso de la Aplicación
Paso 1: Seleccionar Imagen
Utiliza el dropdown "Seleccionar Imagen" para elegir una imagen
La imagen original se mostrará en la pestaña "📷 Original"
Paso 2: Aplicar Operaciones
Navega por las diferentes pestañas de controles:

🎨 Espacios de Color: Experimenta con diferentes espacios de color y canales
🔧 Transformaciones: Aplica rotaciones, volteos y redimensionamiento
🔆 Brillo y Contraste: Ajusta la iluminación de la imagen
🎯 Filtros: Aplica diferentes tipos de filtros y efectos
🎭 Umbrales: Experimenta con binarización y operaciones binarias
🖼️ Fondo: Elimina y cambia fondos de imágenes
Paso 3: Visualizar Resultados
Los resultados se muestran en tiempo real en las pestañas correspondientes de la sección "🖼️ Visualización".

🔧 Funcionalidades Técnicas
Procesamiento en Tiempo Real
Todos los cambios se aplican inmediatamente
Interfaz reactiva con actualización automática
Visualización simultánea de múltiples operaciones
Optimización de Performance
Redimensionamiento automático para visualización
Gestión eficiente de memoria
Procesamiento optimizado con OpenCV
Robustez
Manejo de errores para imágenes no válidas
Validación de parámetros de entrada
Interfaz tolerante a fallos
🎯 Casos de Uso
Educativo
Aprendizaje de procesamiento de imágenes
Experimentación con diferentes técnicas
Comparación visual de resultados
Procesamiento Básico
Mejoramiento de calidad de imagen
Eliminación de fondos
Ajuste de colores y brillo
Análisis de Imágenes
Detección de bordes
Segmentación por color
Análisis de canales
🔬 Funcionalidades Avanzadas Implementadas
Algoritmos de Segmentación
Segmentación por color en múltiples espacios
Creación de máscaras personalizadas
Operaciones morfológicas implícitas
Mejoramiento Adaptativo
Ecualización de histograma local (CLAHE)
Filtros adaptativos
Corrección gamma automática
Visualización Comparativa
Apilado inteligente de imágenes
Comparación lado a lado
Grid de múltiples operaciones
🎨 Interfaz de Usuario
Diseño Responsivo
Organización en pestañas para mejor navegación
Controles intuitivos y bien organizados
Visualización clara de resultados
Experiencia de Usuario
Feedback visual inmediato
Controles con rangos apropiados
Etiquetas descriptivas en español
🐛 Solución de Problemas
Problema: No se muestran imágenes
Solución: Verifica que la carpeta images exista y contenga imágenes válidas (.jpg, .png, .bmp)

Problema: Error al aplicar filtros
Solución: Asegúrate de que el tamaño del kernel sea impar y apropiado para el tamaño de la imagen

Problema: Resultados inesperados en eliminación de fondo
Solución: Ajusta los rangos de color según las características específicas de tu imagen

📊 Métricas de Rendimiento
Tiempo de Procesamiento
Operaciones básicas: < 100ms
Filtros complejos: < 500ms
Operaciones de fondo: < 1s
Memoria
Uso eficiente de memoria con OpenCV
Liberación automática de recursos
Optimización para imágenes grandes
🚀 Extensiones Futuras
Funcionalidades Adicionales
Detección de objetos con Haar Cascades
Reconocimiento de contornos
Estadísticas detalladas de imagen
Mejoras de Interfaz
Histogramas interactivos
Zoom y pan en imágenes
Guardado de configuraciones
📝 Notas Técnicas
Compatibilidad
Python 3.7+
OpenCV 4.x
Gradio 3.x+
Limitaciones
Tamaño máximo de imagen recomendado: 2048x2048
Formatos soportados: JPG, PNG, BMP, TIFF
Procesamiento secuencial (no paralelo)
👥 Contribuciones
Este proyecto fue desarrollado como parte del curso de Multimedia, implementando todas las funcionalidades requeridas y algunas adicionales para demostrar un entendimiento profundo del procesamiento de imágenes con OpenCV.

📄 Licencia
Proyecto académico desarrollado para fines educativos.

Desarrollado con ❤️ usando OpenCV y Gradio

