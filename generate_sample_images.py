import cv2
import numpy as np
import os

def create_sample_images():
    """Crear imágenes de muestra para probar la aplicación"""
    
    # Crear directorio images si no existe
    if not os.path.exists("images"):
        os.makedirs("images")
        print("📁 Carpeta 'images' creada.")
    
    # Imagen 1: Círculos de colores
    img1 = np.zeros((400, 400, 3), dtype=np.uint8)
    img1[:] = (255, 255, 255)  # Fondo blanco
    
    # Círculos de diferentes colores
    cv2.circle(img1, (100, 100), 50, (255, 0, 0), -1)      # Rojo
    cv2.circle(img1, (300, 100), 50, (0, 255, 0), -1)      # Verde
    cv2.circle(img1, (100, 300), 50, (0, 0, 255), -1)      # Azul
    cv2.circle(img1, (300, 300), 50, (255, 255, 0), -1)    # Amarillo
    cv2.circle(img1, (200, 200), 30, (255, 0, 255), -1)    # Magenta
    
    cv2.imwrite("images/circulos_colores.jpg", img1)
    print("✅ Imagen 1 creada: circulos_colores.jpg")
    
    # Imagen 2: Formas geométricas
    img2 = np.zeros((400, 400, 3), dtype=np.uint8)
    img2[:] = (240, 240, 240)  # Fondo gris claro
    
    # Rectángulos
    cv2.rectangle(img2, (50, 50), (150, 150), (0, 255, 255), -1)
    cv2.rectangle(img2, (250, 50), (350, 150), (255, 165, 0), -1)
    
    # Triángulos
    pts1 = np.array([[100, 250], [50, 350], [150, 350]], np.int32)
    pts2 = np.array([[300, 250], [250, 350], [350, 350]], np.int32)
    cv2.fillPoly(img2, [pts1], (128, 0, 128))
    cv2.fillPoly(img2, [pts2], (255, 20, 147))
    
    cv2.imwrite("images/formas_geometricas.jpg", img2)
    print("✅ Imagen 2 creada: formas_geometricas.jpg")
    
    # Imagen 3: Gradientes
    img3 = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Gradiente horizontal
    for i in range(400):
        color_val = int(255 * i / 400)
        img3[:200, i] = (color_val, 255 - color_val, 128)
    
    # Gradiente vertical
    for i in range(400):
        color_val = int(255 * i / 400)
        img3[200:, :, 0] = color_val
        img3[200:, :, 1] = 255 - color_val
        img3[200:, :, 2] = 128
    
    cv2.imwrite("images/gradientes.jpg", img3)
    print("✅ Imagen 3 creada: gradientes.jpg")
    
    # Imagen 4: Patrón de textura
    img4 = np.zeros((400, 400, 3), dtype=np.uint8)
    img4[:] = (255, 255, 255)  # Fondo blanco
    
    # Crear patrón de líneas
    for i in range(0, 400, 20):
        cv2.line(img4, (i, 0), (i, 400), (0, 0, 0), 2)
        cv2.line(img4, (0, i), (400, i), (0, 0, 0), 2)
    
    # Añadir algunos círculos
    for i in range(50, 400, 100):
        for j in range(50, 400, 100):
            cv2.circle(img4, (i, j), 25, (255, 0, 0), 3)
    
    cv2.imwrite("images/patron_textura.jpg", img4)
    print("✅ Imagen 4 creada: patron_textura.jpg")
    
    # Imagen 5: Objeto con fondo verde (para eliminación de fondo)
    img5 = np.zeros((400, 400, 3), dtype=np.uint8)
    img5[:] = (0, 255, 0)  # Fondo verde
    
    # Objeto central
    cv2.circle(img5, (200, 200), 80, (255, 255, 255), -1)  # Círculo blanco
    cv2.circle(img5, (200, 200), 80, (0, 0, 0), 3)         # Borde negro
    cv2.rectangle(img5, (170, 170), (230, 230), (255, 0, 0), -1)  # Cuadrado rojo
    
    cv2.imwrite("images/objeto_fondo_verde.jpg", img5)
    print("✅ Imagen 5 creada: objeto_fondo_verde.jpg")
    
    # Imagen 6: Imagen con ruido
    img6 = np.zeros((400, 400, 3), dtype=np.uint8)
    img6[:] = (100, 150, 200)  # Fondo azul grisáceo
    
    # Añadir formas
    cv2.rectangle(img6, (100, 100), (300, 300), (255, 255, 255), -1)
    cv2.circle(img6, (200, 200), 50, (255, 0, 0), -1)
    
    # Añadir ruido
    noise = np.random.randint(0, 50, (400, 400, 3), dtype=np.uint8)
    img6 = cv2.add(img6, noise)
    
    cv2.imwrite("images/imagen_con_ruido.jpg", img6)
    print("✅ Imagen 6 creada: imagen_con_ruido.jpg")
    
    print("\n🎉 ¡Imágenes de muestra creadas exitosamente!")
    print("📁 Las imágenes están disponibles en la carpeta 'images'")
    print("🚀 Ahora puedes ejecutar la aplicación con: python app.py")

if __name__ == "__main__":
    create_sample_images()