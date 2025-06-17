###########################################################################
###         E X T R A T O R   D E    I M A G E N S    M N I S T         ###
###########################################################################
###              salva as imagens MNIST em uma pasta                    ###
###########################################################################

import os
from tensorflow.keras.datasets import mnist
from PIL import Image

# carregar os dados MNIST
(x_train, y_train), _ = mnist.load_data()

# diretorio onde as imagens serão salvas
output_dir = 'mnist_bmp'
os.makedirs(output_dir, exist_ok=True)

# dicionários para contar as imagens de cada classe
counts = {i: 0 for i in range(10)}

for image, label in zip(x_train, y_train):
    counts[label] += 1
    filename = f"{label}_{counts[label]}.bmp"
    filepath = os.path.join(output_dir, filename)
    img = Image.fromarray(image)
    img.save(filepath)
    print(f"Salvou {filepath}")

print("Extração completa!")
