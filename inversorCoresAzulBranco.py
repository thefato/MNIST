###########################################################################
###   I N V E R S O R    D E   C O R E S    I M A G E N S    M N I S T  ###
###########################################################################
###         inverte as cores das imagens do MNIST em uma pasta          ###
###########################################################################

import os
from PIL import Image

# Diretório onde as imagens estão salvas
input_dir = 'mnist_bmp_inverted'
# Diretório onde as imagens invertidas serão salvas
output_dir = 'mnist_bmp_inverted_final'

os.makedirs(output_dir, exist_ok=True)

# Lista todos os arquivos BMP no diretório de entrada
for filename in os.listdir(input_dir):
    if filename.endswith('.bmp'):
        filepath = os.path.join(input_dir, filename)
        # Abre a imagem
        img = Image.open(filepath)

        # Converte a imagem para RGB 
        img_rgb = img.convert('RGB')

        pixels = img_rgb.load()

        for y in range(img.size[1]):
            for x in range(img.size[0]):
                pixel = img.getpixel((x, y))
                azul = 255 - pixel
                if azul > 20:  # para evitar sujeiras ao fundo e gerar erros no treinamento
                    pixels[x, y] = (0, 0, azul)
                else:
                    pixels[x, y] = (255, 255, 255)  # fundo branco


        
        
        # Cria o novo nome com o sufixo "_inverted_final.bmp"
        name_part = filename[:-4]  # remove ".bmp"
        new_filename = f"{name_part}_inverted_final.bmp"
        new_filepath = os.path.join(output_dir, new_filename)
        # Salva a imagem invertida
        img_rgb.save(new_filepath)
        print(f"Invertida e salva: {new_filepath}")

print("Adição de azul no branco concluída!")
