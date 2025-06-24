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
        #img_rgb = img.convert('RGB')
        # pixels = img_rgb.load()

        width, height = img.size
        img_rgb = Image.new("RGB", (width, height), (255, 255, 255)) #cria um fundo branco ou seja precisa ter uma versão em azul tmb
        pixels_in = img.load()
        pixels_out = img_rgb.load()

       
        for y in range(height):
            for x in range(width):
                value = pixels_in[x, y]
                azul = 255 - value
                if azul > 20:
                    pixels_out[x, y] = (0, 0, azul)
                else:
                    pixels_out[x, y] = (255, 255, 255) #fundo branco

        # Cria o novo nome com o sufixo "_inverted_final.bmp" fundo branco letra azul
        name_part = filename[:-4]  # remove ".bmp"
        new_filename = f"{name_part}_inverted_final.bmp"
        new_filepath = os.path.join(output_dir, new_filename)
        # Salva a imagem invertida
        img_rgb.save(new_filepath)
        print(f"letra azul e salva: {new_filepath}")

        width, height = img.size
        img_rgb = Image.new("RGB", (width, height), (0, 0, 255)) #cria um fundo azul
        pixels_in = img.load()
        pixels_out = img_rgb.load()

       
        for y in range(height):
            for x in range(width):
                value = pixels_in[x, y]
                azul = 255 - value
                if azul > 20:
                    pixels_out[x, y] = (255,255,255)
                else:
                    pixels_out[x, y] = (0, 0, 255) #fundo azul
        
        
        
        # Cria o novo nome com o sufixo "_inverted_final.bmp" fundo azul letra branca
        name_part = filename[:-4]  # remove ".bmp"
        new_filename = f"{name_part}_inverted_azul.bmp"
        new_filepath = os.path.join(output_dir, new_filename)
        # Salva a imagem invertida
        img_rgb.save(new_filepath)
        print(f"letra branca e salva: {new_filepath}")

print("Adição de azul no branco e o inverso concluída!")