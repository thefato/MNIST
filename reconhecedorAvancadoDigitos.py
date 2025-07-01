###########################################################################
###     R E C O N H E C E D O R    D E    D I G I T O S    M N I S T    ###
###########################################################################
###   utiliza o modelo informado para realizar o reconhecimento do      ###
###                digito existente na imagem PNG ou BMP                ###
###########################################################################

import os
import numpy as np
from PIL import Image
import tensorflow as tf  # Importar tensorflow para usar tf.config.list_physical_devices
from tensorflow.keras.models import load_model

# --- Tenta configurar GPU ou informa que usará CPU ---
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs disponíveis para uso: {gpus}. Crescimento de memória habilitado.")
        print(
            "Treinamento será acelerado pela GPU se as bibliotecas CUDA/cuDNN estiverem configuradas."
        )
    except RuntimeError as e:
        print(f"Erro ao configurar GPU: {e}")
        print(
            "GPU pode não estar totalmente configurada ou acessível. Treinamento na CPU."
        )
else:
    print("Nenhuma GPU detectada ou configurada. Treinando na CPU.")

model_dir = "."

model_files = [f for f in os.listdir(model_dir) if f.endswith((".h5", ".keras"))]

if not model_files:
    print(
        f"Nenhum arquivo de modelo (.h5 ou .keras) encontrado na pasta '{model_dir}'."
    )
    print(
        "Por favor, certifique-se de que seus modelos estão salvos no formato correto e nesta pasta."
    )
    exit()

print("\nModelos disponíveis:")
for i, filename in enumerate(model_files):
    print(f"{i + 1}. {filename}")

modelo_carregado = None
nome_modelo_selecionado = None

# Loop para seleção e carregamento do modelo
while modelo_carregado is None:
    escolha = input(
        "Digite o NÚMERO do modelo que deseja usar (ou 's' para sair): "
    ).lower()

    if escolha == "s":
        print("Saindo do programa.")
        exit()

    try:
        escolha_index = int(escolha) - 1

        if 0 <= escolha_index < len(model_files):
            nome_modelo_selecionado = model_files[escolha_index]
            print(f"Você selecionou: {nome_modelo_selecionado}")
            try:
                modelo_carregado = load_model(
                    os.path.join(model_dir, nome_modelo_selecionado)
                )
                print(f"Modelo '{nome_modelo_selecionado}' carregado com sucesso!")
            except Exception as e:
                print(f"Erro ao carregar o modelo: {e}")
                print(
                    "Por favor, verifique a compatibilidade do modelo com sua versão do TensorFlow/Keras ou tente outro modelo."
                )
                # Não sai, permite que o usuário tente novamente
        else:
            print("Escolha inválida. Por favor, digite um número da lista.")
    except ValueError:
        print("Entrada inválida. Por favor, digite um número ou 's'.")


while True:
    input_dir = "imgs"
    imagem_path = None
    nome_imagem = None

    if not os.path.exists(input_dir):
        print(
            f"\nAVISO: A pasta de entrada de imagens '{input_dir}' não existe."
            " Crie-a e coloque suas imagens lá."
        )
        nome_imagem = input(
            "Digite o caminho COMPLETO da imagem de entrada (BMP ou PNG, ou 's' para sair): "
        ).lower()
        if nome_imagem == "s":
            break  # Sai do loop principal
        imagem_path = nome_imagem
    else:
        image_files = [
            f
            for f in os.listdir(input_dir)
            if f.lower().endswith((".bmp", ".png", ".jpg", ".jpeg"))
        ]
        if not image_files:
            print(
                f"\nNenhuma imagem (BMP, PNG, JPG) encontrada na pasta '{input_dir}'."
            )
            nome_imagem = input(
                "Digite o caminho COMPLETO da imagem de entrada (BMP ou PNG, ou 's' para sair): "
            ).lower()
            if nome_imagem == "s":
                break  # Sai do loop principal
            imagem_path = nome_imagem
        else:
            print(f"\nImagens disponíveis na pasta '{input_dir}':")
            for i, filename in enumerate(image_files):
                print(f"{i + 1}. {filename}")

            escolha_imagem_valida = False
            while not escolha_imagem_valida:
                escolha_img = input(
                    "Digite o NÚMERO da imagem que deseja usar (ou 's' para sair): "
                ).lower()
                if escolha_img == "s":
                    break  # Sai do loop de escolha de imagem e, consequentemente, do loop principal
                try:
                    escolha_img_index = int(escolha_img) - 1
                    if 0 <= escolha_img_index < len(image_files):
                        nome_imagem_selecionada = image_files[escolha_img_index]
                        imagem_path = os.path.join(input_dir, nome_imagem_selecionada)
                        escolha_imagem_valida = True
                    else:
                        print("Escolha inválida. Por favor, digite um número da lista.")
                except ValueError:
                    print("Entrada inválida. Por favor, digite um número ou 's'.")

            if escolha_img == "s":  # Verifica se o usuário saiu do loop aninhado
                break
            print(f"Você selecionou a imagem: {nome_imagem_selecionada}")

    # Verifica se o caminho da imagem foi definido antes de tentar processá-la
    if imagem_path is None:
        print("Nenhuma imagem selecionada. Saindo.")
        break  # Sai do loop principal se o usuário saiu antes de selecionar imagem

    try:
        img = Image.open(imagem_path).convert("RGB")
        img = img.resize((28, 28))
        img_array = np.array(img).astype("float32")
        img_array = np.expand_dims(img_array, axis=0)

    except Exception as e:
        print(f"Erro ao processar a imagem '{imagem_path}': {e}")
        # Se houver erro na imagem, permite que o usuário tente outra
        continue

    # Faz a previsão
    predicoes = modelo_carregado.predict(img_array)  # Usa o modelo_carregado
    digito_previsto = np.argmax(predicoes)

    print(f"\nPrevisão do dígito: {digito_previsto}")
    print(f"Confiança: {np.max(predicoes) * 100:.2f}%")

    # Opção de sair após a previsão
    continuar = input(
        "\nPressione Enter para reconhecer outra imagem, ou digite 's' para sair: "
    ).lower()
    if continuar == "s":
        break  # Sai do loop principal

print("\nProcesso de reconhecimento concluído. Até a próxima!")
