import random
import cv2
import numpy as np
import pygame
import sys
from ultralytics import YOLO

# Inicialização do Pygame
pygame.init()
pygame.font.init()

# Configurações da Janela do Jogo
LARGURA_TELA, ALTURA_TELA = 800, 600
ecrã = pygame.display.set_mode((LARGURA_TELA, ALTURA_TELA))
pygame.display.set_caption("Jogo com Controlo por Movimento - YOLOv8 & Pygame")
relogio = pygame.time.Clock()

# Cores
BRANCO = (255, 255, 255)
AZUL = (52, 152, 219)
VERMELHO = (231, 76, 60)
VERDE = (46, 204, 113)
PRETO = (44, 62, 80)
CINZA = (127, 140, 141)

# Fontes
fonte = pygame.font.SysFont("Arial", 28)
fonte_grande = pygame.font.SysFont("Arial", 48)

# Carregar o modelo YOLOv8 standard (altamente fiável e rápido em CPU)
print("A carregar o modelo YOLOv8 Standard...")
model = YOLO("yolov8n.pt")

# Inicializar a Câmara do PC
cap = cv2.VideoCapture(0)
if not cap.isOpened():
  print("Erro: Não foi possível aceder à câmara.")
  sys.exit()

# Variáveis do Jogo
jogador_largura, jogador_altura = 120, 20
jogador_x = LARGURA_TELA // 2 - jogador_largura // 2
jogador_y = ALTURA_TELA - 50

objeto_raio = 15
objeto_x = random.randint(objeto_raio, LARGURA_TELA - objeto_raio)
objeto_y = -50
objeto_velocidade = 6

pontuacao = 0
vidas = 3
jogo_ativo = True

print("Jogo iniciado! Mova-se em frente à câmara para controlar a raquete.")

while True:
  # 1. Gestão de Eventos do Pygame
  for evento in pygame.event.get():
    if evento.type == pygame.QUIT:
      cap.release()
      pygame.quit()
      sys.exit()
    if evento.type == pygame.KEYDOWN and evento.key == pygame.K_r:
      pontuacao = 0
      vidas = 3
      jogo_ativo = True
      objeto_y = -50
      objeto_velocidade = 6

  # 2. Captura e Processamento da Câmara (OpenCV + YOLO)
  ret, frame = cap.read()
  if ret:
    frame = cv2.flip(frame, 1)  # Espelhar imagem para movimento natural
    h_cam, w_cam, _ = frame.shape

    # Inferência focada na classe 0 (Pessoa)
    results = model(frame, conf=0.4, classes=[0], verbose=False)
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
      # Obter as coordenadas da caixa delimitadora da pessoa detetada
      box = boxes[0].xyxy[0].cpu().numpy()
      x1, y1, x2, y2 = map(int, box)

      # Calcular o centro horizontal da pessoa
      center_x = int((x1 + x2) / 2)

      # Mapear a posição horizontal da câmara para a largura do ecrã do Pygame
      jogador_x = np.interp(
          center_x, (50, w_cam - 50), (0, LARGURA_TELA - jogador_largura)
      )

  # 3. Lógica do Jogo
  if jogo_ativo:
    objeto_y += objeto_velocidade

    # Detetar colisão com a barra do jogador
    jogador_rect = pygame.Rect(
        jogador_x, jogador_y, jogador_largura, jogador_altura
    )
    objeto_rect = pygame.Rect(
        objeto_x - objeto_raio,
        objeto_y - objeto_raio,
        objeto_raio * 2,
        objeto_raio * 2,
    )

    if jogador_rect.colliderect(objeto_rect):
      pontuacao += 1
      objeto_y = -50
      objeto_x = random.randint(objeto_raio, LARGURA_TELA - objeto_raio)
      objeto_velocidade += 0.3  # Aumentar dificuldade gradualmente

    # Se o objeto cair no fundo do ecrã
    if objeto_y > ALTURA_TELA:
      vidas -= 1
      objeto_y = -50
      objeto_x = random.randint(objeto_raio, LARGURA_TELA - objeto_raio)
      if vidas <= 0:
        jogo_ativo = False

  # 4. Renderização Gráfica no Pygame
  ecrã.fill(PRETO)

  if jogo_ativo:
    # Desenhar Jogador (Barra)
    pygame.draw.rect(
        ecrã,
        AZUL,
        (jogador_x, jogador_y, jogador_largura, jogador_altura),
        border_radius=10,
    )

    # Desenhar Objeto em queda
    pygame.draw.circle(ecrã, VERDE, (int(objeto_x), int(objeto_y)), objeto_raio)

    # Texto de Pontuação e Vidas
    txt_pontos = fonte.render(f"Pontuação: {pontuacao}", True, BRANCO)
    txt_vidas = fonte.render(f"Vidas: {vidas}", True, VERMELHO)
    ecrã.blit(txt_pontos, (20, 20))
    ecrã.blit(txt_vidas, (LARGURA_TELA - 140, 20))
  else:
    # Ecrã de Fim de Jogo (Game Over)
    txt_Fim = fonte_grande.render("FIM DE JOGO", True, VERMELHO)
    txt_Final = fonte.render(f"Pontuação Final: {pontuacao}", True, BRANCO)
    txt_Reiniciar = fonte.render(
        "Pressione 'R' para Jogar Novamente", True, CINZA
    )

    ecrã.blit(txt_Fim, (LARGURA_TELA // 2 - txt_Fim.get_width() // 2, 200))
    ecrã.blit(
        txt_Final, (LARGURA_TELA // 2 - txt_Final.get_width() // 2, 270)
    )
    ecrã.blit(
        txt_Reiniciar,
        (LARGURA_TELA // 2 - txt_Reiniciar.get_width() // 2, 340),
    )

  pygame.display.flip()
  relogio.tick(30)