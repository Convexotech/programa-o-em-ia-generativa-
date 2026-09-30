import cv2
import numpy as np
import pyautogui
import streamlit as st
from ultralytics import YOLO

# Configurações de segurança do PyAutoGUI
pyautogui.FAILSAFE = True

# Configuração da página Streamlit
st.set_page_config(
    page_title="Controlo do Rato por Movimento - YOLOv8",
    page_icon="🖱️",
    layout="centered",
)

st.title("Controlo do Rato com Movimento (YOLOv8) 🖐️🖱️")
st.markdown(
    "Solução otimizada utilizando o modelo padrão **YOLOv8n**, **OpenCV** e"
    " **PyAutoGUI** a executar em **CPU**."
)

st.warning(
    "⚠️ **Aviso de Segurança:** Para interromper o controlo automático do"
    " rato, desloque bruscamente o cursor para qualquer um dos 4 cantos"
    " extremos do ecrã (Ativação do Fail-Safe)."
)


# Carregar o modelo YOLOv8 standard (deteta pessoas e 80 classes)
@st.cache_resource
def load_yolo_model():
  return YOLO("yolov8n.pt")


with st.spinner("A carregar o modelo de IA (YOLOv8 Standard)..."):
  model = load_yolo_model()

# Obter dimensões do ecrã para mapeamento de coordenadas
screen_width, screen_height = pyautogui.size()

st.markdown("---")
run_control = st.checkbox("Ativar Controlo do Rato")

video_placeholder = st.empty()
status_placeholder = st.empty()

if run_control:
  cap = cv2.VideoCapture(0)

  if not cap.isOpened():
    st.error("Erro ao aceder à câmara.")
  else:
    status_placeholder.info(
        "Controlo ativo! Movimente-se em frente à câmara. Desmarque a caixa"
        " para parar."
    )

    smoothening = 5
    prev_x, prev_y = 0, 0
    curr_x, curr_y = 0, 0

    while run_control:
      ret, frame = cap.read()
      if not ret:
        break

      # Espelhar o frame para interatividade natural (efeito espelho)
      frame = cv2.flip(frame, 1)
      h, w, _ = frame.shape

      # Inferência focada na classe 0 (Pessoa)
      results = model(frame, conf=0.4, classes=[0], verbose=False)
      annotated_frame = frame.copy()

      boxes = results[0].boxes
      if boxes is not None and len(boxes) > 0:
        # Pegar na primeira pessoa detetada (maior caixa ou primeira da lista)
        box = boxes[0].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)

        # Calcular o centro da caixa delimitadora da pessoa
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Desenhar indicador visual no centro detetado
        cv2.circle(
            annotated_frame, (center_x, center_y), 12, (0, 255, 0), cv2.FILLED
        )
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Mapear coordenadas da câmara para a resolução do ecrã do PC
        target_x = np.interp(center_x, (50, w - 50), (0, screen_width))
        target_y = np.interp(center_y, (50, h - 50), (0, screen_height))

        # Suavização para evitar tremores no cursor
        curr_x = prev_x + (target_x - prev_x) / smoothening
        curr_y = prev_y + (target_y - prev_y) / smoothening

        try:
          pyautogui.moveTo(curr_x, curr_y, duration=0)
        except pyautogui.FailSafeException:
          st.error(
              "Fail-Safe ativado! O movimento do rato foi interrompido pelos"
              " cantos do ecrã."
          )
          break

        prev_x, prev_y = curr_x, curr_y

      # Converter para RGB para exibição correta no Streamlit
      annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
      video_placeholder.image(
          annotated_rgb, channels="RGB", use_container_width=True
      )

    cap.release()
    video_placeholder.empty()
    status_placeholder.success("Controlo do rato desativado.")
else:
  st.info("Selecione a caixa acima para iniciar o controlo do rato.")