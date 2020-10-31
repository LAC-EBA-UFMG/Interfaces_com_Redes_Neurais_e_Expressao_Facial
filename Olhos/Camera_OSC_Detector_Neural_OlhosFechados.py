"""
################## Detector Neural de Olhos Fechados ######################

_uuu__@U@__uuu_

_uuu__~U~__uuu_ <-- Detectado!!!

Autor: Sandro Benigno
Data: 31/10/2020

Criado para envio  de OSC via detecção neural de rosto e olhos fechados.
Para a disciplina Introdução às Narrativas Interativas
ministrada pelo professor André Mintz

Modelo otimizado de detecção de rosto baseado na RFB-320:
    (Custo aproximado entre 90~109 MFlops)
    https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB

Esse programa foi baseado na codificação de:
    Cunjian Chen (ccunjian@gmail.com) (pythorch_face_landmark)
    https://github.com/cunjian/pytorch_face_landmark.git


O padrão de marcadores faciais utilizado aqui é o de 68 pontos (Multi-PIE 68):
    Por C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.
    https://ibug.doc.ic.ac.uk/media/uploads/documents/sagonas_cvpr_2013_amfg_w.pdf
    O modelo treinado de detecção, via HOG e árvores de regressão,
    foi retirado da biblioteca DLIB criada por Davis E. King.

O cálculo de aspecto dos olhos segue os parâmetros indicados no artigo:
    "Real-Time Eye Blink Detection using Facial Landmarks"
    de Tereza Soukupová e Jan Cech
    https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf

    Breve explicação:

    A largura dos olhos, em pixels, se modifica em relação à distância
    da câmera. Assim, utiliza-se a medição dessa largura do olho como base de
    comparação com a medição da altura do olhos, para inferir a sua abertura.

    Cada marcador facial é um vetor (x,y) correspondente ao pixel indexado.
    Para cada olho, utilizam-se 6 desses marcadores, sendo
    2 pares(4 pontos) para a altura e um par(2 pontos) para a largura.

        B*   C*
    A*           D*
        F*   E*

    Equação de EAR (Eye Aspect Ratio)
    (|| B - F || + || C - E ||) / ( 2 * || A - D ||)

    Portanto, a equação é uma razão entre a soma das distâncias euclidianas dos vetores da
    altura do olho em função do dobro da distância euclidiana dos vetores da largura.

#######################################################################

"""
import time
import cv2
import numpy as np
import onnx
import vision.utils.box_utils_numpy as box_utils
from caffe2.python.onnx import backend


#Controle de estado em Play = 1 em Stop = 0
estadoVid = 0

#Contro de tempo de olhos fechados
olhosFechados = 0
tempoMinimo = 15 #iterações minimas com olho fechado

#Controle do EAR limiar de acionamento para o olho fechado
setPoint = 0.3 #quanto menor, mais fechado

#Parte OSC para acionamento externo
from pythonosc.dispatcher import Dispatcher
from typing import List, Any
dispatcher = Dispatcher()

#OSC parametros... 
from pythonosc.udp_client import SimpleUDPClient

#Porta de conexão na mesma máquina
client = SimpleUDPClient("127.0.0.1", 8000)

# onnx runtime
import onnxruntime as ort

# import libraries for landmark
from common.utils import BBox,drawLandmark_multiple
from PIL import Image
import torchvision.transforms as transforms

# setup the parameters
resize = transforms.Resize([112, 112])
to_tensor = transforms.ToTensor()

# import the landmark detection models
import onnx
import onnxruntime
onnx_model_landmark = onnx.load("onnx/pfld.onnx")
onnx.checker.check_model(onnx_model_landmark)
ort_session_landmark = onnxruntime.InferenceSession("onnx/pfld.onnx")
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def euclidean_dist(ptA, ptB):
	# funcao que computa a distancia euclidiana
    # entre dois vetores (x,y)
    # Internamente deve ser -> sqrt(abs(x1-x2)^2 + abs(y1-y2)^2)
    return np.sqrt(
                pow(np.abs(ptA[0]-ptB[0]),2) + 
                pow(np.abs(ptA[1]-ptB[1]),2)
            )
    #Esta função abaixo poderia ser utilizada
    #mas a anterior é mais didática
    #return np.linalg.norm(ptA - ptB)

# Parametros da detecção de rosto
def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


label_path = "models/voc-model-labels.txt"

#Modelo ONNX pra 320px
onnx_path = "models/onnx/version-RFB-320.onnx"
class_names = [name.strip() for name in open(label_path).readlines()]

predictor = onnx.load(onnx_path)
onnx.checker.check_model(predictor)
onnx.helper.printable_graph(predictor.graph)
predictor = backend.prepare(predictor, device="CPU")  # default CPU

ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name

# Captura simples, ainda sem otimização de threads e queue
#!!! Otimizar no futuro !!!
cap = cv2.VideoCapture(1)  # capture from camera
threshold = 0.8 #definição probabilística

sum = 0 #num de iterações, exibe no final
while True:
    ret, orig_image = cap.read()
    #Flipando, porque a imagem da captura veio invertida
    orig_image = cv2.flip(orig_image, 0)
    orig_image = cv2.flip(orig_image, 1)
    if orig_image is None:
        print("no img")
        break

    #Transformações pra adaptar ao modelo de dertecção
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 240))
    # image = cv2.resize(image, (640, 480))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    #Detecção de rosto/cabeça com medição de custo
    time_time = time.time()
    confidences, boxes = ort_session.run(None, {input_name: image})
    print("Custo fase 1: {:.6f}s".format(time.time() - time_time))
    boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
    
    #Inserção das marcas, dentro do box da detecção
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"

        #cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
        # Retirando parâmetros do bbox de inserção das landmarks
        out_size = 56
        img=orig_image.copy()
        height,width,_=img.shape
        x1=box[0]
        y1=box[1]
        x2=box[2]
        y2=box[3]
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        size = int(max([w, h])*1.1)
        cx = x1 + w//2
        cy = y1 + h//2
        x1 = cx - size//2
        x2 = x1 + size
        y1 = cy - size//2
        y2 = y1 + size
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        #testando tamanho da cabeça na img
        #criando uma zona de proximidade desejável
        if size < 200 or size > 350:
            cv2.rectangle(img, (x1, y1), (x2,y2), (0,0,255), 2)
            continue

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)   
        new_bbox = list(map(int, [x1, x2, y1, y2]))
        new_bbox = BBox(new_bbox)

        cropped=img[new_bbox.top:new_bbox.bottom,new_bbox.left:new_bbox.right]
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)            
        cropped_face = cv2.resize(cropped, (out_size, out_size))

        if cropped_face.shape[0]<=0 or cropped_face.shape[1]<=0:
            continue
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)    
        cropped_face = Image.fromarray(cropped_face)
        test_face = resize(cropped_face)
        test_face = to_tensor(test_face)
        test_face.unsqueeze_(0)

        #Detecção dos índices do rosto com medição de custo
        start = time.time()             
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(test_face)}
        ort_outs = ort_session_landmark.run(None, ort_inputs)
        end = time.time()
        print('Custo fase 2: {:.6f}s.'.format(end - start))
        landmark = ort_outs[0]
        landmark = landmark.reshape(-1,2)
        landmark = new_bbox.reprojectLandmark(landmark)

        #Debug de verificação das coordenada de dois pontos
        #Serviu para testar a posição de pontos na matriz de pixels
        #que participariam do cálculo de EAR
        '''
        P1 = 37
        P2 = 40

        X1 = landmark[P1-1][0]
        Y1 = landmark[P1-1][1]

        X2 = landmark[P2-1][0]
        Y2 = landmark[P2-1][1]

        cv2.putText(orig_image, "P1({:.0f},{:.0f}) P2({:.0f},{:.0f})"
                        .format(X1,Y1,X2,Y2), (x1, y1-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)'''

        #Cálculo do EAR (Eyes Aspect Ratio)

        A = euclidean_dist(landmark[38-1],landmark[42-1])
        B = euclidean_dist(landmark[39-1],landmark[41-1])
        C = euclidean_dist(landmark[37-1],landmark[40-1])
        earR = (A + B) / (2 * C)

        A = euclidean_dist(landmark[44-1],landmark[48-1])
        B = euclidean_dist(landmark[45-1],landmark[47-1])
        C = euclidean_dist(landmark[43-1],landmark[46-1])
        earL = (A + B) / (2 * C)
        
        #média do EAR dos dois olhos
        earMed = (earL + earR) / 2.0

        #Exibe largura do bbox da cabeça
        cv2.putText(orig_image, "<----> : {:.0f}pix".format(size), (x1, y1-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.putText(orig_image, "EAR (L,R): ({:.2f},{:.2f})".format(earL,earR), (x1, y2+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        #Detectando e sinalizando a piscada
        #if earL < setP and earR < setP: #forma mais dura, exige os dois olhos fechados
        if earMed < setPoint: #sinaliza pela média dos dois olhos
            olhosFechados += 1
        else:
            olhosFechados = 0

        if olhosFechados > tempoMinimo :
            cv2.circle(orig_image, (cx, y1+30), 30, (0,255,0), -1)
            if estadoVid == 0:
                #Envia comando via msg OSC
                client.send_message("/play", (earL,earR))
                estadoVid = 1
        else:
            if estadoVid == 1:
                client.send_message("/pause", (earL,earR))
                estadoVid = 0

        #função que percorre o array landmarks e carimba todos os pontos
        orig_image = drawLandmark_multiple(orig_image, new_bbox, landmark)

    sum += boxes.shape[0]
    
    #Redução para exibição, caso necessário
    orig_image = cv2.resize(orig_image, (0, 0), fx=0.7, fy=0.7)
    
    cv2.imshow('Detector Neural de Piscadas (Tecle [q] para sair)', orig_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print("sum:{}".format(sum))
