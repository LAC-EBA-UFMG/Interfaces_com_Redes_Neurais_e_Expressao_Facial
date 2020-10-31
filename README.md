# Detectores Neural de Expressão Facial

## Abertura dos Olhos

Modelo otimizado de detecção de rosto baseado na RFB-320:
Custo aproximado entre 90~109 MFlops: 
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