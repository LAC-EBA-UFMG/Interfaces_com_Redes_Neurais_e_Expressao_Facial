/*==============================================================

  Gerador de Bolinhas por piscadas de olho via OSC
    
  Autor: Sandro Benigno
  Data: 29/10/2020
  
  Criado para testes de um detector neural de rosto e piscadas de olho.
  Para a disciplina Introdução às Narrativas Interativas
  ministrada pelo professor André Mintz
  
  É baseado no exemplo de uso de OSC como integrador entre Processing e PD
  do mesmo autor em 17/08/2020
  para a disciplina Sistemas Musicais Interativos
  ministrada pelo professor Jalver Bethônico Machado

  Biblioteca de OSC para o Processing disponível em:
  http://www.sojamo.de/libraries/oscp5/

================================================================*/

import oscP5.*; //Importando a biblioteca de OSC para o Processing
  
OscP5 oscP5; //Instanciando o objeto da biblioteca OscP5


//Variáveis para criação das bolinhas e controle dos processos
boolean osc_recebido = false;
long count = 0;
int R,G,B = 0;
color c_x = color(0,0,0);
int bolinha_x = 0;
int bolinha_y = 0;
int raio = 20;

//Função de Configuração
void setup() {
  size(800,800); //Tamanho da Tela
  
  //fullScreen(); //Descomente essa linha, caso queira tela cheia
  frameRate(25); //FPS = 25
  oscP5 = new OscP5(this,49162); //Inicializando o objeto na porta 49162
  background(0); //Tela preta
}

//Função recorrente que desenha na tela
void draw() {
  if (osc_recebido) desenha_bolinha(bolinha_x,bolinha_y);//Recebeu e leu o OSC? Então desenhar.
  if (count >= 500){ //Atingiu 500 bolinhas? Então limpar a tela.
    background(0); //Limpando a tela
    count = 0; //Reiniciando a contagem
  }
}

//Função que desenha bolinhas na tela
//nas coordenadas recebidas de X e Y
void desenha_bolinha(int x, int y){
  colorMode(RGB, 255);
  noStroke();
  fill(R,G,B);
  ellipse(x, y, raio, raio);
  count++;
  println("Bolinha:"+count);
  osc_recebido = false; //Esperando o próximo recebimento
}

/* A função abaixo trata o evento de recebimento de mensagens OSC
ela verifica o endereço, verifica o formato.
Então, caso estejam corretos, ela recebe os valores
converte e atrubui às variáveis */

void oscEvent(OscMessage theOscMessage) {
  /*print("### Mensagem osc recebida --> ");
  print(" Endereço: "+theOscMessage.addrPattern());
  println(" Formato: "+theOscMessage.typetag());*/
  
  //Verificando o endereço "/piscou"
  if(theOscMessage.checkAddrPattern("/piscou")==true){
    //Verificando se a mensagem tem seis floats
    if(theOscMessage.checkTypetag("ff")) {
      
      /*Diferente do código base anterior (para PD)
      Os valores recebidos (EAR de cada olho)
      não estão sendo utilizados aqui
      visto que preferi randomizar tudo*/
      
      bolinha_x = int(random(0,800));
      bolinha_y = int(random(0,800));
      R = int(random(0,255));
      G = int(random(0,255));
      B = int(random(0,255));
      c_x = color(R,G,B); //Montando um vetor com as cores
      raio = int(random(15,50));
      osc_recebido = true; //Indicando recebimento
      return;
    }
  }
}
