
class Network {
  double[][][] weights;
  double[][] bias;
  double[][] neurons; //values
  int layers;
  String nstr;
  boolean[] noSigonLayer;
  int[] layerSizes;
  float lastError = 0;
  Network(String layerSizesz, float randomize, float x) {
    nstr = layerSizesz;
    String[] layerSizesstr = split(layerSizesz, ',');
    layers = layerSizesstr.length;
    layerSizes = new int[layerSizesstr.length];
    for (int i = 0; i<layerSizesstr.length; i++) {
      layerSizes[i] = int(layerSizesstr[i]);
    }
    printArray(layerSizes);
    randomize = abs(randomize);
    noSigonLayer = new boolean[layers];
    
    weights = new double[layerSizes.length][][]; //Make the network the right size.
    neurons = new double[layerSizes.length][];
    bias = new double[layerSizes.length][];
    neurons[0] = new double[layerSizes[0]];
    for (int i = 1; i<layerSizes.length; i++) {
      noSigonLayer[i] = false;
      weights[i] = new double[layerSizes[i]][layerSizes[i-1]];
      neurons[i] = new double[layerSizes[i]];
      bias[i] = new double[layerSizes[i]];
    }
    
    //randomise Weights
    for (int i = 1; i<layerSizes.length; i++) {
      for (int j = 0; j<layerSizes[i]; j++) {
        for (int k = 0; k<layerSizes[i-1]; k++) weights[i][j][k] = random(-randomize, randomize) + x;
        bias[i][j] = random(-randomize, randomize);
      }
    }
    
  }
  
  Network grow(int newNeurons, float randomize, float x) {
    randomize = abs(randomize);
    Network temp = new Network(nstr + "," + newNeurons, randomize, x);
    
    for (int i = 1; i<temp.layers-1; i++) {
      for (int j = 0; j<temp.layerSizes[i]; j++) {
        for (int k = 0; k<temp.layerSizes[i-1]; k++) {
          temp.weights[i][j][k] = weights[i][j][k];
        }
        temp.bias[i][j] = bias[i][j];
      }
      temp.noSigonLayer[i] = noSigonLayer[i];
    }
    
    return temp;
  }
  
  Network backGrow(int newNeurons, float randomize, float x) {
    randomize = abs(randomize);
    Network temp = new Network(newNeurons + "," + nstr, randomize, x);
    
    for (int i = 2; i<temp.layers; i++) {
      for (int j = 0; j<temp.layerSizes[i]; j++) {
        for (int k = 0; k<temp.layerSizes[i-1]; k++) {
          temp.weights[i][j][k] = weights[i-1][j][k];
        }
        temp.bias[i][j] = bias[i-1][j];
      }
      temp.noSigonLayer[i] = noSigonLayer[i-1];
    }
    
    return temp;
  }
  
  double[] think(double inputs[]) {
    return thinkExt(inputs, 0, layers-1);
  }
  
  double[] thinkExt(double inputs[], int thinkFrom, int thinkTo) {

    for (int i = 0; i<neurons.length; i++) {
      neurons[thinkFrom][i] = inputs[i];
    }
    
    double[] outPuts = new double[layerSizes[thinkTo]];
    for (int i = thinkFrom+1; i<=thinkTo; i++) {//layer
      for (int k = 0; k<layerSizes[i]; k++) {
        neurons[i][k] = bias[i][k];
        for (int j = 0; j<layerSizes[i-1]; j++) neurons[i][k] += weights[i][k][j]*neurons[i-1][j];
        if (noSigonLayer[i]) {} else {neurons[i][k] = sigmoid(neurons[i][k]);}
      }
    }
    
    outPuts = neurons[thinkTo];
    
    return outPuts;
  }
  
  double[] learn(double[] inputs, double[] desiredOutputs, float stepSize) {
    double[] error = new double[desiredOutputs.length];
    double[] played = think(inputs);
    for (int i = 0; i<error.length; i++) {
      double E = (played[i] - desiredOutputs[i]);
      error[i] = E;
    }
    return learnFromError(error, stepSize);
  }
  
  double[] getLayerError(double[] inputs, double[] desiredOutputs, int layerNum, boolean negative) {
    double[][] error = new double[neurons.length][];
    double[] played = think(inputs);
    
    for (int i = 0; i<error.length; i++) {
      error[i] = new double[neurons[i].length];
      for (int j = 0; j<error[i].length; j++) error[i][j] = (double) 0;
    }
    
    for (int i = 0; i<layerSizes[layerSizes.length-1]; i++) {
      error[layerSizes.length-1][i]=(played[i] - desiredOutputs[i])*dsigmoid(neurons[layers-1][i]);
    }
    
    for (int i = layers-2; i>0; i--) {
      for (int j = 0; j<layerSizes[i]; j++) {
        double sum = 0;
        for (int k = 0; k< error[i+1].length; k++) {
          sum += weights[i+1][k][j] * error[i+1][k];
        }
        if (noSigonLayer[i]) {error[i][j] = sum;} else {error[i][j] = sum*dsigmoid(neurons[i][j]);}
      }
    }
    
    double[] out = new double[error[layerNum].length];
    if (layerNum == 0) {
      for (int j = 0; j<layerSizes[0]; j++) {
        double sum = 0;
        for (int k = 0; k<error[1].length; k++) {
          sum += weights[1][k][j] * error[1][k];
        }
        out[j] = sum*inputs[j];
      }
    } else {
      for (int j = 0; j<layerSizes[layerNum]; j++) out[j] = error[layerNum][j];
    }
    if (negative) for (int i = 0; i<out.length; i++) out[i] *= -1;
    
    return out;
  }

  double[] learnFromError(double[] inError, float stepSize) {
    double[][] error = new double[neurons.length][];
    
    for (int i = 0; i<error.length; i++) {
      error[i] = new double[neurons[i].length];
      for (int j = 0; j<error[i].length; j++) error[i][j] = (double) 0;
    }
    
    for (int i = 0; i<layerSizes[layerSizes.length-1]; i++) {
          if (noSigonLayer[layers - 1]) {
            error[layerSizes.length-1][i]=inError[i];
          } else {
            error[layerSizes.length-1][i]=inError[i]*dsigmoid(neurons[layers-1][i]);
          }
    }
    
    for (int i = layers-2; i>0; i--) {
      for (int j = 0; j<layerSizes[i]; j++) {
        double sum = 0;
        for (int k = 0; k< error[i+1].length; k++) {
          sum += weights[i+1][k][j] * error[i+1][k];
        }
        if (noSigonLayer[i]) {error[i][j] = sum;} else {error[i][j] = sum*dsigmoid(neurons[i][j]);}
      }
    }
    
    //tweak weights
    for (int i = 1; i<layers; i++) {
      for (int j = 0; j<neurons[i].length; j++) {
        for (int pj = 0; pj<neurons[i-1].length; pj++) {
          double delta = -stepSize * neurons[i-1][pj] * error[i][j];
          weights[i][j][pj] += delta;
        }
        bias[i][j] += -stepSize * error[i][j];
      }
    }
    
    double[] out = new double[neurons[0].length];
      for (int j = 0; j<layerSizes[0]; j++) {
        double sum = 0;
        for (int k = 0; k<error[1].length; k++) {
          sum += weights[1][k][j] * error[1][k];
        }
        out[j] = sum;
      }
      
      return out;
  }
                                                             //0          1            layers
  void learnExt(double[] inputs, double[] desiredOutputs, int in, int learnFrom, int learnTo, float stepSize) {
    double[][] error = new double[neurons.length][];
    double[] played = thinkExt(inputs, in, layers-1);
    
    for (int i = 0; i<error.length; i++) {
      error[i] = new double[neurons[i].length];
      for (int j = 0; j<error[i].length; j++) error[i][j] = (double) 0;
    }
    float totalError = 0;
    for (int i = 0; i<layerSizes[layerSizes.length-1]; i++) {
      error[layerSizes.length-1][i]=(played[i] - desiredOutputs[i])*dsigmoid(neurons[layers-1][i]);
      totalError += abs((float) error[layerSizes.length-1][i]);
    }
    lastError= totalError/float(layerSizes[layerSizes.length-1]);
    //println(error[layers-1][0]); //<- Print error?
    
    for (int i = layers-2; i>learnFrom; i--) {
      for (int j = 0; j<layerSizes[i]; j++) {
        double sum = 0;
        for (int k = 0; k< error[i+1].length; k++) {
          sum += weights[i+1][k][j] * error[i+1][k];
        }
        if (noSigonLayer[i]) {error[i][j] = sum;} else {error[i][j] = sum*dsigmoid(neurons[i][j]);}
      }
    }
    
    //tweak weights
    for (int i = learnFrom; i<learnTo; i++) {
      for (int j = 0; j<neurons[i].length; j++) {
        for (int pj = 0; pj<neurons[i-1].length; pj++) {
          double delta = -stepSize * neurons[i-1][pj] * error[i][j];
          weights[i][j][pj] += delta;
        }
        bias[i][j] += -stepSize * error[i][j];
      }
    }
    
  }
  
  void saveNet(String name) {
    PrintWriter output;
    for (int i = 1; i<layers; i++) {
      output = createWriter("./" + name + "/layer " + i + ".txt");
      for (int j = 0; j<weights[i].length; j++) {
        output.print(bias[i][j] + "|");
        for (int k = 0; k<weights[i][j].length; k++) {
          output.print(weights[i][j][k] + ",");
        }
        output.println("");
        output.flush();
      }
      output.close();
    }
  }
  
  void loadNet(String name) {
    for (int i = 1; i<layers; i++) {
      String[] lines = loadStrings("./" + name + "/layer " + i + ".txt");
      for (int j = 0; j<weights[i].length; j++) {
        String[] yeeted = split(lines[j], '|');
        bias[i][j] = Double.parseDouble(yeeted[0]);
        String[] smashed = split(yeeted[1], ',');
        for (int k = 0; k<weights[i][j].length; k++) {
          weights[i][j][k] = Double.parseDouble(smashed[k]);
        }
      }
    }
  }
  
}
  
float sign(float in) {if (in>0) {return 1;} else {return -1;}}

double sigmoid(double input) { return (float(1) / (float(1) + Math.pow(2.71828182846, -input))); }
double dsigmoid(double input) { return (input*(1-input)); }

double[] PImagetoDoubleArray(PImage img, boolean incolor) {
  double[] out;
  if (incolor) {out = new double[img.pixels.length*3];} else {out = new double[img.pixels.length];}
  img.loadPixels();
  if (incolor) {
    for (int i = 0; i<img.pixels.length;i++) {
      out[i*3] = red(img.pixels[i])/255;
      out[i*3 + 1] = green(img.pixels[i])/255;
      out[i*3 + 2] = blue(img.pixels[i])/255;
    }
  } else {
    for (int i = 0; i<img.pixels.length;i++) {
      out[i] = (red(img.pixels[i]) + green(img.pixels[i]) + blue(img.pixels[i]))/(3*255);
    }
  }
  img.updatePixels();
  return out;
}

PImage DoubleArraytoPImage(double[] in, int wid, int hei) {
  PImage img;
  boolean incolor = !(wid*hei == in.length);
  img = new PImage(wid, hei);
  img.loadPixels();
  if (incolor) {
    for (int i = 0; i<img.pixels.length;i++) {
      img.pixels[i] = color((float) in[i*3]*255,(float) in[i*3 + 1]*255,(float) in[i*3 + 2]*255);
    }
  } else {
    for (int i = 0; i<img.pixels.length;i++) {
      img.pixels[i] = color((float) in[i]*255);
    }
  }
  img.updatePixels();
  return img;
}

class GAN {
  Network net;
  int disStart;
  GAN(String gen, String add, float ran) { 
    String network = gen + "," + add;
    disStart = split(gen, ',').length;
    net = new Network(network, ran,0);
    net.noSigonLayer[disStart-1] = true;
    println(disStart);
  }
  
  void trainThroughBoth(double[] in, double[] out, float lr) {
    net.learnExt(in, out, 0, 1, net.layers, lr);
  }
  
  void trainAdd(double[] in, double[] out, float lr) {
    net.learnExt(in, out, disStart-1, disStart, net.layers, lr);
  }
  
  void trainGen(double[] in, double[] out, float lr) {
    net.learnExt(in, out, 0, 1, disStart-1, lr);
  }
  
  double[] think(double[] in) {
    return net.thinkExt(in, 0, disStart-1);
  }
  
  double[] judge(double[] in) {
    return net.thinkExt(in, disStart-1, net.layers-1);
  }
  
  double[] judgeSelf(double[] in) {
    return net.thinkExt(in, 0, net.layers-1);
  }
}
