class Network {
  double[][][] weights;
  double[][] bias;
  double[][] neurons; //values
  int layers;
  boolean[] noSigonLayer;
  int[] layerSizes;
  float lastError = 0;
  Network(String layerSizesz, float randomize) {
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
        for (int k = 0; k<layerSizes[i-1]; k++) weights[i][j][k] = random(-randomize, randomize);
        bias[i][j] = random(-randomize, randomize);
      }
    }
    
  }
  
  Network grow(int newNeurons, float randomize) {
    randomize = abs(randomize);
    Network temp = new Network(nstr + "," + newNeurons, randomize);
    
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
  
  Network backGrow(int newNeurons, float randomize) {
    randomize = abs(randomize);
    Network temp = new Network(newNeurons + "," + nstr, randomize);
    
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
    for (int i = 0; i<neurons[0].length; i++) neurons[thinkFrom][i] = inputs[i];
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
        if (noSigonLayer[i]) {error[i][j] = sum*neurons[i][j];} else {error[i][j] = sum*dsigmoid(neurons[i][j]);}
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
          error[layerSizes.length-1][i]=inError[i]*dsigmoid(neurons[layers-1][i]);
    }
    
    for (int i = layers-2; i>0; i--) {
      for (int j = 0; j<layerSizes[i]; j++) {
        double sum = 0;
        for (int k = 0; k< error[i+1].length; k++) {
          sum += weights[i+1][k][j] * error[i+1][k];
        }
        if (noSigonLayer[i]) {error[i][j] = sum*neurons[i][j];} else {error[i][j] = sum*dsigmoid(neurons[i][j]);}
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
        if (noSigonLayer[i]) {error[i][j] = sum*neurons[i][j];} else {error[i][j] = sum*dsigmoid(neurons[i][j]);}
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
  
}
  
float sign(float in) {if (in>0) {return 1;} else {return -1;}}

double sigmoid(double input) { return (float(1) / (float(1) + Math.pow(2.71828182846, -input))); }
double dsigmoid(double input) { return (input*(1-input)); }
