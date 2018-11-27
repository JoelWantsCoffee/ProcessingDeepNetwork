class CLayer {
  double[][][] image;
  int W;
  int H;
  int D;
  
  CLayer(PImage og) {
    image = new double[og.width][og.height][3];
    for (int i = 0; i<og.width; i++) {
      for (int j = 0; j<og.height; j++) {
        image[i][j][0] = red(og.get(i,j))/255;
        image[i][j][1] = green(og.get(i,j))/255;
        image[i][j][2] = blue(og.get(i,j))/255;
      }
    }
    W = og.width;
    H = og.height;
    D = 3;
  }
  
  CLayer(int wid, int hei, int dep) {
    W = wid;
    H = hei;
    D = dep;
    image = new double[wid][hei][dep];
    for (int i = 0; i<wid; i++) {
      for (int j = 0; j<hei; j++) {
        for (int k = 0; k<dep; k++) {
          image[i][j][k] = 0;
        }
      }
    }
  }
  
  CLayer(CLayer[] in) {
    W = in[0].W;
    H = in[0].H;
    D = in.length;
    image = new double[W][H][D];
    for (int i = 0; i<in[0].W; i++) {
      for (int j = 0; j<in[0].H; j++) {
        for (int k = 0; k<in.length; k++) {
          image[i][j][k] = in[k].image[i][j][0];
        }
      }
    }
  }
  
  double git(int i, int j, int k) {
    if ((i >= 0) && (i < W)) {
      if ((j >= 0) && (j < H)) {
        if ((k >= 0) && (k < D)) {
          return image[i][j][k];
        } else return 0;
      } else return 0;
    } else return 0;
  }
  
  
  CLayer sig() {
    CLayer out = new CLayer(W, H, D);
    for (int i = 0; i<W; i++) {
      for (int j = 0; j<H; j++) {
        for (int k = 0; k<D; k++) {
          //out.image[i][j][k] = sigmoid(image[i][j][k]);
          out.image[i][j][k] = image[i][j][k];
          if (image[i][j][k] < 0) out.image[i][j][k] = 0;
        }
      }
    }
    return out;
  }
  
  double[] average() {
    double[] out = new double[D];
    for (int k = 0; k<D; k++) {
      for (int j = 0; j<H; j++) {
        for (int i = 0; i<W; i++) {
          out[k] += image[i][j][k];
        }
      }
      out[k] /= W*H;
    }
    return out;
  }
  
  CLayer pool(int factor) {
    CLayer out = new CLayer(floor(W/factor), floor(H/factor), D);
    for (int i = 0; i<W-1; i++) {
      for (int j = 0; j<H-1; j++) {
        for (int k = 0; k<D; k++) {
          out.image[floor(i/factor)][floor(j/factor)][k] += image[i][j][k];
        }
      }
    }
    for (int i = 0; i<out.W; i++) {
      for (int j = 0; j<out.H; j++) {
        for (int k = 0; k<out.D; k++) {
          out.image[i][j][k] /= factor*factor;
        }
      }
    }
    return out;
  }
  
  PImage export() {
    double[] RT = new double[W*H];
    PImage yeet = new PImage(W, H);
    for (int i = 0; i<W; i++) {
      for (int j = 0; j<H; j++) {
        for (int k = 0; k<D; k++) {
          RT[i+j*W] += image[i][j][k]*255;
        }
        yeet.pixels[i+j*W]  = color(round((float)RT[i+j*W]/D));
      }
    }
    return yeet;
  }
}

class Kernal {
  double[][][] weights;
  double bias;
  int radius;
  int wid;
  int depth;
  Kernal(int rad, int dep, float random) {
    random = abs(random);
    radius = rad;
    depth = dep;
    wid = rad*2+1;
    weights = new double[wid][wid][dep];
    for (int i = 0; i<weights.length; i++) {
      for (int j = 0; j<weights.length; j++) {
        for (int k = 0; j<weights.length; j++) {
          weights[i][j][k] = random(random);
        }
      }
    }
  }
  
  CLayer convolve(CLayer in) {
    CLayer out = new CLayer(in.W, in.H, 1);
    for (int i = 0; i < in.W; i++) {
      for (int j = 0; j < in.H; j++) {
        out.image[i][j][0] = operate(in, i, j);
      }
    }
    return out;
  }
  
  void convolveLearn(CLayer in, CLayer error, float lr) {
    for (int i = 0; i < in.W; i++) {
      for (int j = 0; j < in.H; j++) {
        learn(in, error, i, j, lr);
      }
    }
  }
  
  double operate(CLayer in, int i, int j) {
    double rTotal = 0;
    for (int k = 0; k < depth; k++) {
      for (int ii = -radius; ii<=radius; ii++) {
        for (int jj = -radius; jj<=radius; jj++) {
          rTotal += in.git(i+ii,j+jj,k) * weights[ii+radius][jj+radius][k];
        }
      }      
    }
    rTotal /= depth*wid*wid;
    rTotal += bias;
    return rTotal;
  }
  
  void learn(CLayer in, CLayer error, int i, int j, float lr) {
    double RT = 0;
    for (int k = 0; k < depth; k++) {
      for (int ii = -radius; ii<=radius; ii++) {
        for (int jj = -radius; jj<=radius; jj++) {
          weights[ii+radius][jj+radius][k] += in.git(i+ii,j+jj,k) * error.git(i+ii,j+jj,k) * -lr;
          RT += error.git(i+ii,j+jj,k);
        }
      }      
    }
    bias += RT/(wid*depth*wid) * -lr;
  }
}

class ConvNetwork {
  CLayer[] layer;
  Kernal[][] kernals;
  int[] operations;
  int[] locals;
  Network FC;
  /*
  0 kernal
  1 sigmoid
  2 pool
  3 fullyConnected
  */
  ConvNetwork(String in, float random) {
    in = "yeet," + in;
    random = abs(random);
    String[] coms = split(in, ',');
    layer = new CLayer[coms.length];
    operations = new int[coms.length];
    locals = new int[coms.length];
    int krt = 0;
    for (int i = 1; i<coms.length; i++) {
      if (coms[i].charAt(0) == 'k') {
        locals[i] = krt;
        operations[i] = 0;
        krt++;
      } else if (coms[i].charAt(0) == 'p') {
        locals[i] = int(coms[i].substring(1));
        operations[i] = 2;
      } else if (coms[i].charAt(0) == 's') {
        locals[i] = 0;
        operations[i] = 1;
      } else if (coms[i].charAt(0) == 'f') {
        locals[i] = 0;
        operations[i] = 3;
        FC = new Network(split(coms[i-1],'.')[1] + "," + coms[i].substring(1), random);
      }
    }
    kernals = new Kernal[krt][];
    krt = 0;
    operations[0] = -1;
    for (int i = 1; i<coms.length; i++) {
      if (coms[i].charAt(0) == 'k') {
        kernals[krt] = new Kernal[int(split(coms[i], '.')[1])];
        for (int j = 0; j<kernals[krt].length; j++) {
          if (i == 1) {
            kernals[krt][j] = new Kernal(int(split(coms[i], '.')[0].substring(1)), 3, random);
          } else {
            kernals[krt][j] = new Kernal(int(split(coms[i], '.')[0].substring(1)), kernals[krt-1].length, random);
          }
        }
        krt++;
      }
    }
    printArray(operations);
  }
  
  double[] think(PImage in) {
    double[] out = new double[1];
    layer[0] = new CLayer(in);
    for (int i = 1; i<operations.length; i++) {
      if (operations[i] == 0) { //kernal
        CLayer[] stacks = new CLayer[kernals[locals[i]].length];
        for (int j = 0; j<stacks.length; j++) {
           stacks[j] = kernals[locals[i]][j].convolve(layer[i - 1]);
        }
        layer[i] = new CLayer(stacks);
      } else if (operations[i] == 1) { //sigmoid
        layer[i] = layer[i-1].sig();
      } else if (operations[i] == 2) { //pool
        layer[i] = layer[i-1].pool(locals[i]);
      } else if (operations[i] == 3) { //Network
        double[] averaged = layer[i-1].average();
        out = FC.think(averaged);
      }
    }
    return out;
  }
  
  void learn(PImage in, double[] out, float lr) {
    think(in);
    FC.learn(FC.neurons[0], out, lr);
    double[] FCinerror = FC.getLayerError(FC.neurons[0], out, 0, false);
    CLayer[] layerError = new CLayer[layer.length];
    for (int i = 0; i<layer.length-1; i++) {
      layerError[i] = new CLayer(layer[i].W, layer[i].H, layer[i].D);
    }
    for (int k = 0; k<layerError[layerError.length - 2].D; k++) {
      for (int i = 0; i<layerError[layerError.length - 2].W; i++) {
        for (int j = 0; j<layerError[layerError.length - 2].H; j++) {
          layerError[layerError.length - 2].image[i][j][k] = FCinerror[k];
        }
      }
    }
    for (int i = layerError.length - 3; i<0; i--) {
      if (operations[i] == 0) { //kernal
        CLayer[] stacks = new CLayer[kernals[locals[i]].length];
        for (int j = 0; j<stacks.length; j++) {
           stacks[j] = kernals[locals[i]][j].convolve(layerError[i + 1]);
        }
        layerError[i] = new CLayer(stacks);
      } else if (operations[i] == 1) { 
        layerError[i] = layerError[i+1].sig();
      } else if (operations[i] == 2) { //pool
        layerError[i] = layerError[i+1].pool(1/locals[i]);
      }
    }
    
    for (int i = layerError.length - 2; i<0; i--) {
      for (int j = 0; j<kernals[locals[i]].length; j++) {
         kernals[locals[i]][j].convolveLearn(layer[i], layerError[i], lr);
      }
    }
  }
  
  
}

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
  
  double[] think(double inputs[]) {
    return thinkExt(inputs, 0, layers-1);
  }
  double[] thinkExt(double inputs[], int thinkFrom, int thinkTo) {
    for (int i = 0; i<inputs.length; i++) neurons[thinkFrom][i] = inputs[i];
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
  
  void learn(double[] inputs, double[] desiredOutputs, float stepSize) {
    learnExt(inputs, desiredOutputs, 0, 1, layers, stepSize);
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

  void learnFromError(double[] inputs, double[] inError, float stepSize) {
    think(inputs);
    double[][] error = new double[neurons.length][];
    
    for (int i = 0; i<error.length; i++) {
      error[i] = new double[neurons[i].length];
      for (int j = 0; j<error[i].length; j++) error[i][j] = (double) 0;
    }
    
    for (int i = 0; i<layerSizes[layerSizes.length-1]; i++) {
      error[layerSizes.length-1][i]=inError[i];
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
