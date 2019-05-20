import java.util.Random;
import java.util.Arrays;

   public class NeuralNet{   
     private NeuralNet(){}

    private double[] inputLayer;
    private double[] hiddenLayer;
    private double[] secondHiddenLayer;
    private double[] outputLayer;
    private double[] expectedOutput;
    private double[][] weightsIH;
    private double[][] weightsHO;
    private double[][] weightsHH;    
//    private static String text;

    private final static double LEARNING_RATE = 0.1;

     public NeuralNet(int sizeInput,int sizeHidden, int sizeOutput){
       inputLayer = new double[sizeInput+1];
       inputLayer[sizeInput] = 1.0;  // нейрон смещения  
       hiddenLayer = new double[sizeHidden+1];
       hiddenLayer[sizeHidden] = 1.0; // нейрон смещения
       expectedOutput = new double[sizeOutput];	   
       outputLayer = new double[sizeOutput];
       secondHiddenLayer = new double[4];
       secondHiddenLayer[3] = 1.0; // нейрон смещения
        

        weightsHH = new double[hiddenLayer.length-1][secondHiddenLayer.length];
        weightsIH = new double[secondHiddenLayer.length-1][inputLayer.length];
        weightsHO = new double[sizeOutput][hiddenLayer.length];	

       for(int i=0;i<weightsHH.length;i++){
          for(int j=0;j<weightsHH[i].length;j++){
	      weightsHH[i][j] = getRandom();
	  }
       }

       for(int i=0;i<weightsIH.length;i++){
          for(int j=0;j<weightsIH[i].length;j++){
	       weightsIH[i][j] = getRandom();
	  }
       }
      
       for(int i=0;i<weightsHO.length;i++){
          for(int j=0;j<weightsHO[i].length;j++){
	       weightsHO[i][j] = getRandom();
	  }
       }
       
     }

        public void startTrainingSet(double[][] dataset){
          for(int i=0;i<dataset.length;i++){
	    System.out.println("Iteration "+i);
  
   	    for (int j = 0; j < inputLayer.length-1; j++) {
                inputLayer[j] = dataset[i][j];
            }
            for(int j=0;j<expectedOutput.length;j++){
                expectedOutput[j] = dataset[i][inputLayer.length-1+j];
            }
            feedForward();
	    showError();
	   
            System.out.println("InputLayer: "+Arrays.toString(inputLayer));
       
            System.out.println("ExpectedOutput: " + Arrays.toString(expectedOutput));
       	  
            System.out.println("OutputLayer: "+Arrays.toString(outputLayer));
            Error obj =  backPropagation();
	    updateWeights(obj);
	  }                           
           
	
	}
      public void feedForward(){      
        for(int i=0;i<secondHiddenLayer.length-1;i++){
           double counter=0.0;		
	   for(int j=0;j<inputLayer.length;j++){
	        counter+=inputLayer[j]*weightsIH[i][j];   
	   }
	   secondHiddenLayer[i] = getSigmoid(counter);
	}
        
        for(int i=0;i<hiddenLayer.length-1;i++){
	   double counter = 0.0;
	   for(int j=0;j<secondHiddenLayer.length;j++){
	        counter+=secondHiddenLayer[j]*weightsHH[i][j];
	   }
	   hiddenLayer[i] = getSigmoid(counter);
	  
	}

	 for(int i=0;i<outputLayer.length;i++){
	     double counter = 0.0;
	     for(int j=0;j<hiddenLayer.length;j++){
	        counter+=hiddenLayer[j]*weightsHO[i][j];
	     }
	     outputLayer[i] = getSigmoid(counter);
	 }
           
      }
         
    
      public Error backPropagation(){
         double[] errorsOutputLayer = getError(); 
         double[] errorsHiddenLayer = new double[hiddenLayer.length-1];
	 for(int i=0;i<errorsHiddenLayer.length;i++){
            double error = 0.0;		 
	     for(int j=0;j<outputLayer.length;j++){
	         error+=(errorsOutputLayer[j]*getDerivativeSigmoid(outputLayer[j])*weightsHO[j][i]);
		
	     }
	     errorsHiddenLayer[i] = error;
	 }
         double[] errorsSecondHiddenLayer = new double[secondHiddenLayer.length-1];
         for(int i=0;i<secondHiddenLayer.length-1;i++){
	     double error = 0.0;
	     for(int j=0;j<hiddenLayer.length-1;j++){
	                 error+=errorsHiddenLayer[j]*getDerivativeSigmoid(hiddenLayer[j])*weightsHH[j][i];

	     }
	    errorsSecondHiddenLayer[i] = error; 
	 }	 
            
	 Error obj = new Error(errorsHiddenLayer,errorsOutputLayer,errorsSecondHiddenLayer);
	 return obj;
      }

      public void updateWeights(Error obj){
          double[] errorsOutputLayer = obj.getOutputErrors();
	  double[] errorsHiddenLayer = obj.getHiddenErrors();
	  double[] errorsSecondHiddenLayer = obj.getSecondHiddenErrors();
	 for(int i=0;i<weightsHO.length;i++){
	    for(int j=0;j<weightsHO[i].length;j++){
	           weightsHO[i][j] = weightsHO[i][j]+LEARNING_RATE*errorsOutputLayer[i]*getDerivativeSigmoid(outputLayer[i])*hiddenLayer[j];

	    }
	 }  
         
	 for(int i=0;i<weightsHH.length;i++){
	    for(int j=0;j<weightsHH[i].length;j++){
	    weightsHH[i][j] = weightsHH[i][j]+LEARNING_RATE*errorsHiddenLayer[i]*getDerivativeSigmoid(hiddenLayer[i])*secondHiddenLayer[j];
	   
	    }
	 }


	 for(int i=0;i<weightsIH.length;i++){
	    for(int j=0;j<weightsIH[i].length;j++){
	weightsIH[i][j] = weightsIH[i][j]+LEARNING_RATE*errorsSecondHiddenLayer[i]*getDerivativeSigmoid(secondHiddenLayer[i])*inputLayer[j];
	    }
	 }
      
      }
  
      private double[] getError(){
        double[] errors = new double[outputLayer.length];	      
       for(int i=0;i<outputLayer.length;i++){	      
	 errors[i] = (expectedOutput[i]-outputLayer[i]);      
       }
       return errors;
      }
      
      private void showError(){
       double error = 0.0;      
        for(int i=0;i<outputLayer.length;i++){
	   error+= Math.pow((expectedOutput[i]-outputLayer[i]),2);
	}
    	System.out.println("Error: "+error);
       //	return error;
      // System.out.println("Error: "+error);	
      }
      private double getSigmoid(double x){
          //double sigmoid = 1/(1+Math.pow(Math.E,-x));
	   double sigmoid = 1.0/(1.0+ (Math.exp(-x)));
           return sigmoid;
      }


      private double getDerivativeSigmoid(double x){
            double derivative = (1-x)*x;  
         // double derivative = getSigmoid(x)*(1-getSigmoid(x));
	  return derivative;
      }

      private double getRandom(){
         Random random = new Random();
	 double weight = random.nextDouble();
	 return weight;
      }

      class Error{
        private double[] hiddenErrors;
	private double[] outputErrors;
	private double[] secondHiddenErrors;
               public Error(double[] hidden,double[] output,double[] second){
	          this.hiddenErrors = hidden;
		  this.outputErrors = output;
		  this.secondHiddenErrors = second;
	       }
        public double[] getHiddenErrors(){ return hiddenErrors;}
        public double[] getOutputErrors(){ return  outputErrors;}
        public double[] getSecondHiddenErrors(){return secondHiddenErrors;};	
      }
   }
