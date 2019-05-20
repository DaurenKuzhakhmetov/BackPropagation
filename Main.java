import java.util.Random;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        
        Random random = new Random();
        double[][] dataSet = new double[70000][3];		
     	for (int i = 0; i < dataSet.length; i++) {
           
	  //  int a = 1;
          // int b = 0;	    
              int a = (int) (2 * random.nextDouble());
              int b = (int) (2 * random.nextDouble());
	  
            int c = a ^ b;
            dataSet[i][0] = a;
            dataSet[i][1] = b;
             dataSet[i][2] = c; 
                          
	} 
      //  double[][] dataSet = new double[70000][11];  
        
        NeuralNet neuralNet = new NeuralNet(2, 3, 1);
        neuralNet.startTrainingSet(dataSet);
    }

}
