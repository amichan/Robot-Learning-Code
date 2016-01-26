package ca.ubc.ece.backprop;

import java.io.IOException;

public class BPDriver {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

		for(int test=0; test<1; test++){
		NeuralNet Trial = new NeuralNet(9,20,0.4,0.9); 
		//numInputs=2, numHidden=4, learningRate=0.2, momentum=0
		double error=0.0;
		double  outputCurr=0.0;
		double X[]=new double[10]; //vector input to neural net
		double Target=0;;
		int numEpochs = 0;
		double RMSError=0.0;
			
		try {
			NeuralNet.loadLUT("/Users/amichan/Documents/workspace/backprop/bin/ca/ubc/ece/lut/Daphne.data/TrainDataP.csv");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		//NeuralNet.initializeTrainingDataBinary(); //load xor training data
		do{
			RMSError=0.0;	
			for(int trainingPairNum=0; trainingPairNum<128; trainingPairNum++)
			{
				for(int a=0;a<5;a++){ //load training inputs
					X=Trial.indexState(trainingPairNum,a);
					//for(int i=0; i<10; i++){
					//	System.out.print(X[i]+"\t");
					//}
					//System.out.print(trainingPairNum+"\t" + a +"\t");
					
					Target=Double.valueOf((Trial.QTableNum[trainingPairNum][a]).trim()).doubleValue();						
					outputCurr=Trial.outputFor(X);
					error=Trial.train(X, Target);			
					RMSError=RMSError+0.5*error*error;
					
			
				//System.out.print("\terrorCurr:\t");
				//System.out.print(error);
				//System.out.print("\toutputCurr:\t");
				//System.out.print(outputCurr);
				//System.out.print("\tTarget:\t");
				//System.out.println(Target);	
				}
			}						
			numEpochs++;
			if(numEpochs>990000)
			{
				RMSError=0.004;
			}
			if(numEpochs==2)System.out.print(numEpochs+"\t" +RMSError+ "\n");
			if(numEpochs%100==0) System.out.print(numEpochs+"\t"+RMSError+"\n");
				
		}while(RMSError>0.04608);//(RMSError>0.05);
		System.out.println("numEpochs:\t"+numEpochs);
		
		}
	}  
}
