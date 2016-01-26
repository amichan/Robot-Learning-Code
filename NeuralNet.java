package ca.ubc.ece.backprop;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.math.*;
import java.util.StringTokenizer;

import ca.ubc.ece.lut.LUT;

import robocode.RobocodeFileOutputStream;

/*
 * @author Michan
 */

public class NeuralNet {

/*
 *  Private members of this class
 */
	

/*
 * Public attributes of this class. Used to capture the largest Q  
 */
	public static double[] hiddenNeurons; //hidden neuron vector
	public static double[] lastWeightChangeHiddenOutput; //momentum
	public static double [][] lastWeightChangeInputToHidden; //momentum
	public static double []weightHiddenOutput;//= {-0.1401,0.4919,-0.2913,-0.3979,0.3581};//text weights
	public static double [][]weightInputToHidden;//={ {-0.3378, 0.2771, 0.2895, -0.3329},{0.1970,0.3191,-0.1448, 0.3594},{0.3099, 0.1904, -0.0347, -0.4861} };//text weights
	public static double sigma_a=-.1; //sigmoid bound a
	public static double sigma_b=.1; //sigmoid bound b
	public static double gamma=sigma_b-sigma_a; //for custom sigmoid and prime
	public static double n=-sigma_a; //for custom sigmoid and prime
	public static int numHidden;
	public static int numInputs;
	public static double learningRate;
	public static double momentumWeight;
	
	public static int numPatterns;
	public static double[][] trainingDataInput;
	public static double[] trainingDataOutput;

	public static int patNum; //process variable
	public static double y_output;
	
	//this is not great way to do this but ok
	public static int numRows=128;
	public static int numCol=5;
	public static String [][] QTableNum = new String [numRows][numCol];	
	
/* 
 * Constructor.
 */
	 public NeuralNet (int argNumInputs, 
			 int argNumHidden,
			 double argLearningRate,
			 double argMomentum)
	 {
		 numHidden=argNumHidden; 
		 numInputs=argNumInputs+1; //add 1 for bias
		 learningRate=argLearningRate;
		 momentumWeight=argMomentum;
		 lastWeightChangeHiddenOutput=new double[numHidden+1]; //momentum
		 lastWeightChangeInputToHidden=new double [numInputs][numHidden]; //momentum
		 hiddenNeurons=new double[numHidden+1];//add 1 for bias of 1 at [0]
		 numPatterns=128*5; //4xor
		 trainingDataInput = new double[numPatterns][numInputs];	//xor
		 trainingDataOutput= new double[numPatterns]; 
		 y_output=0.0;
		 weightHiddenOutput=new double[numHidden+1];
	     weightInputToHidden=new double[numInputs][numHidden];
		 initializeWeights();
		 System.out.println(numHidden);
		 System.out.println(numInputs);
	 }
	
/*
 * 	 Return a sigmoid of the input X //make private 
 */
	private double sigmoid( double x ) { 
		double temp = Math.exp(-x);
		return (1/(1+temp));
	}	
/* 
 * This method implements a bipolar sigmoid  
 */
	private double customSigmoid( double x ) {
		
		double temp=Math.exp(-x);
		temp=1/(1+temp);
		temp=gamma*temp-n;
		return temp;
		
	}
			 
/* 
 * Initialize the weights to random values.
 */

	private void initializeWeights() { 
		double temp=0.0;
	
		for(int j=0;j<=numHidden;j++)
		{
			temp=(Math.random()-0.5);
			if((Math.abs(temp))<0.15){
				temp=temp*10;
			}
			weightHiddenOutput[j]=temp; //make random
		}			
		for(int j=0;j<numHidden;j++)
		{	
			for(int k=0;k<numInputs;k++)
			{	
				temp=(Math.random()-0.5);	
				if((Math.abs(temp))<0.15)	
				{	
					temp=temp*10;	
				}					
				weightInputToHidden[k][j]=temp; //make random			
			}			
		}
	} 

/* 
 * Computes output of the NN without training. ie a forward pass
 * pass in training pair including bias
 */
	public double outputFor( double[] X) 
	{ 
		hiddenNeurons[0]=1.0; //apply bias of 1 at hiddenNeurons Zo
		for (int j=1; j<=numHidden; j++) //count through hidden neurons not bias (4 in assign 1)
		{			
			hiddenNeurons[j]=0.0;
			for(int i=0;i<numInputs;i++) //count through bias and inputs x1,x2
			{
				hiddenNeurons[j]=hiddenNeurons[j]+(X[i]*weightInputToHidden[i][j-1]);
			}
			hiddenNeurons[j]=customSigmoid(hiddenNeurons[j]);//bipolar
			//hiddenNeurons[j]=sigmoid(hiddenNeurons[j]);//binary
		}		
		y_output=0.0;
		for(int j=0; j<=numHidden; j++)
		{
			y_output=y_output+hiddenNeurons[j]*weightHiddenOutput[j];
		}	
		y_output=(customSigmoid(y_output));//bipolar
		return y_output; 
		//return (sigmoid(y_output));//binary
	}
					 
/*
 * This method is used to update the weights of the neural net. 
 * Returns the error that is the difference of 
 * the target input and the outputFor(inputStateVector)that was updated y_output
 * in call to outputFor just before
 */
	public double train( double[] argInputVector, double argTargetOutput)
	{		
		double errorDelta=0.0;
		double outputError=0.0;
		double hiddenDelta =0.0;
		double[] weightChangeHiddenOutput = new double[numHidden+1]; //momentum
		double[][] weightChangeInputToHidden= new double [numInputs][numHidden]; //momentum

		//backpropagation to hidden layer
		outputError=(argTargetOutput-y_output); //bipolar
	    errorDelta=(outputError*(1/gamma)*(n+y_output)*(gamma-n-y_output));//bipolar
    	//outputError=argTargetOutput-(sigmoid(y_output));//binary/bipolar
		//errorDelta=outputError*sigmoid(y_output)*(1-sigmoid(y_output));//hard coded - switch to bipolar/binary	
		for(int j=0; j<=numHidden; j++) //j=0 is bias, where hiddenNeurons[0]=1
		{
			//calculate weight correction, added momentum here
			weightChangeHiddenOutput[j]=(learningRate*errorDelta*hiddenNeurons[j])+(momentumWeight*lastWeightChangeHiddenOutput[j]); 			
			lastWeightChangeHiddenOutput[j]=weightChangeHiddenOutput[j];
		}
		//backpropagation to input layer	
		for(int j=1; j<=numHidden; j++) //offset since bias=1 term at hiddenNeurons[0]
		{
			hiddenDelta=weightHiddenOutput[j]*errorDelta;
			hiddenDelta=(hiddenDelta*(1/gamma)*(n+hiddenNeurons[j])*(gamma-n-hiddenNeurons[j]));//bipolar
			//hiddenDelta=hiddenDelta*(hiddenNeurons[j])*(1-hiddenNeurons[j]); //binary, sigmoid already applied to hiddenNeuron values
			for(int i=0; i<numInputs; i++) //i=0, X[0] is bias=1
			{
				//calculate weight correction, added momentum here
				weightChangeInputToHidden[i][j-1]=(learningRate*hiddenDelta*argInputVector[i])+(momentumWeight*lastWeightChangeInputToHidden[i][j-1]); 	
				lastWeightChangeInputToHidden[i][j-1]=weightChangeInputToHidden[i][j-1];
			}
		}
		//weight change updates Hidden Output
		for(int j=0; j<=numHidden; j++)
		{					
			weightHiddenOutput[j]=(weightHiddenOutput[j]+weightChangeHiddenOutput[j]); 			
		}		
		//weight change updates InputToHidden	
		for(int j=1; j<=numHidden; j++)
		{
			for(int i=0; i<numInputs; i++)	
			{					
				weightInputToHidden[i][j-1]=(weightInputToHidden[i][j-1]+weightChangeInputToHidden[i][j-1]);				
			}		
		}		
	return outputError;	
	}
	
/*
 * saves the weight arrays to a file. 
 */
	public void save( File argFile){	
		PrintStream writer;
		try {
			 System.out.println(numHidden);
			 System.out.println(numInputs);
			writer = new PrintStream(new RobocodeFileOutputStream(argFile));
			String saveFile="weightInputToHidden\n";
			for(int j=0; j<numHidden; j++){
				saveFile+="\n";
				for(int i=0; i<numInputs; i++){					
					saveFile+=weightInputToHidden[i][j]+"\t";				
				}		
			}					
			String saveFile2="\n\nweightHiddenOutput";
			for(int a=0;a<=numHidden; a++){
				saveFile2+="\n";
				saveFile2+=weightHiddenOutput[a]+"\t";		
			}
			writer.append(saveFile);
			writer.append(saveFile2);
		    writer.close();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
	}		
		
		
	
				 			 
/*
 * loads the LUT Q values from file. format of the file is expected to follow
 */
	
	public static void loadLUT( String argFileName) throws IOException{
		 
		File file = new File(argFileName);
		BufferedReader bufRdr;
		try {
			bufRdr = new BufferedReader(new java.io.FileReader(file));	
		String line = null;
		int row = 0;
		int col = 0;
		//read each line of text file
		while((line = bufRdr.readLine()) != null && row < numRows)
		{		
			StringTokenizer st = new StringTokenizer(line,",");
			while (st.hasMoreTokens())
			{
				//get next token and store it in the array
				
				QTableNum[row][col] = st.nextToken();
				
				col++;
			}
			col = 0;
			row++;
		}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}	
	
		
	}
	
	public static void initializeTrainingDataBinary(){
		//binary
		
		trainingDataInput[0][1]=1;
		trainingDataInput[0][2]=0;
		trainingDataInput[0][0]=1;//bias
		trainingDataOutput[0]=1;
		
		trainingDataInput[1][1]=0;
		trainingDataInput[1][2]=1;
		trainingDataInput[1][0]=1;//bias
		trainingDataOutput[1]=1;
		
		trainingDataInput[2][1]=1;
		trainingDataInput[2][2]=1;
		trainingDataInput[2][0]=1;//bias
		trainingDataOutput[2]=0;
		
		trainingDataInput[3][1]=0;
		trainingDataInput[3][2]=0;
		trainingDataInput[3][0]=1;//bias
		trainingDataOutput[3]=0;
	}
			 
	public static void initializeTrainingDataBipolar(){
		//bipolar
		trainingDataInput[0][1]=1;
		trainingDataInput[0][2]=-1;
		trainingDataInput[0][0]=1;//bias
		trainingDataOutput[0]=1;
		
		trainingDataInput[1][1]=-1;
		trainingDataInput[1][2]=1;
		trainingDataInput[1][0]=1;//bias
		trainingDataOutput[1]=1;
		
		trainingDataInput[2][1]=1;
		trainingDataInput[2][2]=1;
		trainingDataInput[2][0]=1;//bias
		trainingDataOutput[2]=-1;
		
		trainingDataInput[3][1]=-1;
		trainingDataInput[3][2]=-1;
		trainingDataInput[3][0]=1;//bias
		trainingDataOutput[3]=-1;
	}
	
		
	//helper function---------
		
	public double[] indexState(int argState, int argAction) {
			 int length=10;
		double [] stateVector=new double[length];
		for(int i=0; i< length; i++) stateVector[i]=-1;
			 
		stateVector[0]=1;//bias
		stateVector[1]=(argState/32)%4;
		stateVector[2]=(argState/8)%4;
		stateVector[3]=(argState/2)%4;
		stateVector[4]=(argState/1)%2;
		if(argAction==0) stateVector[5]=1;
		if(argAction==1) stateVector[6]=1;
		if(argAction==2) stateVector[7]=1;
		if(argAction==3) stateVector[8]=1;
		if(argAction==4) stateVector[9]=1;	
			 
		
			 return stateVector;
	}
	
	public double getWeightHO(int index){
		return weightHiddenOutput[index];
	}
	public double getWeightIH(int index1, int index2){
		return weightInputToHidden[index1][index2];
	}
	public void setWeightHO(int index, double value){
		weightHiddenOutput[index]=value;
	}
	public void setWeightIH(int index1,int index2, double value){
		weightInputToHidden[index1][index2]=value;
	}
	public void printWeights(){
		for(int i=3; i<8; i++) System.out.print(weightHiddenOutput[i]+"\t");
	}

}//End of public class NeuralNet
				 