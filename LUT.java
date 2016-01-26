package ca.ubc.ece.lut;

import java.io.File;
import java.util.Random;
import java.io.IOException;
import java.math.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;

import robocode.RobocodeFileOutputStream;

/*
 * Lookup table class for reinforcement learning 
 * ECE 592, Alison Michan, 2010
 */

public class LUT {
	/*
	 * Public attributes of this class. Used to capture the largest Q
	 */
	public static int stateMap [][][][];
	public static int numStates=0;
	public static int numActions = 5;	
	public static final int NumTargetDistance = 4; //hardcoded in robot
	public static final int NumTargetBearing = 4;  //hardcoded in robot
	public static final int NumHeading = 4; //hardcoded in robot
	public static final int isHitWall=2; //hardcoded in robot 0 or 1
	public double QTable[][]; // 2-d array for state table	
	double discountRate;
	double learningRate;
	double upperBoundQ;
	double lowerBoundQ;
	
	/*
	 * Constructor.
	 */
	public LUT(int argNumInputs, int argNumHidden, double argLearningRate, 
			double argAlpha, double argLowerQ,double argUpperQ) {
		//argNumInputs and argNumHidden are disregarded for reinforcement
		discountRate=argAlpha;
		learningRate=argLearningRate;
		upperBoundQ=argUpperQ;
		lowerBoundQ=argLowerQ;	
		stateMap= new int[NumHeading][NumTargetDistance][NumTargetBearing][isHitWall];
		int count = 0;
	    for (int a = 0; a < NumHeading; a++)
	      for (int b = 0; b < NumTargetDistance; b++)
	        for (int c = 0; c < NumTargetBearing; c++)
	        	for (int d = 0; d <isHitWall ; d++){
	          stateMap[a][b][c][d] = count++;
	          System.out.println(a + "\t" +b+ "\t"+c+"\t"+d);
	        	}
	    numStates= count;    
	    QTable= new double[numStates][numActions];	    
		initializeLUT();	
	    System.out.println(numActions);
	    System.out.println(numStates);
	    System.out.println("exit constructor");
	}

	/**
	 * Initialize the look-up table to all zeros.
	 */
	private void initializeLUT() {
		for(int a=0;a<numStates;a++)
			for(int b=0; b<numActions; b++)
				QTable[a][b]=0.0;
		System.out.println("exit initialize");
	}
	
	/**
	 * Returns the QValue of a state/action pair
	 */
	public double getQValue(int state, int action)
	{
		double temp=QTable[state][action];
		return temp;  
	}
	/**
	 * Returns the number of states
	 */
	public int getNumStates()
	{
		return numStates;  
	}
	/**
	 * Returns the number of actions
	 */
	public int getNumActions()
	{
		return numActions;  
	}
	/**
	 * Returns the max QValue of a state (not a state vector)
	 */
	public double getMaxQValue(int argState)
	  {
	    double Qtemp = Double.NEGATIVE_INFINITY;
	    for (int i = 0; i < numActions; i++)
	    {
	      if (QTable[argState][i] > Qtemp)
	        Qtemp = QTable[argState][i];
	    }
	    return Qtemp;	    
	  }
	
	/**
	 * A helper method that translates a vector being used to index the look up
	 * table - returns state number from state vector input
	 */
	public int indexFor(int[] argVector) {		
		int [] stateVector=argVector;
		int state=0 ;
		state+=stateVector[0]*32;
		state+=stateVector[1]*8;
		state+=stateVector[2]*2;
		state+=stateVector[3]*1;
		
		return state;
	}

	/**
	 * Retrieves the value stored in that location of the 
	 * look up table that output Q value?
	 * Returns best action, for Qmax
	 */
	public int outputFor(int[] argStateVector) {		
		int stateTemp=indexFor(argStateVector);
		//find the Qmax value by finding all Q values of the action pairs
		int actionTemp=0;
		double Qtemp=Double.NEGATIVE_INFINITY;
		for(int i=0; i<numActions; i++){
			if(QTable[stateTemp][i]>Qtemp) {
				Qtemp=QTable[stateTemp][i];
				actionTemp=i;
			}
		}
		
		return actionTemp; //return best action (maxQ)
	}

	/**
	 * Will replace the value currently stored in the 
	 * location of the look up table
	 */
	public void train(int[] argCurrentStateVector, int argCurrentAction, double argReward) {
		int currentState=this.indexFor(argCurrentStateVector);
		double oldQ=this.getQValue(currentState,argCurrentAction);		
		double temp = learningRate*(argReward + discountRate * this.getMaxQValue(currentState)-oldQ);
		temp=oldQ+temp;
		if(temp>upperBoundQ) temp=upperBoundQ;
		if(temp<lowerBoundQ) temp=lowerBoundQ;
		QTable[currentState][argCurrentAction]=temp;
	}	
  
	/**
	 * Will attempt to write only the 'visited' elements of the look up table
	 */
	public void writeToFile(File fileHandle) {


		PrintStream saveFile = null;
		
		try 
		{	
			saveFile = new PrintStream( new RobocodeFileOutputStream( fileHandle ));
		}
		catch (IOException e) 
		{
			System.out.println( "*** Could not create output stream for LUT save file.");
		}
		
		//saveFile.println( maxIndex );
		int numEntriesSaved = 0;
		for (int i=0; i<numStates; i++)
		{
			for(int j=0; j<numActions; j++)
			{
				saveFile.println( i +"\n"+ getQValue(i, j) );
				numEntriesSaved ++;
			}
		}
		saveFile.close();
		System.out.println ( "--+ Number of LUT table entries saved is " + numEntriesSaved );
	}
	
	public void replace(LUT argLookupQPrevious){		
		for(int a=0;a<numStates;a++)
			for(int b=0; b<numActions; b++)
				QTable[a][b]=argLookupQPrevious.getQValue(a,b);				
	}
}// End of public class NeuralNet
