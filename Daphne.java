package ca.ubc.ece.lut; 
import java.awt.Color;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;

import ca.ubc.ece.backprop.NeuralNet;

import robocode.AdvancedRobot;
import robocode.BattleEndedEvent;
import robocode.BulletHitEvent;
import robocode.DeathEvent;
import robocode.HitByBulletEvent;
import robocode.HitRobotEvent;
import robocode.HitWallEvent;
import robocode.RobocodeFileOutputStream;
import robocode.Robot;
import robocode.RoundEndedEvent;
import robocode.ScannedRobotEvent;
import robocode.WinEvent;

/**
 * Daphne Robot - created from Mathew Nelson's sample MyFirstRobot
 * EECE 592 Part 3 - Reinforcement Learning && Neural Net
 * Alison Michan, December 2010 - robot uses reinforcement learning with LUT class
 */
public class Daphne extends AdvancedRobot {
	//declarations
	static double learningRateOrigR=0.1;
	static double learningRateR=0.1;
	static double learningRateNN=0.3;
	static double discount=0.2;
	static double momentum=0.9;
	static double epsilonGreedy=0.15;	//initialize epsilonGreedy	
	static double epsilonGreedyOrig=0.15;
	
	//static LUT lookupQ = new LUT(2, 12, learningRate, 0.1,-1.0, 1.0); //constructor, STATIC	
	//static LUT lookupQPrevious = new LUT(2, 12, 0.15, 0.2,-1.0, 1.0); //constructor, STATIC	
	static NeuralNet Trial = new NeuralNet(9,20,learningRateNN, momentum); //constructor, STATIC
	
	static double []weightHPrevious=new double[21];
    static double [][]weightIHPrevious=new double[10][21];

	double [] stateVector = new double[5]; //hard code numStates
	double [] currentStateVector = new double[5]; //hardcode stateVector	
	double [] stateAction = new double[10]; //hardcode stateVector

	//process variables
	double currBearing=0.0;
	double currHeading=0.0;
	double currDistance=0.0;
	double currRadar=0.0;
	double reward=0.0; //reward values hardcoded
	int isHitWall=0;
	double BReward=0.5;
	double SReward=0.02;
	double BNReward=-0.5;
	double SNReward=-0.02;	
	int currentAction = 1;		//initialize action a'
	int previousAction = 1;		//initialize previous action a
	static boolean trainFlag=false;
	static int numTrainRounds=0;
	static int numTestRounds=0;
	static int numWonRounds=0;
	static int numWonRoundsPrev=0;
	static String saveWins="";
	static String saveWinsTrain="";
	static int total=0;
	double error=0.0;
	double  outputCurr=0.0;
	double Target=0;;

	
	public void run() {
		setBodyColor(Color.pink); 	//set colors
		setGunColor(Color.gray);
		setRadarColor(Color.pink);	
		currentStateVector[0]=1; //bias
		stateVector[0]=1; //bias
		for(int i=1; i<5; i++){
			currentStateVector[i]=0;
			stateVector[i]=0;
		}		
		setAdjustGunForRobotTurn(false);
		setAdjustRadarForGunTurn(true);	
		
		while (true) {						
			double epsilonGreedyTemp = Math.random(); //epsilon greedy
			if(epsilonGreedyTemp<epsilonGreedy) 
			 {
				double tempE=Math.random();
			//	System.out.println(tempE);
				if(tempE<=0.2) {currentAction = 0;}
				if(tempE>0.2 && tempE<=0.4) {currentAction = 1;}
				if(tempE>0.4 && tempE<=0.6) {currentAction = 2;}
				if(tempE>0.6 && tempE<=0.8) {currentAction = 3;}
				if(tempE>0.8) {currentAction = 4;}			
			 }	
			
			doAction(currentAction); //do currentAction a		
			turnRadarRightRadians(6.28); //observe r and stateVector s in onScannedRobot
			stateAction=indexState(currentStateVector, currentAction);//returns (s,a) combined in a vector
			outputCurr=Trial.outputFor(stateAction); //returns Q(s,a)
			Target=outputCurr+learningRateR*(reward + discount * getMaxQ(stateVector)-outputCurr); //Target is newQ, stateVector is (s') and MaxQ found at a'
			if(trainFlag) error=Trial.train(stateAction, Target);	//update NeuralNet weights		
//			System.out.print("\nrewardMain:\t"+reward+"\n");
			reward=0.0;//re-set reward
			isHitWall=0;//re-set isHitWall
			currentStateVector=stateVector; //update currentState s to s'	
			currentAction=getMaxQAction(currentStateVector); //update a to a' from the maxQ action
			
			
			//for(int i=0; i<10; i++) System.out.print(stateAction[i]+" ");
			//System.out.print("\n");
			//System.out.print("\terrorCurr:\t");
			//System.out.print(error);
			//System.out.print("\toutputCurr:\t");
			//System.out.print(outputCurr);
			//System.out.print("\tTarget:\t");
			//System.out.println(Target);			
		}
	}	
	public void onScannedRobot(ScannedRobotEvent e) {	
		stateVector[0]=1;//bias
		stateVector[1]=getHeading(e.getHeadingRadians());
		currHeading=getHeadingRadians();
		stateVector[2]=getTargetDistance(e.getDistance());
		currDistance=e.getDistance();
		stateVector[3]=getTargetBearing(e.getBearingRadians());	
		currBearing=e.getBearingRadians();
		currRadar=getRadarHeadingRadians();	
		stateVector[4]=isHitWall;	
		//System.out.println(stateVector);		
	}
	
	//public void onHitRobot(HitRobotEvent e) {
	//	reward=reward+SNReward;
	//}
	
	public void onHitByBullet(HitByBulletEvent e) {
		reward=reward+SNReward;
	}
	public void onBulletHit(BulletHitEvent e) {
		reward=reward+SReward;
		//System.out.print("\nreward:\t"+reward+"\n");
	}
	public void onHitWall(HitWallEvent e)
	  {
	    reward=reward+SNReward/2;
	    isHitWall=1;
	  }	
	public void onDeath(DeathEvent e) {
		reward=reward+BNReward;	
		outputCurr=Trial.outputFor(stateAction); //returns Q(s,a)
		Target=outputCurr+learningRateR*(reward + discount * getMaxQ(stateVector)-outputCurr); //Target is newQ, stateVector is (s') and MaxQ found at a'
		if(trainFlag) error=Trial.train(stateAction, Target);	//update NeuralNet weights	
		//System.out.print("\nreward:\t"+reward+"\n");
		reward=0;
		isHitWall=0;
	}
	public void onWin(WinEvent e) {
		reward=reward+BReward;	
		outputCurr=Trial.outputFor(stateAction); //returns Q(s,a)
		Target=outputCurr+learningRateR*(reward + discount * getMaxQ(stateVector)-outputCurr); //Target is newQ, stateVector is (s') and MaxQ found at a'
		if(trainFlag) error=Trial.train(stateAction, Target);	//update NeuralNet weights
		//System.out.print("\nreward:\t"+reward+"\n");
		reward=0;
		isHitWall=0;
		numWonRounds++;
	}
	
//---------Helper Functions-------------------
	
	public void assignWeightsPrev(){
		for(int i=0; i<=20; i++)
			weightHPrevious[i]=Trial.getWeightHO(i);
	    for(int j=0; j<10; j++)
	    	for(int k=0; k<20; k++)
	    		weightIHPrevious[j][k]=Trial.getWeightIH(j,k);
	}
	public void updateWeightsPrev(){
		for(int i=0; i<=20; i++)
			Trial.setWeightHO(i,weightHPrevious[i]);
	    for(int j=0; j<10; j++)
	    	for(int k=0; k<20; k++)
	    		Trial.setWeightIH(j,k,weightIHPrevious[j][k]);
		
	}
	
	
	public void doAction(int argAction)
	{
		if(argAction==0) //aim fire
	    {		
			double temp = currBearing;
			if(temp>0) turnLeftRadians(-1*temp);
			if(temp<0) turnRightRadians(temp);
			fire(1);
	    }
		if(argAction==1) //chase
		{
			double temp = currBearing;
			if(temp>0) turnLeftRadians(-1*temp);
			if(temp<0) turnRightRadians(temp);
			ahead(currDistance/2 + 15);					
		}
		if(argAction==2) //spin retreat fire
		{	
			turnLeftRadians(3.14/2); //spin
			back(125);
			fire(1);
		}			
		if(argAction==3) //ahead
		{	
			ahead(50);	
		}
		if(argAction==4) //fire random
		{	
			fire(3);			
		}
	}
	
	public double getMaxQ(double currentStateVector[])
	{
		double maxQ=Double.NEGATIVE_INFINITY;
		for(int i=0; i<5; i++){
			double temp=Trial.outputFor(indexState(currentStateVector, i));
			if(temp>maxQ){ 
				maxQ=temp;
			}
		}
		return maxQ;
	}
	
	public int getMaxQAction(double currentStateVector[]){
		double maxQ=Double.NEGATIVE_INFINITY;
		int action=0;
		for(int i=0; i<5; i++){
			double temp=Trial.outputFor(indexState(currentStateVector, i));
			if(temp>maxQ){ 
				maxQ=temp;
				action=i;
			}
		}
		return action;	
	}
	
	/*
	 * input is state vector and action number
	 * returns state action vector for use in neural net
	 */
	
	public double [] indexState(double[]state, int action){
		
		double [] SA = new double[10];
		for(int i=0; i<10; i++)
			SA[i]=-1;
		for(int i=0; i<5; i++)
			SA[i]=state[i];
		if(action==0) SA[5]=1;
		if(action==1) SA[6]=1;
		if(action==2) SA[7]=1;
		if(action==3) SA[8]=1;
		if(action==4) SA[9]=1;
		
		return SA;
	}
	public int getHeading(double arg)
	{
		//4 
		int temp=0;
		if (arg>=0 && arg<(Math.PI/2)) temp=0;
		if (arg>=Math.PI/2 && arg<(Math.PI)) temp=1;
		if (arg>=Math.PI && arg<(Math.PI*3/2)) temp=2;
		if (arg>=(Math.PI*3/2)) temp =3;
		return temp;
	}
	public int getTargetDistance(double arg)
	{	//4 close, near, far, really far
		int temp=(int)(arg/100);
		if(temp>3) temp=3;
		return (temp);
	}
	public int getTargetBearing(double arg)
	{
		//4 
		int temp=0;
		arg=arg+Math.PI;
		if (arg>=0 && arg<(Math.PI/2)) temp=0;
		if (arg>=Math.PI/2 && arg<(Math.PI)) temp=1;
		if (arg>=Math.PI && arg<(Math.PI*3/2)) temp=2;
		if (arg>=(Math.PI*3/2)) temp =3;
		return temp;
	}
		
	public void onRoundEnded(RoundEndedEvent e){
		
		System.out.println("trainFlag:"+trainFlag+" numTrainRounds:" + numTrainRounds + " numTestRounds:" 
				+ numTestRounds +" numWon" +numWonRounds+ "Previous" +numWonRoundsPrev
				+ "Greedy" +epsilonGreedy + "totalRounds" +total+ "alpha" +learningRateR
				+"NN_Learning" +Trial.learningRate+"NN_M" +Trial.momentumWeight);	
		System.out.print("\nSaveWinsTrain: "+saveWinsTrain);		
			
		if(trainFlag) {
			if(numTrainRounds==100) 
			{ //alternate between training and no training in multiples
				numTrainRounds=0;				
				System.out.print("inside decision: numWon: "+numWonRounds+" numPrev: "+numWonRoundsPrev);
				//use if want to only accept 'improved' q-table
				if(numWonRoundsPrev>numWonRounds){ 
					//revert weights		
					updateWeightsPrev();				
					System.out.print("\ninside prev if \n");
					//do not update numWonRoundsPrev
					}	
				else{ 
					//keep new weights in previous holder
					assignWeightsPrev();									
					numWonRoundsPrev=numWonRounds; //update numMaxWins
					System.out.print("\ninside else\n");
					}							
				saveWinsTrain=saveWinsTrain+Integer.toString(numWonRounds);
				saveWinsTrain=saveWinsTrain+"\t";	
				numWonRounds=0;
				total=total+100;
				trainFlag = false;
				System.out.print("\nTrial\t"+Trial.getWeightHO(3)+"\t"+Trial.getWeightHO(4)+"\t"+Trial.getWeightHO(5)+"\n");
				System.out.print("Previous\t"+weightHPrevious[3]+"\t"+weightHPrevious[4]+"\t"+weightHPrevious[5]+"\n");
	
		
				
				
			} else{
				numTrainRounds++;
				System.out.print("\nTrial\t"+Trial.getWeightHO(3)+"\t"+Trial.getWeightHO(4)+"\t"+Trial.getWeightHO(5)+"\n");
				System.out.print("Previous\t"+weightHPrevious[3]+"\t"+weightHPrevious[4]+"\t"+weightHPrevious[5]+"\n");
	
				//use to exponentially decay epsilonGreedy and/or learning rate
				//if(epsilonGreedy>0.08) epsilonGreedy=epsilonGreedyOrig*Math.exp(-total/100);
				//if(learningRateR>0.08) learningRateR=learningRateOrigR*Math.exp(-total/100);
				//if(learningRateR<0.08) learningRateR=0.08;
				//if(epsilonGreedy<0.08) epsilonGreedy=0.08;
			}		
		} else {
			if(numTestRounds==5) { //alternate between training and no training in multiples
				numTestRounds = 0;
				//use if want to only accept 'improved' q-table
				//if(numWonRoundsPrev>numWonRounds){ 
				//	Trial.replace(TrialPrevious);
				//	}	
				//else{ 
				//	TrialPrevious.replace(Trial);
				//	numWonRoundsPrev=numWonRounds;
				//	}			
				//saveWins=saveWins+Integer.toString(numWonRounds);
				//saveWins=saveWins+"\t";	
				
				//numWonRounds=0;
				trainFlag = true;			
			} else{
				numTestRounds++;
				//total++;			
			}
			
		}
	}
	
	/**
	 * At the end of a battle, write q-table to file as well as number of wins stats
	 */
	
	public void onBattleEnded(BattleEndedEvent e){
		System.out.println("Battle ended!");
		Trial.save(getDataFile("log.csv"));
		
		PrintStream writer;
		try {
			writer = new PrintStream(new RobocodeFileOutputStream(getDataFile("wins.csv")));
			writer.append(saveWins +"\n" + saveWinsTrain);
		    writer.close();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

	}
}	
