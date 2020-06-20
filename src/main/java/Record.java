package main.java;

public class Record {

	
	private String dataset;		//name of the project	
	private int trainRel;		//number of releases used for training
	private float train;		//%training  (data on training / total data)
	private float trainDef;		//%defective in training
	private float testDef;		//%defective in testing
    private String classifier;	//name of the classifier
    private String balancing;	//name of the balancing
	private String featureSel;	//feature selection
	private double tp;				//true positive
	private double fp;				//false positive
	private double tn;				//true negative
	private double fn;				//false negative
	private double precision;	//precision
	private double recall;		//recall
	private double roc;			//roc area
	private double kappa;		//kappa
	
	
	public Record(String dataset, int trainRel, String featureSel, String balancing) {
	
		this.dataset = dataset;
		this.trainRel = trainRel;
		this.train = 0;
		this.trainDef = 0;
		this.testDef = 0;
		this.classifier = null;
		this.balancing = balancing;
		this.featureSel = featureSel;
		this.tp = 0;
		this.fp = 0;
		this.tn = 0;
		this.fn = 0;
		this.precision = 0;
		this.recall = 0;
		this.roc = 0;
		this.kappa = 0;
		
	}


	public String getDataset() {
		return dataset;
	}


	public void setDataset(String dataset) {
		this.dataset = dataset;
	}


	public int getTrainRel() {
		return trainRel;
	}


	public void setTrainRel(int trainRel) {
		this.trainRel = trainRel;
	}


	public float getTrain() {
		return train;
	}


	public void setTrain(float train) {
		this.train = train;
	}


	public float getTrainDef() {
		return trainDef;
	}


	public void setTrainDef(float trainDef) {
		this.trainDef = trainDef;
	}


	public float getTestDef() {
		return testDef;
	}


	public void setTestDef(float testDef) {
		this.testDef = testDef;
	}


	public String getClassifier() {
		return classifier;
	}


	public void setClassifier(String classifier) {
		this.classifier = classifier;
	}


	public String getBalancing() {
		return balancing;
	}


	public void setBalancing(String balancing) {
		this.balancing = balancing;
	}


	public String getFeatureSel() {
		return featureSel;
	}


	public void setFeatureSel(String featureSel) {
		this.featureSel = featureSel;
	}


	public double getTp() {
		return tp;
	}


	public void setTp(double tp) {
		this.tp = tp;
	}


	public double getFp() {
		return fp;
	}


	public void setFp(double fp) {
		this.fp = fp;
	}


	public double getTn() {
		return tn;
	}


	public void setTn(double tn) {
		this.tn = tn;
	}


	public double getFn() {
		return fn;
	}


	public void setFn(double fn) {
		this.fn = fn;
	}


	public double getPrecision() {
		return precision;
	}


	public void setPrecision(double precision) {
		this.precision = precision;
	}


	public double getRecall() {
		return recall;
	}


	public void setRecall(double recall) {
		this.recall = recall;
	}


	public double getRoc() {
		return roc;
	}


	public void setRoc(double roc) {
		this.roc = roc;
	}


	public double getKappa() {
		return kappa;
	}


	public void setKappa(double kappa) {
		this.kappa = kappa;
	}

}
