package main.java;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

import org.json.JSONArray;
import org.json.JSONObject;

import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.instance.Resample;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.supervised.instance.SpreadSubsample;

public class AnalyzeData {

	
    private static Logger logger;
    private static final String TRAINING = "training";
    private static final String TESTING = "testing";
    
    static {

        System.setProperty("java.util.logging.config.file", "logging.properties");
        logger = Logger.getLogger(AnalyzeData.class.getName());
    }
      
    
    //Creates a new arff file using the data in the csv specified in path
    public String createArff(String project, String path, String splitBy, List<String> attributes, int release, String name) {
    	
		String fileName = project + name + ".arff";
		String line = null;
		int i;

		if((name.compareTo(TESTING) != 0) && (name.compareTo(TRAINING) != 0)) {
			logger.severe("Invalid argument name");
			return line;
		}
		
		
		try (BufferedReader br = new BufferedReader(new FileReader(path));
				PrintStream printer = new PrintStream(new File(fileName))) {

			
	    	printer.println("@relation "+project);
	    	
	    	for(i=0; i<attributes.size(); i++) {
	        	printer.println("@attribute "+attributes.get(i));
	    	}

	    	printer.println("@data");

			
	    	//READING FROM CSV FILE
	    	
			line = br.readLine();	//Ignore first line
			
            while ((line = br.readLine()) != null) {
            	
                String[] values = line.split(splitBy);
                int currentRelease = Integer.parseInt(values[0]);
                
                if(currentRelease > release || ((name.compareTo(TRAINING) == 0) && currentRelease == release)) {
                	break;
                }
                                    
                StringBuilder sb = new StringBuilder();
                sb.append(values[2]);
                
                for(i=3;i<values.length;i++) {
                	sb.append(","+values[i]);
                }
                
                printer.println(sb.toString());
            }

        } catch (IOException e) {
        	logger.severe(e.toString());
        }
				
		return fileName;
    	
    	
    }
    
	
    //Writes the dataset in a csv file
    public void createCsv(String project, List<Record> records) throws FileNotFoundException {
    
    	String output = project + ".csv";
    	Record r = null;
    	
    	PrintStream printer = new PrintStream(new File(output));
    	
    	printer.println("Dataset;Training release;%Training;%Defective in training;%Defective in testing;Classifier;Balancing;"
    			+ "Feature selection;TP;FP;TN;FN;Precision;Recall;ROC Area;Kappa");
	   	    	
    	
    	for(int i=0; i<records.size(); i++) {
    		
    		r = records.get(i);
    		
    		printer.println(r.getDataset()+";"+r.getTrainRel()+";"+r.getTrain()+";"+r.getTrainDef()+";"+r.getTestDef()+";"
    				+r.getClassifier()+";"+r.getBalancing()+";"+r.getFeatureSel()+";"+r.getTp()+";"+r.getFp()+";"+r.getTn()+";"
    				+r.getFn()+";"+r.getPrecision()+";"+r.getRecall()+";"+r.getRoc()+";"+r.getKappa());
    		
    	}
    	
    	printer.close();
    	
    } 
    
    
    //Uses RandomForest / NaiveBayes / Ibk as classifiers
    public void classifier(Instances training, Instances testing, FilteredClassifier fc, Record r, int i) {
		Evaluation eval;
    	int trainSize = 0;
    	int testSize = 0;
    	
		trainSize = training.size();
		testSize = testing.size();
    	
		try {
			eval = new Evaluation(testing);
		
	    	//The parameter i specifies which classifier to use
	    	switch(i) {
	    	
	    	case 0:
	    		//RandomForest
	    		r.setClassifier("RandomForest");
	    		
	    	   	RandomForest randomForest = new RandomForest();        	
	        	fc.setClassifier(randomForest);
	    		fc.buildClassifier(training);		
	    		
	    		break;
	    		
	    	case 1:
	    		//NaiveBayes
	    		r.setClassifier("NaiveBayes");
	    		
	    	 	NaiveBayes naiveBayes = new NaiveBayes();        	
	        	fc.setClassifier(naiveBayes);    		
	    		fc.buildClassifier(training);
	    		
	    		break;
	    		
	    	case 2:
	    		//Ibk
	    		r.setClassifier("Ibk");
	    		
	    		IBk ibk = new IBk();
	        	fc.setClassifier(ibk);    		
	    		fc.buildClassifier(training);
	    		
	    		break;
	    	
	    	default:
	    		
	    		logger.severe("Illegal value for argument i");
	    		
	    		break;
	    		
	    	}
	    	
			eval.evaluateModel(fc, testing);
	    	
			r.setTrain((float)trainSize/(trainSize+testSize));
			r.setTp(eval.numTruePositives(1));
			r.setFp(eval.numFalsePositives(1));
			r.setTn(eval.numTrueNegatives(1));
			r.setFn(eval.numFalseNegatives(1));
			r.setPrecision(eval.precision(1));
			r.setRecall(eval.recall(1));
			r.setRoc(eval.areaUnderROC(1));
			r.setKappa(eval.kappa());
	    	
		} catch (Exception e) {
			logger.severe(e.toString());
		}
		
	}
   
    
    //Applies no sampling / oversampling / undersampling / SMOTE for balancing
    public List<Record> balancing(String project, int releases, String featureSel, Instances training, Instances testing,
    		double percent){
		    
    	Record r = null;
    	FilteredClassifier fc = null;
    	int i = 0;
    	String[] opts;
    	
    	List<Record> records = new ArrayList<>();
    	
		//no sampling
    	
    	for(i=0; i<3; i++) {
    		r = new Record(project,releases,featureSel,"No sampling");
    		classifier(training, testing, new FilteredClassifier(), r, i);
    		records.add(r);
    	}
    	
    	//oversampling
    	fc = new FilteredClassifier();

    	Resample resample = new Resample();
    	opts = new String[]  {"-B", "1.0", "-Z", String.valueOf(2*percent*100)};

    	try {
			resample.setOptions(opts);
	    	resample.setInputFormat(training);
		} catch (Exception e) {
			logger.severe(e.toString());
		}

    	fc.setFilter(resample);    	
    	
    	for(i=0; i<3; i++) {
    		r = new Record(project,releases,featureSel,"Oversampling");
    		classifier(training, testing, fc, r, i);
    		records.add(r);
    	}
    	
		//undersampling
    	fc = new FilteredClassifier();

		SpreadSubsample  spreadSubsample = new SpreadSubsample();
		opts = new String[]{ "-M", "1.0"};

		try {
			spreadSubsample.setOptions(opts);
		} catch (Exception e) {
			logger.severe(e.toString());
		}
		
		fc.setFilter(spreadSubsample);
    	
    	for(i=0; i<3; i++) {
    		r = new Record(project,releases,featureSel,"Undersampling");
    		classifier(training, testing, fc, r, i);
    		records.add(r);
    	}

    	//SMOTE
    	fc = new FilteredClassifier();
    	
	    SMOTE smote = new SMOTE();

	    try {
			smote.setInputFormat(training);
		} catch (Exception e) {
			logger.severe(e.toString());
		}
		
	    fc.setFilter(smote);
    	
    	for(i=0; i<3; i++) {
    		r = new Record(project,releases,featureSel,"SMOTE");
    		classifier(training, testing, fc, r, i);
    		records.add(r);
    	}
    	
	    return records;	
    }
    
    
    //Gathers info for each release
    public void getInfo(String path, String separator, int[] releases, int[] buggy) {
    	String line = null;
    	
    	//Taking info from csv file in path
		try (BufferedReader br = new BufferedReader(new FileReader(path))) {

			line = br.readLine();	//Ignore first line
			
            while ((line = br.readLine()) != null) {
            	
                String[] values = line.split(separator);
                int num = Integer.parseInt(values[0])-1;
                
                releases[num]++;
                
                if(values[values.length-1].compareTo("Yes") == 0) {
                    buggy[num]++;
                }
            }    

        } catch (IOException e) {
        	logger.severe(e.toString());
        }	
    	
    }
    
    
    //Uses walk forward as evaluation technique
    public void walkForward(String project, String path, String separator, List<String> attributes, int maxRelease) throws FileNotFoundException {
    	
    	String training = null;		//name of the file with the training dataset
    	String testing = null;		//name of the file with the testing dataset
    	Record r = null;
    	int trainData = 0;
    	int trainBuggy = 0;
    	double percent = 0;
    	
    	List<Record> records = new ArrayList<>();	//list with the records to write in the output csv file
    	int[] releases = new int[maxRelease];	//number of buggy classes for each release
    	int[] buggy = new int[maxRelease];		//tot number of classes in the release
    	
    	getInfo(path, separator, releases, buggy);
    	    	
    	logger.info("Analyzing data ....");
    	
    	for(int i=1; i<releases.length; i++) {
    		
    		training = createArff(project,path,separator,attributes,i+1,TRAINING);
    		testing = createArff(project,path,separator,attributes,i+1,TESTING);
    		
    		trainData += releases[i-1];
    		trainBuggy+= buggy[i-1];
    		
    		percent = (double)(trainBuggy + buggy[i])/(trainData + releases[i]);
    		
    		DataSource trainSource;
			try {
				trainSource = new DataSource(training);
	    		Instances trainingNoFilter = trainSource.getDataSet();
	    		
	    		DataSource testSource = new DataSource(testing);
	    		Instances testingNoFilter = testSource.getDataSet();
	    		
	    		
	    		//No selection
	    		int numAttrNoFilter = trainingNoFilter.numAttributes();
	    		trainingNoFilter.setClassIndex(numAttrNoFilter - 1);
	    		testingNoFilter.setClassIndex(numAttrNoFilter - 1);
	    		
	    		List<Record> noSelection = balancing(project, i, "No selection", trainingNoFilter, testingNoFilter, percent);
	    		
	    		for(int j=0; j<noSelection.size(); j++) {
	    			r = noSelection.get(j);
	    			r.setTrain((double)trainData/(trainData+releases[i]));
	    			r.setTrainDef((double)trainBuggy/trainData);
	    			r.setTestDef((double)buggy[i]/releases[i]);
	    			records.add(r);
	    		}
	    		
	    		
	    		//Best first feature selection
	    		AttributeSelection filter = new AttributeSelection();
	
	    		CfsSubsetEval eval = new CfsSubsetEval();
	    		GreedyStepwise search = new GreedyStepwise();
	    		search.setSearchBackwards(true);
	
	    		filter.setEvaluator(eval);
	    		filter.setSearch(search);
	    		filter.setInputFormat(trainingNoFilter);
	
	    		Instances trainingFiltered = Filter.useFilter(trainingNoFilter, filter);
	    		Instances testingFiltered = Filter.useFilter(testingNoFilter, filter);		
	    		
	    		int numAttrFiltered = trainingFiltered.numAttributes();
	    		
	    		trainingFiltered.setClassIndex(numAttrFiltered - 1);
	    		testingFiltered.setClassIndex(numAttrFiltered - 1);
	
	    		List<Record> bestFirst = balancing(project, i, "Best first", trainingFiltered, testingFiltered, percent);
	    		
	    		for(int j=0; j<bestFirst.size(); j++) {
	    			r = bestFirst.get(j);
	    			r.setTrain((double)trainData/(trainData+releases[i]));
	    			r.setTrainDef((double)trainBuggy/trainData);
	    			r.setTestDef((double)buggy[i]/releases[i]);
	    			records.add(r);
	    		}
			} catch (Exception e) {
				logger.severe(e.toString());
			}

    		
    	}

    	createCsv(project,records);

    	logger.info("DONE");    	
    	
    }
    
	
//-----------------------------------------------------------------------------------------------------------------------------------------------------
	
	public static void main(String[] args) throws IOException {
		List<String> attributes = new ArrayList<>();
		
		JSONReader jr = new JSONReader();	   
		   
		//Taking the configuration from config.json file
		BufferedReader reader = new BufferedReader(new FileReader ("config.json"));
		String config = jr.readAll(reader);
		JSONObject jsonConfig = new JSONObject(config);

		String project = jsonConfig.getString("project");
		int releases = jsonConfig.getInt("releases");
		String path = jsonConfig.getString("path");
		String separator = jsonConfig.getString("separator");
		JSONArray array = jsonConfig.getJSONArray("attributes");
	   
		for(int i=0; i<array.length(); i++) {
			attributes.add(array.get(i).toString());
		}
		
		reader.close();
		
		AnalyzeData ad = new AnalyzeData();
		
		try {
			ad.walkForward(project, path, separator, attributes, releases);
		} catch (Exception e) {
        	logger.severe(e.toString());
		}		

	}
	
}
