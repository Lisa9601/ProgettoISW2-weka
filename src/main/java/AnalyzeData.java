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
    
    
    static {

        System.setProperty("java.util.logging.config.file", "logging.properties");
        logger = Logger.getLogger(AnalyzeData.class.getName());
    }
    
	
	//Creates a new arff file for training using the data in the csv file in path
	public String createTrainingArff(String project, String path, String splitBy, List<String> attributes, int release) throws FileNotFoundException {
		
		String fileName = project + "training.arff";
		String line = null;
		int i;
		
    	PrintStream printer = new PrintStream(new File(fileName));
    	
    	printer.println("@relation "+project);
    	
    	for(i=0; i<attributes.size(); i++) {
        	printer.println("@attribute "+attributes.get(i));
    	}
		
    	
    	printer.println("@data");
    	
		try (BufferedReader br = new BufferedReader(new FileReader(path))) {

			line = br.readLine();	//Ignore first line
			
            while ((line = br.readLine()) != null) {

                String newLine = "";
            	
                String[] values = line.split(splitBy);
                
                if(Integer.parseInt(values[0]) >= release) {
                	break;
                }
                
                newLine = values[2];
                
                for(i=3;i<values.length;i++) {
                	newLine = newLine + "," + values[i];
                }
                
                printer.println(newLine);

            }

        } catch (FileNotFoundException e) {
        	logger.severe(e.toString());
        } catch (IOException e) {
        	logger.severe(e.toString());
        }
		
		printer.close();
		
		return fileName;
		
	}
	
	
	//Creates a new arff file for testing using the data in the csv file in path
	public String createTestingArff(String project, String path, String splitBy, List<String> attributes, int release) throws FileNotFoundException {
		
		String fileName = project + "testing.arff";
		String line = null;
		int i;
		
    	PrintStream printer = new PrintStream(new File(fileName));
    	
    	printer.println("@relation "+project);
    	
    	for(i=0; i<attributes.size(); i++) {
        	printer.println("@attribute "+attributes.get(i));
    	}
		
    	
    	printer.println("@data");
    	
		try (BufferedReader br = new BufferedReader(new FileReader(path))) {

			line = br.readLine();	//Ignore first line
			
            while ((line = br.readLine()) != null) {
            	
                String newLine = "";
            	
                String[] values = line.split(splitBy);
                
                int currentRelease = Integer.parseInt(values[0]);
                
                if(currentRelease > release) {
                	break;
                }
                else if(currentRelease == release) {
                    newLine = values[2];
                    
                    for(i=3;i<values.length;i++) {
                    	newLine = newLine + "," + values[i];
                    }
                    
                    printer.println(newLine);
                    
                }

            }

        } catch (FileNotFoundException e) {
        	logger.severe(e.toString());
        } catch (IOException e) {
        	logger.severe(e.toString());
        }
		
		printer.close();
		
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
    public void classifier(Instances training, Instances testing, FilteredClassifier fc, Record r, int i) throws Exception {
    	Evaluation eval = new Evaluation(testing);
    	int trainSize = 0;
    	int testSize = 0;
    	
		trainSize = training.size();
		testSize = testing.size();
    	
    	//The parameter i specifies which classifier to use
    	switch(i) {
    	
    	case 0:
    		//RandomForest
    		r.setClassifier("RandomForest");
    		
    	   	RandomForest RandomForest = new RandomForest();        	
        	fc.setClassifier(RandomForest);
    		fc.buildClassifier(training);		
    		
    		break;
    		
    	case 1:
    		//NaiveBayes
    		r.setClassifier("NaiveBayes");
    		
    	 	NaiveBayes NaiveBayes = new NaiveBayes();        	
        	fc.setClassifier(NaiveBayes);    		
    		fc.buildClassifier(training);
    		
    		break;
    		
    	case 2:
    		//Ibk
    		r.setClassifier("Ibk");
    		
    		IBk ibk = new IBk();
        	fc.setClassifier(ibk);    		
    		fc.buildClassifier(training);
    		
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
    	
    }
   
    
    //Applies no sampling / oversampling / undersampling / SMOTE for balancing
    public void balancing(String project, int releases, String featureSel, Instances training, Instances testing, List<Record> records) throws Exception {
		    
    	Record r = null;
    	FilteredClassifier fc = null;
    	int i = 0;
    	
		//no sampling
    	fc = new FilteredClassifier();
    	
    	for(i=0; i<3; i++) {
    		r = new Record(project,releases,featureSel,"No sampling");
    		classifier(training, testing, new FilteredClassifier(), r, i);
    		records.add(r);
    	}
    	
    	//oversampling
    	fc = new FilteredClassifier();

    	Resample resample = new Resample();
		resample.setInputFormat(training);
    	
    	for(i=0; i<3; i++) {
    		r = new Record(project,releases,featureSel,"Oversampling");
    		classifier(training, testing, new FilteredClassifier(), r, i);
    		records.add(r);
    	}
    	
		//undersampling
    	fc = new FilteredClassifier();

		SpreadSubsample  spreadSubsample = new SpreadSubsample();
		String[] opts = new String[]{ "-M", "1.0"};
		spreadSubsample.setOptions(opts);
		fc.setFilter(spreadSubsample);
    	
    	for(i=0; i<3; i++) {
    		r = new Record(project,releases,featureSel,"Undersampling");
    		classifier(training, testing, new FilteredClassifier(), r, i);
    		records.add(r);
    	}

    	//SMOTE
    	fc = new FilteredClassifier();
    	
	    SMOTE smote = new SMOTE();
		smote.setInputFormat(training);
		fc.setFilter(smote);
    	
    	for(i=0; i<3; i++) {
    		r = new Record(project,releases,featureSel,"SMOTE");
    		classifier(training, testing, new FilteredClassifier(), r, i);
    		records.add(r);
    	}
	    	
    }
    
    
    //Uses walk forward as evaluation technique
    public void walkForward(String project, String path, String separator, List<String> attributes, int maxRelease) throws Exception {
    	
    	String training = null;		//name of the file with the training dataset
    	String testing = null;		//name of the file with the testing dataset
    	
    	List<Record> records = new ArrayList<>();	//list with the records to write in the output csv file
    	
    	logger.info("Analyzing data ....");
    	
    	for(int i=2; i<=maxRelease; i++) {
    		
    		training = createTrainingArff(project,path,separator,attributes,i);
    		testing = createTestingArff(project,path,separator,attributes,i);
    		
    		DataSource trainSource = new DataSource(training);
    		Instances trainingNoFilter = trainSource.getDataSet();
    		
    		DataSource testSource = new DataSource(testing);
    		Instances testingNoFilter = testSource.getDataSet();
    		
    		//No selection
    		int numAttrNoFilter = trainingNoFilter.numAttributes();
    		trainingNoFilter.setClassIndex(numAttrNoFilter - 1);
    		testingNoFilter.setClassIndex(numAttrNoFilter - 1);
    		
    		balancing(project, i-1, "No selection", trainingNoFilter, testingNoFilter, records);
    		
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

    		balancing(project, i-1, "Best first", trainingFiltered, testingFiltered, records);
    		
    	}

    	createCsv(project,records);

    	logger.info("DONE");    	
    	
    }
    
	
//-----------------------------------------------------------------------------------------------------------------------------------------------------
	
	public static void main(String args[]) throws IOException {
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
