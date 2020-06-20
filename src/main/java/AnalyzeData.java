package main.java;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;

import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class AnalyzeData {

	
    //Writes the dataset in a csv file
    public void writeCsv(String project) throws FileNotFoundException {
    
    	String output = project + ".csv";
    	
    	PrintStream printer = new PrintStream(new File(output));
    	
    	printer.println("Dataset;Training release;%Training;%Defective in training;%Defective in testing;EPV before;EPV after;Classifier;"
    			+ "Balancing;Feature selection;TP;FP;TN;FN;Precision;Recall;ROC Area;Kappa");
	   	    	
    	
    	//CICLO QUì
    	
    	printer.close();
    	
    } 
	
	
	
	public static void main(String args[]) throws Exception{
		DataSource source2 = new DataSource("C:/Program Files/Weka-3-8/data/breast-cancerNOTK.arff");
		Instances testingNoFilter = source2.getDataSet();

		DataSource source = new DataSource("C:/Program Files/Weka-3-8/data/breast-cancerKnown.arff");
		Instances noFilterTraining = source.getDataSet();
		//create AttributeSelection object
		AttributeSelection filter = new AttributeSelection();
		//create evaluator and search algorithm objects
		CfsSubsetEval eval = new CfsSubsetEval();
		GreedyStepwise search = new GreedyStepwise();
		//set the algorithm to search backward
		search.setSearchBackwards(true);
		//set the filter to use the evaluator and search algorithm
		filter.setEvaluator(eval);
		filter.setSearch(search);
		//specify the dataset
		filter.setInputFormat(noFilterTraining);
		//apply
		Instances filteredTraining = Filter.useFilter(noFilterTraining, filter);
		
		int numAttrNoFilter = noFilterTraining.numAttributes();
		noFilterTraining.setClassIndex(numAttrNoFilter - 1);
		testingNoFilter.setClassIndex(numAttrNoFilter - 1);
		
		int numAttrFiltered = filteredTraining.numAttributes();

		
		System.out.println("No filter attr: "+ numAttrNoFilter);
		System.out.println("Filtered attr: "+ numAttrFiltered);
		
		RandomForest classifier = new RandomForest();

		
		//evaluation with no filtered
		Evaluation evalClass = new Evaluation(testingNoFilter);
		classifier.buildClassifier(noFilterTraining);
	    evalClass.evaluateModel(classifier, testingNoFilter); 
		
		System.out.println("AUC no filter = "+evalClass.areaUnderROC(1));
		System.out.println("Kappa no filter = "+evalClass.kappa());
	
		//evaluation with filtered
		filteredTraining.setClassIndex(numAttrFiltered - 1);
		Instances testingFiltered = Filter.useFilter(testingNoFilter, filter);
		testingFiltered.setClassIndex(numAttrFiltered - 1);
		classifier.buildClassifier(filteredTraining);
	    evalClass.evaluateModel(classifier, testingFiltered);
		
		System.out.println("AUC filtered = "+evalClass.areaUnderROC(1));
		System.out.println("Kappa filtered = "+evalClass.kappa());

	}
	
}
