package CombineAlgorithmResults;
/**
 * A Java class that run several algorithms and combines their results
 *  */

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import weka.classifiers.bayes.ComplementNaiveBayes;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class CombineAlgorithmResults {

	static ArrayList<String> negativeTweets = new ArrayList<String>();
	static ArrayList<String> positiveTweets = new ArrayList<String>();
	
	static CombineAlgorithmResults car = new CombineAlgorithmResults();

	public FilteredClassifier loadModel(String fileName) {

		FilteredClassifier classifier = new FilteredClassifier();

		try {

			ObjectInputStream in = new ObjectInputStream(new FileInputStream(fileName));
			Object tmp = in.readObject();
			classifier = (FilteredClassifier) tmp;
			in.close();
			//System.out.println("Loaded model: " + fileName);
		} 
		catch (Exception e) {
			// Given the cast, a ClassNotFoundException must be caught along with the IOException
			System.out.println("Problem found when reading: " + fileName + e.getMessage());
		}
		return classifier;
	}

	public Instances makeInstance(String str) {

		Instances instances;
		String tweet = new String(str);
		tweet = tweet.toLowerCase();
		tweet = tweet.replaceAll("[^a-zöçþðüý ]", " ");

		// Create the attributes, class and text
		FastVector fvNominalVal = new FastVector(3);
		fvNominalVal.addElement("Negative");
		fvNominalVal.addElement("Positive");
		Attribute attribute1 = new Attribute("class", fvNominalVal);
		Attribute attribute2 = new Attribute("text",(FastVector) null);
		// Create list of instances with one element
		FastVector fvWekaAttributes = new FastVector(2);
		fvWekaAttributes.addElement(attribute2);
		fvWekaAttributes.addElement(attribute1);
		instances = new Instances("Test relation", fvWekaAttributes, 1);           
		// Set class index
		instances.setClassIndex(1);
		// Create and add the instance
		Instance instance = new Instance(2);
		instance.setValue(attribute2, tweet);
		instances.add(instance);
		return instances;
	}

	public double calculateAccuracy(FilteredClassifier classifier) throws Exception {
		
		double truePositiveCount = 0;
		double falsePositiveCount = 0;
		double trueNegativeCount = 0;
		double falseNegativeCount = 0;
		
		for(int i = 0; i < negativeTweets.size(); i++) {

			if(classifier.classifyInstance(car.makeInstance(negativeTweets.get(i)).firstInstance()) == 0)
				trueNegativeCount++;
			else
				falseNegativeCount++;
		}
		for(int i = 0; i < positiveTweets.size(); i++) {

			if(classifier.classifyInstance(car.makeInstance(positiveTweets.get(i)).firstInstance()) == 1)
				truePositiveCount++;
			else
				falsePositiveCount++;
		}
		double FmeasurePositive = truePositiveCount / (truePositiveCount + falsePositiveCount);
		double FmeasureNegative = trueNegativeCount / (trueNegativeCount + falseNegativeCount);
		System.out.println("FmeasurePositive is: " + FmeasurePositive);
		System.out.println("FmeasureNegative is: " + FmeasureNegative);
		return (truePositiveCount + trueNegativeCount) / (truePositiveCount + trueNegativeCount + falsePositiveCount + falseNegativeCount);
	}
	
public double calculateAccuracyCombine(FilteredClassifier[] classifier) throws Exception {
		
		double truePositiveCount = 0;
		double falsePositiveCount = 0;
		double trueNegativeCount = 0;
		double falseNegativeCount = 0;
		double score = 0;
		
		for(int i = 0; i < negativeTweets.size(); i++) {
			score = 0;
			for(int j = 0; j < classifier.length; j++) {
			
				score += classifier[j].classifyInstance(car.makeInstance(negativeTweets.get(i)).firstInstance());
			}
			if(score < 3.5)
				trueNegativeCount++;
			else
				falseNegativeCount++;
		}
		for(int i = 0; i < positiveTweets.size(); i++) {
			score = 0;
			for(int j = 0; j < classifier.length; j++) {
				
				score += classifier[j].classifyInstance(car.makeInstance(positiveTweets.get(i)).firstInstance());
			}
			if(score > 2.5)
				truePositiveCount++;
			else
				falsePositiveCount++;
		}
		double FmeasurePositive = truePositiveCount / (truePositiveCount + falsePositiveCount);
		double FmeasureNegative = trueNegativeCount / (trueNegativeCount + falseNegativeCount);
		System.out.println("FmeasurePositive is: " + FmeasurePositive);
		System.out.println("FmeasureNegative is: " + FmeasureNegative);
		return (truePositiveCount + trueNegativeCount) / (truePositiveCount + trueNegativeCount + falsePositiveCount + falseNegativeCount);
	}

	@SuppressWarnings("resource")
	public static void main (String[] args) throws Exception {

		
		String str;
		BufferedReader in = new BufferedReader(new FileReader("negative.txt"));
		while ((str = in.readLine()) != null) {
			negativeTweets.add(str);
		}

		in = new BufferedReader(new FileReader("positive.txt"));
		while ((str = in.readLine()) != null) {
			positiveTweets.add(str);
		}
		
		FilteredClassifier[] classifier = new FilteredClassifier[5]; 
				
		classifier[0] = car.loadModel("filterednaivebayes.model");
		System.out.println("Naive bayes");
		System.out.println("Accuracy (Real Test Data) is " + car.calculateAccuracy(classifier[0]) + "\n");

		classifier[1] = car.loadModel("filteredcomplement.model");
		System.out.println("Complement");
		System.out.println("Accuracy (Real Test Data) is " + car.calculateAccuracy(classifier[1]) + "\n");
		
		classifier[2] = car.loadModel("filteredmultinomial.model");
		System.out.println("Multinomial");
		System.out.println("Accuracy (Real Test Data) is " + car.calculateAccuracy(classifier[2]) + "\n");
		
		classifier[3] = car.loadModel("filteredlibsvm.model");
		System.out.println("SVM");
		System.out.println("Accuracy (Real Test Data) is " + car.calculateAccuracy(classifier[3]) + "\n");
		
		classifier[4] = car.loadModel("filteredj48.model");
		System.out.println("J48");
		System.out.println("Accuracy (Real Test Data) is " + car.calculateAccuracy(classifier[4]) + "\n");

		System.out.println("Combined");
		System.out.println("Accuracy is " + car.calculateAccuracyCombine(classifier));

		// load dataset
		DataSource source = new DataSource("data.arff");
		Instances data = source.getDataSet();

		if (data.classIndex() == -1)
			data.setClassIndex(0);

		// create algoritms instances and load their models
		NaiveBayes nb = new NaiveBayes();
		nb = (NaiveBayes) weka.core.SerializationHelper.read("naivebayes.model");

		NaiveBayesMultinomial nbm = new NaiveBayesMultinomial();
		nbm = (NaiveBayesMultinomial) weka.core.SerializationHelper.read("naivebayesmultinomial.model");

		ComplementNaiveBayes cnb = new ComplementNaiveBayes();
		cnb = (ComplementNaiveBayes) weka.core.SerializationHelper.read("complementnaivebayes.model");

		LibSVM svm = new LibSVM();
		svm = (LibSVM) weka.core.SerializationHelper.read("libsvm.model");

		J48 j48 = new J48();
		j48 = (J48) weka.core.SerializationHelper.read("j48.model");

		int truePositiveCount = 0;
		int falsePositiveCount = 0;
		int trueNegativeCount = 0;
		int falseNegativeCount = 0;

		for(int i = 0; i < data.numInstances(); i++) {

			double d = nb.classifyInstance(data.instance(i));
			d += nbm.classifyInstance(data.instance(i));
			d += cnb.classifyInstance(data.instance(i));
			d += svm.classifyInstance(data.instance(i));
			d += j48.classifyInstance(data.instance(i));

			if(d > 2.5) { // more than 3 algorithms predicts positive

				if(data.instance(i).classValue() > 0.5) { // it is really positive

					truePositiveCount++;
				}
				else {

					falseNegativeCount++;
				}
			} else { // more than 3 algorithms predicts negative

				if(data.instance(i).classValue() > 0.5) {

					falsePositiveCount++;
				}
				else { // // it is really negative

					trueNegativeCount++;
				}
			}
		}
		System.out.println("\nTrue Positive: " + truePositiveCount);
		System.out.println("False Positive: " + falsePositiveCount);
		System.out.println("True Negative: " + trueNegativeCount);
		System.out.println("False Negative: " + falseNegativeCount);
		
		double precision = ((double)(truePositiveCount) / (double)(truePositiveCount + falsePositiveCount));
				double recall = ((double)(truePositiveCount) / (double)(truePositiveCount + falseNegativeCount));
				double fmeasure = (2 * precision * recall) / (precision + recall);
				System.out.println("precision: " + precision + "recall: " + recall + "fmeasure" + fmeasure);
				
				precision = ((double)(trueNegativeCount) / (double)(trueNegativeCount + falseNegativeCount));
				recall = ((double)(trueNegativeCount) / (double)(trueNegativeCount + falsePositiveCount));
				fmeasure = (2 * precision * recall) / (precision + recall);
				System.out.println("precision: " + precision + "recall: " + recall + "fmeasure" + fmeasure);
				
		System.out.println("f positive: " + ((double)(truePositiveCount) / 
				(double)(truePositiveCount + falsePositiveCount)));
		System.out.println("f negative: " + ((double)(trueNegativeCount) / 
				(double)(trueNegativeCount + falseNegativeCount)));
		System.out.println("Accuracy: " + ((double)(truePositiveCount+trueNegativeCount) / 
				(double)(truePositiveCount + falsePositiveCount + trueNegativeCount + falseNegativeCount)));
	}
}	
