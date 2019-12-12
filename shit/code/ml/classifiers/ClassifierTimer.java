package ml.classifiers;

import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;

public class ClassifierTimer {
	/**
	 * Calculates the time to train and test the classifier averaged over numRuns on
	 * 80/20 splits of the data
	 * 
	 * @param classifier
	 * @param dataset 
	 */
	public static void timeClassifier(Classifier classifier, DataSet dataset, int numRuns){
		long trainSum = 0;
		long classifySum = 0;
		
		for( int i = 0; i < numRuns; i++ ){
			DataSetSplit split = dataset.split(0.8);			

			System.gc();
			long start = System.currentTimeMillis();
			classifier.train(split.getTrain());
			trainSum += System.currentTimeMillis() - start;

			System.gc();
			start = System.currentTimeMillis();
			classifyExamples(classifier, split.getTest());
			classifySum += System.currentTimeMillis() - start;
		}

		System.out.println("Average train time: " + ((double)trainSum)/numRuns/1000 + "s");
		System.out.println("Average test time: " + ((double)classifySum)/numRuns/1000 + "s");
	}

	/**
	 * Classify all of the examples with the classifier. We don't care about the results
	 * just that the classify function gets called for all of the examples.
	 * 
	 * @param classifier
	 * @param dataset
	 */
	private static void classifyExamples(Classifier classifier, DataSet dataset){
		for( Example e: dataset.getData() ){
			classifier.classify(e);
		}
	}	
}
