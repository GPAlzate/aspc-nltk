package ml.classifiers;

import ml.data.DataSet;
import ml.data.Example;

/**
 * Interface for a classifier.
 * 
 * @author dkauchak
 *
 */
public interface Classifier {
	/**
	 * Train this classifier based on the data set
	 * 
	 * @param data
	 */
	public void train(DataSet data);
	
	/**
	 * Classify the example.  Should only be called *after* train has been called.
	 * 
	 * @param example
	 * @return the class label predicted by the classifier for this example
	 */
	public double classify(Example example);
	
	public double confidence(Example example);
}
