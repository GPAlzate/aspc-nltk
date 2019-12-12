package ml.data;

import java.util.ArrayList;

/**
 * Interface defining the data preprocessing
 * 
 * @author dkauchak
 *
 */
public interface DataPreprocessor {
	/**
	 * Preprocess the training data
	 * 
	 * @param train
	 */
	public void preprocessTrain(DataSet train);
	
	/**
	 * Preprocess the testing data
	 * 
	 * @param test
	 */
	public void preprocessTest(DataSet test);
}
