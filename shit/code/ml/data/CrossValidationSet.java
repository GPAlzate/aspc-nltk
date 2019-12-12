package ml.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

/**
 * Data set for storing and retrieving an n-fold cross validation data set.
 * 
 * @author dkauchak
 *
 */
public class CrossValidationSet {
	private DataSet dataset;
	private int numSplits;
	private boolean randomize = false;
	
	/**
	 * Generate a cross-validation with numSplits on dataset
	 * 
	 * @param dataset
	 * @param numSplits the number of splits for the data set
	 */
	public CrossValidationSet(DataSet dataset, int numSplits){
		this.dataset = dataset;
		this.numSplits = numSplits;
	}
	
	/**
	 * Generate a *random* cross-validation with numSplits on dataset
	 * 
	 * @param dataset
	 * @param numSplits the number of splits for the data set
	 * @param ranomize whether or not to randomize the data before creating splits
	 */
	public CrossValidationSet(DataSet dataset, int numSplits, boolean randomize){
		this.dataset = dataset;
		this.numSplits = numSplits;
		this.randomize = randomize;
	}
	
	/**
	 * Get the number of splits in this cross validation
	 * 
	 * @return the number of splits
	 */
	public int getNumSplits(){
		return numSplits;
	}
	
	/**
	 * Which split number to retrieve.  Splits start
	 * at 0.
	 * 
	 * @param splitNum the split number requested
	 * @return
	 */
	public DataSetSplit getValidationSet(int splitNum){
		if( splitNum >= numSplits || splitNum < 0){
			return null;
		}else{
			ArrayList<Example> data;
			
			if( randomize ){
				data = (ArrayList<Example>)dataset.getData().clone();
				Collections.shuffle(data, new Random(System.nanoTime()));
			}else{
				data = dataset.getData();
			}
			
			// this tends to make the last split more off-sized, but it will suffice
			int partSize = (int)(data.size()/numSplits);
			int begin = partSize*splitNum;
			int end = splitNum == numSplits-1 ? data.size() : partSize*(splitNum+1);
			
			DataSet train = new DataSet(dataset.getFeatureMap());
			DataSet test = new DataSet(dataset.getFeatureMap());
			
			for( int i = 0; i < data.size(); i++ ){
				// check if we're in the test range
				if( i >= begin && i < end ){
					test.addData(data.get(i));
				}else{
					train.addData(data.get(i));
				}
			}
			
			return new DataSetSplit(train, test);
		}
	}
}
