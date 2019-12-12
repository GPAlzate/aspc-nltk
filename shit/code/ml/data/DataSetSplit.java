package ml.data;

/**
 * A train/test split of the data
 * 
 * @author dkauchak
 */
public class DataSetSplit {
	private DataSet train;
	private DataSet test;
	
	/**
	 * Create a new data set split
	 * 
	 * @param train
	 * @param test
	 */
	public DataSetSplit(DataSet train, DataSet test){
		this.train = train;
		this.test = test;
	}

	/**
	 * Get the training portion of this split
	 * 
	 * @return
	 */
	public DataSet getTrain() {
		return train;
	}

	/**
	 * Get the testing portion of this split
	 * 
	 * @return
	 */
	public DataSet getTest() {
		return test;
	}
	
	public String toString(){
		return "Train: " + train.getData().size() + "\tTest: " + test.getData().size();
	}
}
