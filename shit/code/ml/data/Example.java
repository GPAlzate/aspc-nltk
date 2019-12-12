package ml.data;

import java.util.Collections;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

/**
 * A class to represent an example
 * 
 * @author dkauchak
 */
public class Example {
	// We'll use a sparse representation of the features.
	// Rather than keeping the name of the feature (i.e. the header information), we'll
	// index the features starting at 0.  The DataSet class will keep a mapping from
	// feature index to the actual name.
	private HashMap<Integer, Double> sparseData;
	private double label;  // the label (assuming it has one)
	
	public Example(){
		sparseData = new HashMap<Integer, Double>();
	}
	
	public Example(Example e){
		// copy everything
		label = e.label;
		sparseData = new HashMap<Integer,Double>(e.sparseData);
	}
	
	/**
	 * Add a feature with value to this example.
	 * 
	 * @param featureNum  the index of the feature to be added
	 * @param value  the value to be added
	 */
	public void addFeature(int featureNum, double value){
		sparseData.put(featureNum, value);
	}
		
	/**
	 * Get the value associated with this feature.
	 * 
	 * @param featureNum
	 * @return the value for featureNum for this example
	 */
	public double getFeature(int featureNum){
		return sparseData.containsKey(featureNum) ? sparseData.get(featureNum) : 0.0;
	}
	
	/**
	 * Set the values of the associated features with feature index featureNum.
	 * 
	 * @param featureNum
	 * @param value
	 */
	public void setFeature(int featureNum, double value){
		sparseData.put(featureNum, value);
	}
	
	/**
	 * Get all the features that this example has (indices).
	 * 
	 * @return the set of features
	 */
	public Set<Integer> getFeatureSet(){
		return sparseData.keySet();
	}
	
	/**
	 * Set the label associated with this example.
	 * 
	 * @param label
	 */
	public void setLabel(double label){
		this.label = label;
	}
	
	/**
	 * Get the label associated with this example.
	 * 
	 * @return the example label
	 */
	public double getLabel(){
		return label;
	}
	
	/**
	 * Checks for equality between two examples *ignoring* the label
	 * 
	 * @param other another example (Data object)
	 * @return whether or not these examples have the same feature values
	 */
	public boolean equalFeatures(Example other){
		return sparseData.equals(other.sparseData);
	}
	
	public String toString(){
		StringBuffer buffer = new StringBuffer();
		buffer.append(label);
		
		ArrayList<Integer> indices = new ArrayList<Integer>(sparseData.keySet());
		Collections.sort(indices);
		
		for( int featureIndex: indices){
			buffer.append(" " + featureIndex + ":" + valueToString(sparseData.get(featureIndex)));
		}
		
		return buffer.toString();
	}
	
	public String toString(HashMap<Integer, String> featureMap){
		StringBuffer buffer = new StringBuffer();
		buffer.append(label);
		
		ArrayList<Integer> indices = new ArrayList<Integer>(sparseData.keySet());
		Collections.sort(indices);
		
		for( int featureIndex: indices){
			buffer.append(" " + featureMap.get(featureIndex) + ":" + valueToString(sparseData.get(featureIndex)));
		}
		
		return buffer.toString();
	}

	/**
	 * CSV representation of this example
	 * 
	 * @return csv representation
	 */
	public String toCSVString(){
		StringBuffer buffer = new StringBuffer();
		
		ArrayList<Integer> indices = new ArrayList<Integer>(sparseData.keySet());
		Collections.sort(indices);
		
		for( int featureIndex: indices){
			buffer.append(valueToString(sparseData.get(featureIndex)) + ",");
		}
		
		buffer.append( valueToString(label) );
		
		return buffer.toString();
	}
	
	/**
	 * Helper function to output integers as integer strings instead of as doubles
	 * 
	 * @param val
	 * @return
	 */
	private String valueToString(double val){
		if( val == (int)val){
			return Integer.toString((int)val);
		}else{
			return Double.toString(val);
		}
	}
}
