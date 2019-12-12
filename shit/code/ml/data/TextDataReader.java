package ml.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

import ml.classifiers.DecisionTreeClassifier;
import ml.utils.HashMapCounter;

/**
 * A class for reading text examples.
 * 
 * Each example should be a single line in the file.  The line should start with a numerical
 * label and then the words of the example should follow with each word separated by whitespace.
 * 
 * @author dkauchak
 *
 */
public class TextDataReader implements Iterator<Example>{
	private String nextLine; // next line in the file
	private BufferedReader in; // source to be reading data from
	
	// keep track of the mapping of the words to their feature index
	private HashMap<String, Integer> wordToFeature = new HashMap<String,Integer>();
	private int currentFeature = 0;
		
	/**
	 * @param textFile the text file containing the examples
	 */
	public TextDataReader(String textFile){		
		try {
			in = new BufferedReader(new FileReader(textFile));
			nextLine = in.readLine();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	@Override
	public boolean hasNext() {
		return nextLine != null;
	}

	@Override
	public Example next() {
		Example data = null;
		
		if( hasNext() ){
			data = new Example();
			
			// parse the line
			String[] parts = nextLine.split("\\s+");
			
			data.setLabel(Double.parseDouble(parts[0]));
	
			// do a little bit of preprocessing and count how
			// many times each word occurs
			HashMapCounter<String> counter = new HashMapCounter<String>();
			
			for( int i = 1; i < parts.length; i++ ){
				String w = parts[i].toLowerCase();
				
				// check if it has at least one alphabet character
				if( !w.matches("[^a-z]+")){
					counter.increment(w);
				}
			}
			
			for( String word: counter.keySet() ){
				if( !wordToFeature.containsKey(word) ){
					wordToFeature.put(word, currentFeature);
					currentFeature++;
				}
				
				data.addFeature(wordToFeature.get(word), counter.get(word));
			}
			
			try {
				nextLine = in.readLine();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		return data;
	}
	
	@Override
	public void remove() {
		// OPTIONAL, so we won't implement
	}

	/**
	 * Get the feature mapping (i.e. association from feature index to word) for
	 * all of the examples read so far.  Generally should only be called after
	 * all of the examples have been read.
	 * 
	 * @return feature map
	 */
	public HashMap<Integer, String> getFeatureMap(){
		HashMap<Integer, String> featureMap = new HashMap<Integer, String>();
		
		for( String word: wordToFeature.keySet() ){
			featureMap.put(wordToFeature.get(word), word);
		}
		
		return featureMap;
	}	
}