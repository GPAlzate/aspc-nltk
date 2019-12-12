package ml.data;

import java.util.Iterator;
import java.io.*;

/**
 * A class for reading data examples from CSV files
 * 
 * @author dkauchak
 *
 */
public class CSVDataReader implements Iterator<Example>{
	private String nextLine; // next line in the file
	private BufferedReader in; // source to be reading data from
	private int labelIndex; // the index that the label is at (0-based)
		
	/**
	 * Create a new CSVReader to read the data from in.  The stream must only
	 * contain lines with examples on them (i.e. no header information, etc.).
	 * 
	 * @param in
	 * @param labelIndex the index where the label of the data is at
	 */
	public CSVDataReader(BufferedReader in, int labelIndex){
		this.labelIndex = labelIndex;
		this.in = in;
		
		try {
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
			String[] parts = nextLine.split(",");
			
			data.setLabel(Double.parseDouble(parts[labelIndex]));
	
			int featureIndex = 0;
			
			for( int i = 0; i < parts.length; i++ ){
				if( i != labelIndex ){
					data.addFeature(featureIndex, Double.parseDouble(parts[i]));
					featureIndex++;
				}
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
}