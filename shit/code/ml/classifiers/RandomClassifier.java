package ml.classifiers;

import java.util.Random;

import ml.data.DataSet;
import ml.data.Example;

/**
 * A classifier that randomly labels examples as either 0 or 1.
 * 
 * @author dkauchak
 *
 */
public class RandomClassifier implements Classifier{
	private Random rand = new Random();
	
	@Override
	public void train(DataSet data) {
		// easiest training method ever!
	}

	@Override
	public double classify(Example example) {
		return rand.nextInt(2) == 1? 1.0 : -1.0;
	}

	@Override
	public double confidence(Example example) {
		return 1.0; // super confident!
	}
}
