package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Set;

import ml.data.DataSet;
import ml.data.Example;

/**
 * Average perceptron classifier.  We'll utilize most of the code from the
 * PerceptronClassifier class and only need to rewrite the train method.
 * 
 * @author dkauchak
 *
 */
public class AveragePerceptronClassifier extends PerceptronClassifier {
	
	public void train(DataSet data) {
		initializeWeights(data.getAllFeatureIndices());
		
		ArrayList<Example> training = (ArrayList<Example>)data.getData().clone();

		int total = 0;
		int lastUpdate = 1;
		
		// initialize the weights
		HashMap<Integer, Double> sumWeights = getZeroWeights(weights.keySet());
		double sumB = 0;
		
		for( int it = 0; it < iterations; it++ ){
			Collections.shuffle(training);
			
			for( Example e: training ){
				if( getPrediction(e) != e.getLabel() ){
					double label = e.getLabel();

					// update the weights
					for( Integer featureIndex: weights.keySet() ){
						double featureValue = e.getFeature(featureIndex);
						double oldSumWeight = sumWeights.get(featureIndex);
						double oldWeight = weights.get(featureIndex);

						// update the aggregate weights
						sumWeights.put(featureIndex, oldSumWeight + lastUpdate*oldWeight);

						
						// update the basic weights
						weights.put(featureIndex, oldWeight + featureValue*label);						
					}

					// update sumB
					sumB += lastUpdate*b;
					
					// update b
					b += label;
					
					lastUpdate = 0;
				}
				
				total++;
				lastUpdate++;
			}
		}
		
		// normalize the weights and save back into the weights vector
		for( Integer featureIndex: sumWeights.keySet() ){
			weights.put(featureIndex, sumWeights.get(featureIndex)/total);
		}
		
		b = sumB/total;
	}
}
