package ml.classifiers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import ml.data.DataSet;
import ml.data.Example;
import ml.utils.HashMapCounter;

/**
 * A Naive Bayes classifier, which assumes conditional independence between
 * feature values given the label.
 * 
 * @author xkoo
 * @author galzate
 */
public class NBClassifierOld implements Classifier {

	/** regularization/smoothing parameter */
	private double lambda = 0;
	private boolean onlyPosFeatures = false;

	/** counts of number of examples for each label */
	private HashMapCounter<Double> labelCounts;

	/** a map that maps each label to another map that keeps a feature count */
	private Map<Double, HashMapCounter<Integer>> labelFeatureCount;

	/** counts the total number of examples in dataset */
	private int totalCount;

	/** indices of ALL possible features in the set */
	private Set<Integer> allFeatures;

	/** keep track of last classified example to avoid repeat calculations */
	private Example lastExample = null;

	/** keep track of most likely label for last classified example */
	private double lastPrediction = -1;

	/** keep track of prob of most likely label for last classified example */
	private double lastLogProb = -1;

	/**
	 * Trains the model by keeping a count of number of instances of positive
	 * feature values for all features across all examples given each label.
	 * 
	 * @param data
	 *            DataSet object
	 */
	public void train(DataSet data) {

		labelCounts = new HashMapCounter<Double>();
		labelFeatureCount = new HashMap<Double, HashMapCounter<Integer>>();
		ArrayList<Example> examples = data.getData();
		totalCount = examples.size();
		allFeatures = data.getAllFeatureIndices();

		// Go thru entire list of examples
		for (Example e : examples) {
			// For each example get the label and feature values
			double label = e.getLabel();

			// Increment number of examples for this label
			labelCounts.increment(label);

			// if the label hasn't been found
			if (!labelFeatureCount.containsKey(label)) {
				// create a new negative and positive feature count for that label
				// and add it to the feature counts for the current label
				labelFeatureCount.put(label, new HashMapCounter<Integer>());
			}

			// get all features for this example and iterate through them
			Set<Integer> features = e.getFeatureSet();
			for (int f : features) {
				// Keep track of # of pos features (available and nonzero) given a label
				if (f > 0) {
					// increment the positive count for this label
					labelFeatureCount.get(label).increment(f);
				}
			}
		}
	}

	/**
	 * @return the most likely label (i.e. with the highest log probability)
	 */
	public double classify(Example example) {
		// If we haven't classified/gotten the confidence of this example, recalculate
		// the predicted label and the associated probability
		if (example != lastExample) {
			recalculatePrediction(example);
		}
		return lastPrediction;
	}

	/**
	 * @return the log probability of the most likely label
	 */
	public double confidence(Example example) {

		// If we haven't classified/gotten the confidence of this example, recalculate
		// the predicted label and the associated probability
		if (example != lastExample) {
			recalculatePrediction(example);
		}
		return lastLogProb;
	}

	/**
	 * If we haven't already classified/gotten the confidence of an example, we
	 * store the predicted label and the associated probability into an instance
	 * variable
	 * 
	 * @param example
	 *            new example to be classified
	 */
	private void recalculatePrediction(Example example) {

		// System.out.println("here");

		// New example is the last to be classified
		lastExample = example;

		// Grab set of features (to reduce number of method calls)
		Set<Double> labels = labelCounts.keySet();

		// Get maximum probability
		double maxLogProb = -Double.MAX_VALUE;

		// ... of the example given a label, for all labels
		for (double l : labels) {
			double logProb = getLogProb(example, l);

			// Reassign max log probability and the associated label
			if (logProb > maxLogProb) {
				maxLogProb = logProb;
				lastPrediction = l;
			}
		}
		lastLogProb = maxLogProb;
	}

	/**
	 * Set the lambda (regularization/smoothing parameter) value
	 * 
	 * @param lambda
	 */
	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

	/**
	 * Toggle between calculating probability only across positive features or
	 * across ALL features
	 * 
	 * @param onlyPosFeatures
	 *            boolean
	 */
	public void setUseOnlyPositiveFeatures(boolean onlyPosFeatures) {
		this.onlyPosFeatures = onlyPosFeatures;
	}

	/**
	 * Gets the log probability of the feature values of a certain example given a
	 * certain label
	 * 
	 * @param ex
	 *            the example in question
	 * @param label
	 *            the label to be given
	 * @return log(p(y) * Pi(p(x_i | y)))
	 */
	public double getLogProb(Example ex, double label) {
		// If only counting pos features (i.e. ones appearing in each specific ex)
		Set<Integer> features = ex.getFeatureSet();
		// System.out.println(features.size());
		// System.out.println(allFeatures.size());
		double prob = Math.log10((double) labelCounts.get(label) / (double) totalCount);
		if (onlyPosFeatures) {
			// We want to only iterate thru features that appear in the example
			for (int f : features) {
				prob += Math.log10(getFeatureProb(f, label)); // Get the cumulative product
			}
		}
		// If counting ALL features across the entire dataset
		else {
			// We want to iterate thru all features in the dataset
			for (int f : allFeatures) {
				// Cumulative product will have a 0 and result in 0 if the feature doesn't
				// appear for label. This can only happen if lambda = 0 --> return -inf
				if (prob == 0) {
					return Double.NEGATIVE_INFINITY;
				}

				// Get the cumulative product; if it appears
				// add log(p(x = pos | y)),
				// else add log(p(x = neg | y))
				if (features.contains(f)) {
					// System.out.println("Hit " + f);
					prob += Math.log10(getFeatureProb(f, label));
				} else {
					// System.out.println("Miss " + f);
					prob += Math.log10(1 - getFeatureProb(f, label));
				}
			}
		}
		// Return log of cumulative product
		return prob;
	}

	/**
	 * Gets the (smoothed) probability of a certain feature x_i occurring in an
	 * example given label y for that example.
	 * 
	 * p(x_i|y) = (count(x_i, y) + lambda) / (count(y) + poss_vals_of_x_i * lambda)
	 * 
	 * Note: There are 2 possible values for x_i in this dataset, since a word can
	 * either appear or not
	 * 
	 * @param featureIndex
	 *            x_i, index of the feature in the example
	 * @param y,
	 *            label of the example
	 * @return p(x_i | y)
	 */
	public double getFeatureProb(int featureIndex, double label) {
		// Count of positive instances of a feature given an example / counts of that
		// example in the data.
		return ((double) labelFeatureCount.get(label).get(featureIndex) + lambda)
				/ ((double) labelCounts.get(label) + 2 * lambda);
	}

}
