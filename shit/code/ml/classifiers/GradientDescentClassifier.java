package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Set;

import ml.data.DataSet;
import ml.data.Example;

/**
 * Gradient descent classifier allowing for two different loss functions and
 * three different regularization settings.
 * 
 * @author dkauchak
 * @author galzate
 * @author xkoo
 *
 */
public class GradientDescentClassifier implements Classifier {
	// constants for the different surrogate loss functions
	public static final int EXPONENTIAL_LOSS = 0;
	public static final int HINGE_LOSS = 1;

	// constants for the different regularization parameters
	public static final int NO_REGULARIZATION = 0;
	public static final int L1_REGULARIZATION = 1;
	public static final int L2_REGULARIZATION = 2;

	// should we print statements corresponding with questions?
	private boolean printQuestion1 = false;
	private boolean printQ3Q4 = false;
	private boolean printQuestion5 = false;

	protected int lossType = 0;
	protected int regType = 0;

	/** the feature weights */
	protected HashMap<Integer, Double> weights;

	/** the intersect weight */
	protected double b = 0;

	/** regularization rate */
	protected double lambda = 0.01;

	/** learning rate */
	protected double eta = 0.01;

	protected int iterations = 10;

	/**
	 * Get a weight vector over the set of features with each weight set to 0
	 * 
	 * @param features the set of features to learn over
	 * @return
	 */
	protected HashMap<Integer, Double> getZeroWeights(Set<Integer> features) {
		HashMap<Integer, Double> temp = new HashMap<Integer, Double>();

		for (Integer f : features) {
			temp.put(f, 0.0);
		}

		return temp;
	}

	/**
	 * Initialize the weights and the intersect value
	 * 
	 * @param features
	 */
	protected void initializeWeights(Set<Integer> features) {
		weights = getZeroWeights(features);
		b = 0;
	}

	/**
	 * Set loss function type. Defaults to no regularization.
	 * 
	 * 
	 * @param lossCode the kind of loss function to use. exponential:0, hinge:1
	 */
	public void setLoss(int lossCode) {
		lossType = lossCode;
	}

	/**
	 * Set regularization method. Defaults to no regularization.
	 * 
	 * @param lossCode the kind of loss function to use. exponential:0, hinge:1
	 */
	public void setRegularization(int regCode) {
		regType = regCode;
	}

	/**
	 * Sets lambda, the regularization rate
	 * 
	 * @param newLambda
	 */
	public void setLambda(double newLambda) {
		lambda = newLambda;
	}

	/**
	 * Sets eta, the learning rate
	 * 
	 * @param newEta
	 */
	public void setEta(double newEta) {
		eta = newEta;
	}

	/**
	 * Set the number of iterations the perceptron should run during training
	 * 
	 * @param iterations
	 */
	public void setIterations(int iterations) {
		this.iterations = iterations;
	}

	/**
	 * Sets the boolean that decides whether or not to print the weights
	 * before/after every example, for question 1
	 * 
	 * @param print true or false
	 */
	public void setPrintQuestion1(boolean print) {
		printQuestion1 = print;
	}

	/**
	 * Sets the boolean that decides whether or not to print the total loss for
	 * every iteration, for question 3, 4
	 * 
	 * @param print true or false
	 */
	public void setPrintQ3Q4(boolean print) {
		printQ3Q4 = print;
	}

	/**
	 * Sets the boolean that decides whether or not to print the final iteration's
	 * loss, for question 5 (finding optimal lambda and eta)
	 * 
	 * @param print
	 */
	public void setPrintQuestion5(boolean print) {
		printQuestion5 = print;
	}

	/**
	 * Trains the dataset using gradient descent. Updates the weights and bias for
	 * every example, with loss correction and regularization.
	 * 
	 * @param data dataset object to be trained on
	 */
	public void train(DataSet data) {
		initializeWeights(data.getAllFeatureIndices());
		ArrayList<Example> training = new ArrayList<Example>(data.getData());

		// Saves the weights/bias when we calculate loss so we can use these working
		// weights when we regularize
		HashMap<Integer, Double> weightsInProgress = new HashMap<Integer, Double>();
		double bInProgress = b;

		// Used for question 5. Gets the loss at the final iteration
		double finalLoss = 0;

		for (int it = 0; it < iterations; it++) {
			// Don't shuffle if we're handling question 1
			if (!printQuestion1)
				Collections.shuffle(training);
			else
				System.out.println("~~~~~~~~~~~~~~~~~~~ ITERATION " + (it) + ": ~~~~~~~~~~~~~~~~~~~");

			// Keep track of the sum of the loss values for each iteration (question 3, 4)
			double lossSum = 0;

			// Iterate through every example in the training set
			for (Example e : training) {

				// All of the following are constant within a single example
				// The label of the example
				double y_i = e.getLabel();
				// y' = w_i * x_i + b
				double yPrime = getDistanceFromHyperplane(e, weights, b);
				// calculate the derivatives of the loss functions
				double hingeDeriv = (y_i * yPrime < 1) ? 1 : 0, expDeriv = Math.exp(-y_i * yPrime);

				// Experiment for (3), getting the summed loss functions across each iteration
				lossSum += loss(y_i, yPrime);

				// Prints values of weights/bias (for question 1) before weight/bias update
				if (printQuestion1) {
					System.out.println("BEFORE EXAMPLE:\n\tWeights:\t" + weights.toString() + "\n\tBias:   \t" + b
							+ "\n-------------------");
				}

				/*
				 * Calculate weight updates wrt the loss correction update bias as if it were
				 * associated with some feature == 1. store these values into temp variables.
				 * bInProgress is updated outside of the method, because it is a local variable
				 * of a primitive type and changes made to the variable won't be reflected
				 * outside the method
				 */
				switch (lossType) {
				case EXPONENTIAL_LOSS:
					calculateWeightsLossCorrection(weightsInProgress, e, expDeriv);
					bInProgress = b + eta * y_i * expDeriv;
					break;
				case HINGE_LOSS:
					calculateWeightsLossCorrection(weightsInProgress, e, hingeDeriv);
					bInProgress = b + eta * y_i * hingeDeriv;
					break;
				}

				// update the values of weights, b.
				calculateRegularization(weightsInProgress, bInProgress, e);

				// Prints values of weights/bias (for question 1) after weight/bias update
				if (printQuestion1) {
					System.out.println("AFTER EXAMPLE:\n\tWeights:\t" + weights.toString() + "\n\tBias:   \t" + b
							+ "\n-------------------");
				}
			}

			// Prints the summed loss for each iteration (question 3, 4)
			if (printQ3Q4)
				System.out.println("Summed loss @ it=" + it + "\t" + lossSum);

			finalLoss = lossSum;
		}
		if (printQuestion5)
			System.out.println(finalLoss);
	}

	/**
	 * Predicts the label of an example using a perception classifier, trained using
	 * gradient descent.
	 * 
	 * @pre train() has been called
	 * @param example example to the classified
	 * @return classification for the given example
	 */
	public double classify(Example example) {
		return getPrediction(example);
	}

	/**
	 * @param example the example whose confidence we want
	 * @return confidence of the given example, i.e. its distance from the
	 *         hyperplane
	 */
	public double confidence(Example example) {
		return Math.abs(getDistanceFromHyperplane(example, weights, b));
	}

	/**
	 * Get the prediction from the current set of weights on this example
	 * 
	 * @param e the example to predict
	 * @return
	 */
	protected double getPrediction(Example e) {
		return getPrediction(e, weights, b);
	}

	/**
	 * Get the prediction from the on this example from using weights w and inputB
	 * 
	 * @param e      example to predict
	 * @param w      the set of weights to use
	 * @param inputB the b value to use
	 * @return the prediction
	 */
	protected static double getPrediction(Example e, HashMap<Integer, Double> w, double inputB) {
		double sum = getDistanceFromHyperplane(e, w, inputB);
		if (sum > 0) {
			return 1.0;
		} else if (sum < 0) {
			return -1.0;
		} else {
			return 0;
		}
	}

	/**
	 * 
	 * @param e
	 * @param w
	 * @param inputB
	 * @return
	 */
	protected static double getDistanceFromHyperplane(Example e, HashMap<Integer, Double> w, double inputB) {
		double sum = inputB;

		// for(Integer featureIndex: w.keySet()){
		// only need to iterate over non-zero features
		for (Integer featureIndex : e.getFeatureSet()) {
			sum += w.get(featureIndex) * e.getFeature(featureIndex);
		}

		return sum;
	}

	/**
	 * Calculates the loss correction for each weight, and stores them into the
	 * weightsInProgress hashmap.
	 * 
	 * @param weightsInProgress hashmap to contain loss corrected weights
	 * @param e                 example
	 * @param lossDeriv         since derivative is constant for each
	 */
	private void calculateWeightsLossCorrection(HashMap<Integer, Double> weightsInProgress, Example e,
			double lossDeriv) {

		Set<Integer> features = e.getFeatureSet();
		double y_i = e.getLabel();
		// calculate loss correction based on loss type. we adjust the weights for every
		// feature before moving on to next example
		for (Integer feature : features) {
			double w_j = weights.get(feature);
			double x_ij = e.getFeature(feature);
			// Keep an in progress weight so that we can be consistent as we do
			// regularization
			weightsInProgress.put(feature, w_j + eta * y_i * x_ij * lossDeriv);
		}
		//
	}

	/**
	 * Calculates the regularization for each weight, and updates the weights field.
	 * Does the same for the bias and the b field.
	 * 
	 * @param weightsInProgress hashmap of loss corrected weights
	 * @param bInProgress       loss corrected bias
	 * @param e                 example
	 */
	private void calculateRegularization(HashMap<Integer, Double> weightsInProgress, double bInProgress, Example e) {

		Set<Integer> features = e.getFeatureSet();

		// calculate regularization and update the weights/bias.
		switch (regType) {
		case NO_REGULARIZATION:
			// If no regularization, regularization function is just 0, i.e. just (deep)
			// copy the weightsInProgress into the actual weights
			for (Integer feature : features) {
				weights.put(feature, weightsInProgress.get(feature));
			}
			b = bInProgress; // also copy the bias
			break;
		case L1_REGULARIZATION:
			// For L1, our weights are
			for (Integer feature : features) {
				weights.put(feature, weightsInProgress.get(feature) - eta * lambda * sign(weights.get(feature)));
			}
			b = bInProgress - eta * lambda * sign(b);
			break;
		case L2_REGULARIZATION:
			for (Integer feature : features) {
				weights.put(feature, weightsInProgress.get(feature) - eta * lambda * weights.get(feature));
			}
			b = bInProgress - eta * lambda * b;
			break;
		}

	}

	/**
	 * Gets the sign of w
	 * 
	 * @param w the weight whose sign to find
	 * @return -1 if w is negative, 1 if positive, 0 otherwise
	 */
	private int sign(double w) {
		return (w < 0) ? -1 : ((w > 0) ? 1 : 0);
	}

	/**
	 * Calculates loss function for a given y_i (label) and yPrime (w*x+b), i.e. for
	 * a given example
	 * 
	 * @param y_i
	 * @param yPrime
	 * @return loss function value
	 */
	private double loss(double y_i, double yPrime) {
		switch (lossType) {
		case EXPONENTIAL_LOSS:
			return Math.exp(-y_i * yPrime);
		case HINGE_LOSS:
			return Math.max(0, 1 - y_i * yPrime);
		default:
			return 0;
		}
	}

	/**
	 * @return string representation of the classifier
	 */
	public String toString() {
		StringBuffer buffer = new StringBuffer();

		ArrayList<Integer> temp = new ArrayList<Integer>(weights.keySet());
		Collections.sort(temp);

		for (Integer index : temp) {
			buffer.append(index + ":" + weights.get(index) + " ");
		}

		return buffer.substring(0, buffer.length() - 1);
	}
}
