package ml;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import ml.classifiers.*;
import ml.data.*;

/**
 * Experimenter class for asmt 7.
 * 
 * @author xkoo
 * @author galzate
 */
public class ExperimenterAsmt7 {

	/**
	 * Main method, runs question experiments
	 * 
	 * @param args
	 *            unused
	 */
	public static void main(String[] args) {
		DataSet ds = new DataSet("./data/wines.train", DataSet.TEXTFILE);
		question1And2(ds, true);
		question1And2(ds, false);
		question3(ds);
		question4(ds);

	}

	/**
	 * Run tests for questions 1 and 2
	 * 
	 * @param ds
	 *            dataset
	 * @param usePositive
	 *            positive features only (true) or all features (false)
	 */
	private static void question1And2(DataSet ds, boolean usePositive) {
		// Positive features only
		NBClassifierOld nbc = new NBClassifierOld();
		nbc.setUseOnlyPositiveFeatures(usePositive);

		DataSetSplit dss = ds.split(0.8);
		DataSet trainDS = dss.getTrain();
		DataSet testDS = dss.getTest();

		List<Example> trainArr = trainDS.getData();
		List<Example> testArr = testDS.getData();

		// Only need to train once for changing lambdas
		nbc.train(trainDS);
		for (double i = 0.01; i < 0.25; i += 0.01) {
			nbc.setLambda(i);

			// Keep track of number correct
			int testCorrect = 0, trainCorrect = 0;
			for (Example e : testArr) {
				if (nbc.classify(e) == e.getLabel())
					testCorrect++;
			}
			for (Example e : trainArr) {
				if (nbc.classify(e) == e.getLabel())
					trainCorrect++;
			}

			System.out
					.println("lambda | testAcc | trainAcc: " + i + "\t" + (double) testCorrect / (double) testArr.size()
							+ "\t" + (double) trainCorrect / (double) trainArr.size());
		}

	}

	/**
	 * Run tests for question 3
	 * 
	 * @param ds
	 *            dataset
	 */
	private static void question3(DataSet ds) {
		// Comparing a positive features and an all features NB classifier
		NBClassifierOld nbcPos = new NBClassifierOld();
		NBClassifierOld nbcAll = new NBClassifierOld();
		nbcPos.setUseOnlyPositiveFeatures(true);
		nbcAll.setUseOnlyPositiveFeatures(false);

		System.out.println("pos,test\t\tpos,train\t\tall,test\t\tall,train");

		// 25 repetitions
		for (int i = 0; i < 25; i++) {

			int testCorrectPos = 0, trainCorrectPos = 0, testCorrectAll = 0, trainCorrectAll = 0;

			DataSetSplit dss = ds.split(0.8);
			DataSet trainDS = dss.getTrain();
			DataSet testDS = dss.getTest();

			List<Example> trainArr = trainDS.getData();
			List<Example> testArr = testDS.getData();

			// Retrain each time for the random split
			nbcPos.train(trainDS);
			nbcAll.train(trainDS);

			// Using best lambdas from 1 and 2.
			nbcPos.setLambda(.02);
			nbcAll.setLambda(.09);

			// Get testing accuracies
			for (Example e : testArr) {
				if (nbcPos.classify(e) == e.getLabel())
					testCorrectPos++;
				if (nbcAll.classify(e) == e.getLabel())
					testCorrectAll++;
			}

			// Get training accuracies
			for (Example e : trainArr) {
				if (nbcPos.classify(e) == e.getLabel())
					trainCorrectPos++;
				if (nbcAll.classify(e) == e.getLabel())
					trainCorrectAll++;
			}

			// Accuracies for each iteration
			System.out.println((double) testCorrectPos / (double) (testArr.size()) + "\t"
					+ (double) trainCorrectPos / (double) (trainArr.size()) + "\t"
					+ (double) testCorrectAll / (double) (testArr.size()) + "\t"
					+ (double) trainCorrectAll / (double) (trainArr.size()));

		}
	}

	/**
	 * Runs tests for question 4
	 * 
	 * @param ds
	 *            dataset
	 */
	private static void question4(DataSet ds) {
		NBClassifierOld nbc = new NBClassifierOld();
		nbc.setUseOnlyPositiveFeatures(true);
		nbc.setLambda(0.02);

		// list of prediction-confidence pairs along with the associated label
		List<double[]> triples = new ArrayList<>();

		// train model on 90/10 split
		DataSetSplit dss = ds.split(0.9);
		DataSet trainDS = dss.getTrain();
		DataSet testDS = dss.getTest();

		List<Example> testArr = testDS.getData();

		// Train classifier
		nbc.train(trainDS);

		// sort prediction-confidence-label triples by decreasing order of confidence
		for (Example e : testArr)
			triples.add(new double[] { nbc.classify(e), nbc.confidence(e), e.getLabel() });

		// anonymous function to compare confidences
		Collections.sort(triples, (double[] p1, double[] p2) -> {
			return ((Double) p2[1]).compareTo((Double) p1[1]);
		});
		List<Double> accuracies = new ArrayList<Double>(triples.size());

		// Keep track of accuracy
		int correct = 0, total = 0;
		for (double[] triple : triples) {

			// increment the number of correct classifications so far
			if (triple[0] == triple[2])
				correct++;

			accuracies.add((double) correct / (double) ++total);
		}

		// Print out the thresholds in order
		for (double[] triple : triples) {
			System.out.println(triple[1]);
		}
		// Print out the accuracies corresponding to those thresholds in order
		for (double d : accuracies)
			System.out.println(d);

	}

}