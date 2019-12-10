package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;

import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;
import ml.utils.HashMapCounter;
import ml.utils.Pair;

public class NBClassifier implements Classifier {

	private double lambda = 0.00;
	private DataSet data;
	private ArrayList<Example> dataArr;
	private HashMapCounter<String> hmap;
	private double prediction;
	private boolean pos = false;

	public NBClassifier() {
	}

	@Override
	public void train(DataSet data) {
		hmap = new HashMapCounter<String>();
		this.data = data;
		dataArr = data.getData();
		for (int i = 0; i < dataArr.size(); i++) { // loop to get each example
			Example example = dataArr.get(i);
			for (Integer feature : example.getFeatureSet()) { // loop to get each feature within
				if (example.getFeature(feature) != 0.0) {
					//System.out.println("increment");
					
					String pair = Integer.toString(feature) + "," + Double.toString(example.getLabel());
					hmap.increment(pair); // positive feature since available AND
																			// non-zero
					//System.out.println(hmap.get(pair));
				}
			}
		}
		
		
	}

	@Override
	public double classify(Example example) {

		Double max = -Double.MAX_VALUE;
		double prediction = 0.0;
		

		for (double label : data.getLabels()) {
			double prob = this.getLogProb(example, label);
			//System.out.println(prob);
			if (prob > max) {
				max = prob;
				prediction = label;
			}
		}
		//System.out.println(prediction +"pred" );
		this.prediction = prediction;
		return prediction;

	}

	@Override
	public double confidence(Example example) {

		double MLL = this.getLogProb(example, prediction);

		return MLL;
	}

	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

	public void setUseOnlyPositiveFeatures(boolean pos) {
		this.pos = pos;
	}

	public double getLogProb(Example ex, double label) {
		int labelCount = 0;
		for (int i = 0; i < dataArr.size(); i++) {
			Example e = dataArr.get(i);
			if (e.getLabel() == label) {
				labelCount++;
			}
		}
		
		double labelProb = labelCount / (double) dataArr.size(); // TODO: Check for correctness
		//System.out.println(labelProb+" hahaha");
		double sum = 0.0;
		for (Integer featureIndex : ex.getFeatureSet()) {
			double featureProb = this.getFeatureProb(featureIndex, label);
			//System.out.println(featureProb+"rub");
			sum += Math.log10(featureProb);
		}

		if (pos) {
			sum += Math.log10(labelProb);
			return sum;

		} else {
			for (Integer featureIndex2 : data.getAllFeatureIndices()) {
				if (!ex.getFeatureSet().contains(featureIndex2)) {
					double featureProb2 = 1 - this.getFeatureProb(featureIndex2, label);
					//System.out.println(featureProb2+"rub2");
					sum += Math.log10(featureProb2);
				}
			}
			sum += Math.log10(labelProb);
			return sum;
		}
	}

	public double getFeatureProb(int featureIndex, double label) {
		int labelCount = 0;
		for (int i = 0; i < dataArr.size(); i++) {
			Example e = dataArr.get(i);
			if (e.getLabel() == label) {
				labelCount++;
			}
		}
		String pair = Integer.toString(featureIndex) + "," + Double.toString(label);
		double featureProb = (hmap.get(pair) + lambda)
				/ (labelCount + data.getAllFeatureIndices().size() * lambda); // TODO: CHECK whether to use
																				// getALLFeatureIndices or
																				// exampleFeatureSet
		return featureProb;
	}

	public static void main(String[] args) {

		DataSet data = new DataSet("code/preprocessed.txt", DataSet.TEXTFILE);
		NBClassifier nb = new NBClassifier();
		DataSetSplit cvs = data.split(.8);
		
		
			DataSetSplit dataSplit = cvs;
			DataSet trainData = cvs.getTrain();
			DataSet testData = cvs.getTest();
			nb.setUseOnlyPositiveFeatures(true);
			nb.setLambda(0.02);
			nb.train(trainData);
			
			for (double j = 0.00; j < 0.03; j += 0.005) {
				nb.setLambda(j);
				int count = 0;
				for (int i = 0; i < testData.getData().size(); i++) {
					Example example = testData.getData().get(i);
					double pred = nb.classify(example);
					if (pred == example.getLabel()) {
						count++;
					}
				}
				
				System.out.println(j + "\t" + (double) count / (double) testData.getData().size());
			}
				
			
			
			
//			ArrayList<Double[]> arr = new ArrayList<Double[]>();
//			for (int j = 0; j < testData.getData().size(); j++) {
//				Example example = testData.getData().get(j);
//				double pred = nb.classify(example);
//				
//				double confidence = nb.confidence(example);
//				if (pred != example.getLabel()) {
//					Double[] newarr = new Double[2];
//					newarr[0] = 0.0;
//					newarr[1] = confidence;
//					arr.add(newarr);
//				}else {
//					count += 1;
//					Double[] newarr = new Double[2];
//					newarr[0] = 1.0;
//					newarr[1] = confidence;
//					arr.add(newarr);
//				}
//				System.out.println(count/ (double) testData.getData().size());
//			
//				
//				
				
//			}
//			Comparator<Double[]> byconf = 
//					(Double[] o1, Double[] o2)->o2[1].compareTo(o1[1]);
//			Collections.sort(arr, byconf);
//			String accuracies = "";
//			int correct = 0;
//			
//			for (int j = 0; j < arr.size(); j ++) {
//				double conf = arr.get(j)[1];
//				correct += arr.get(j)[0];
//			
//				double acc = (double) correct / (double) j;
//				System.out.println(conf + "," + acc);
//			}
			
			
		
		}
	
}
