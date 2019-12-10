package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;
import ml.utils.HashMapCounter;

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
					// System.out.println("increment");

					String pair = Integer.toString(feature) + "," + Double.toString(example.getLabel());
					hmap.increment(pair); // positive feature since available AND
											// non-zero
					// System.out.println(hmap.get(pair));
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
			// System.out.println(prob);
			if (prob > max) {
				max = prob;
				prediction = label;
			}
		}
		// System.out.println(prediction +"pred" );
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
		// System.out.println(labelProb+" hahaha");
		double sum = 0.0;
		for (Integer featureIndex : ex.getFeatureSet()) {
			double featureProb = this.getFeatureProb(featureIndex, label);
			// System.out.println(featureProb+"rub");
			sum += Math.log10(featureProb);
		}

		if (pos) {
			sum += Math.log10(labelProb);
			return sum;

		} else {
			for (Integer featureIndex2 : data.getAllFeatureIndices()) {
				if (!ex.getFeatureSet().contains(featureIndex2)) {
					double featureProb2 = 1 - this.getFeatureProb(featureIndex2, label);
					// System.out.println(featureProb2+"rub2");
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
		double featureProb = (hmap.get(pair) + lambda) / (labelCount + data.getAllFeatureIndices().size() * lambda); // TODO:
																														// CHECK
																														// whether
																														// to
																														// use
																														// getALLFeatureIndices
																														// or
																														// exampleFeatureSet
		return featureProb;
	}

	public static void main(String[] args) {

		DataSet data = new DataSet("code/preprocessed2.txt", DataSet.TEXTFILE);
		NBClassifier nb = new NBClassifier();
		DataSetSplit cvs = data.split(.8);

		DataSetSplit dataSplit = cvs;
		DataSet trainData = cvs.getTrain();
		DataSet testData = cvs.getTest();
		nb.setUseOnlyPositiveFeatures(true);
		nb.setLambda(0.02);

		System.out.println("training");

		nb.train(trainData);

		System.out.println("done training");

//		for (double j = 0.00; j < 0.03; j += 0.005) {
//			nb.setLambda(j);
//			int count = 0;
//		for (int i = 0; i < testData.getData().size(); i++) {
//			Example example = testData.getData().get(i);
//			double pred = nb.classify(example);
//			if (pred == example.getLabel()) {
//				count++;
//			}
//		}
//		System.out.println(j + "\t" + (double) count / (double) testData.getData().size());
//		
//	}
		
		// Time test
		ArrayList<Example> dataArray = data.getData();
		long start = System.currentTimeMillis();
		for (int i = 0; i < dataArray.size(); i++) {
			Example example = dataArray.get(i);
			nb.classify(example);
		}
		long end = System.currentTimeMillis();
		
		System.out.println("Time taken: " + (end - start));
		
		// Get positiveness/negativeness/neutralness of each word
		HashMap<Integer, String> fmap = trainData.getFeatureMap();

		HashMap<String, Double> positives = new HashMap<String, Double>();
		HashMap<String, Double> negatives = new HashMap<String, Double>();
		HashMap<String, Double> neutrals = new HashMap<String, Double>();

		for (String s : nb.hmap.keySet()) {
			String[] ss = s.split(",");
			String word = fmap.get(Integer.parseInt(ss[0]));
			double label = Double.parseDouble(ss[1]);
			if (label == 1) {
				positives.put(word, (double) nb.hmap.get(s));
			} else if (label == -1) {
				negatives.put(word, (double) nb.hmap.get(s));
			} else {
				neutrals.put(word, (double) nb.hmap.get(s));
			}
		}

		HashMap<String, Double> totalWordCountMap = new HashMap<String, Double>();

		// Get hashmap of totals so we can normalize scores
		ArrayList<HashMap<String, Double>> hashMaps = new ArrayList<HashMap<String, Double>>();
		hashMaps.add(positives);
		hashMaps.add(negatives);
		hashMaps.add(neutrals);
		for (HashMap<String, Double> hashMap : hashMaps) {
			for (String s : hashMap.keySet()) {
				// 0 if null
				Double posCount = (positives.get(s) != null) ? positives.get(s) : 0;
				Double negCount = (negatives.get(s) != null) ? negatives.get(s) : 0;
				Double neutCount = (neutrals.get(s) != null) ? neutrals.get(s) : 0;

				if (totalWordCountMap.get(s) == null)
					totalWordCountMap.put(s, posCount + negCount + neutCount);
			}
		}

		// Replace counts with probabilities of each word
		for (HashMap<String, Double> hashMap : hashMaps) {
			for (String s : hashMap.keySet()) {
				double count = hashMap.get(s);
				double denom = totalWordCountMap.get(s);

				hashMap.put(s, count / denom);
			}
		}

		nb.train(data);

		LinkedHashMap<String, Double> sortedPositives = positives.entrySet().stream()
				.sorted(Map.Entry.comparingByValue()).collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue,
						(oldValue, newValue) -> oldValue, LinkedHashMap::new));

		LinkedHashMap<String, Double> sortedNegatives = negatives.entrySet().stream()
				.sorted(Map.Entry.comparingByValue()).collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue,
						(oldValue, newValue) -> oldValue, LinkedHashMap::new));

		LinkedHashMap<String, Double> sortedNeutrals = neutrals.entrySet().stream().sorted(Map.Entry.comparingByValue())
				.collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (oldValue, newValue) -> oldValue,
						LinkedHashMap::new));

//		for (String s : sortedPositives.keySet()) {
//			System.out.println(s + "\t" + sortedPositives.get(s));
//		}
//		for (String s : sortedNegatives.keySet()) {
//			System.out.println(s + "\t" + sortedNegatives.get(s));
//		}
//		for (String s : sortedNeutrals.keySet()) {
//			System.out.println(s + "\t" + sortedNeutrals.get(s));
//		}

		// Tricky sentences with negation and idiomatic phrases
		String[] trickySentences = { "its a love hate relationship", "the class is fucking cool",
				"this kung fu class is cool", "must take class professor is very insightful about war and violence",
				"sentiment analysis has never been good", "sentiment analysis has never been this good",
				"most automated sentiment analysis tools are shit", "other sentiment analysis tools can be quite bad",
				"without a doubt excellent idea", "not such a badass after all", "without a doubt an excellent idea",
				"this class is an absolute disaster", "this class is a disaster" };

		// Reverse the index to feature hashmap
		HashMap<String, Integer> wordToFeatureIndex = new HashMap<String, Integer>();
		for (Integer i : fmap.keySet()) {
			wordToFeatureIndex.put(fmap.get(i), i);
		}

		// Construct test examples
		for (String sentence : trickySentences) {
			String[] words = sentence.split(" ");
			HashMapCounter<String> wordMap = new HashMapCounter<String>();
			for (String w : words) {
				wordMap.increment(w);
			}
			Example e = new Example();
			for (String w : wordMap.keySet()) {
//				System.out.println(w + ": " + wordToFeatureIndex.get(w));
				e.addFeature(wordToFeatureIndex.get(w), wordMap.get(w));
			}

			double pred = nb.classify(e);

			System.out.println(pred + "\t" + sentence);
		}

//		ArrayList<Double[]> arr = new ArrayList<Double[]>();
//		for (int j = 0; j < testData.getData().size(); j++) {
//			Example example = testData.getData().get(j);
//			double pred = nb.classify(example);
//
//			double confidence = nb.confidence(example);
//			if (pred != example.getLabel()) {
//				Double[] newarr = new Double[2];
//				newarr[0] = 0.0;
//				newarr[1] = confidence;
//				arr.add(newarr);
//			} else {
//				count += 1;
//				Double[] newarr = new Double[2];
//				newarr[0] = 1.0;
//				newarr[1] = confidence;
//				arr.add(newarr);
//			}
//			System.out.println(count / (double) testData.getData().size());
//
//		}
//		Comparator<Double[]> byconf = (Double[] o1, Double[] o2) -> o2[1].compareTo(o1[1]);
//		Collections.sort(arr, byconf);
//		String accuracies = "";
//		int correct = 0;
//
//		for (int j = 0; j < arr.size(); j++) {
//			double conf = arr.get(j)[1];
//			correct += arr.get(j)[0];
//
//			double acc = (double) correct / (double) j;
//			System.out.println(conf + "," + acc);
//		}

	}

}
