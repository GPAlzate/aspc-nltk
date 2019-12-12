package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.PriorityQueue;

import ml.data.DataSet;
import ml.data.Example;
import ml.utils.HashMapCounter;

public class KNNClassifier implements Classifier {
	private DataSet train;
	private int k = 5;
	
	@Override
	public void train(DataSet data) {
		train = data;
		// done!
	}

	@Override
	public double classify(Example example) {
		//PriorityQueue<Double> q = new PriorityQueue<Double>(k);
		return getLabel(example)[0];
	}
	
	@Override
	public double confidence(Example example) {
		return getLabel(example)[0];
	}
	
	private double[] getLabel(Example example){
		ArrayList<ScoredExample> distances = new ArrayList<ScoredExample>();
		
		for( Example e: train.getData() ){
			distances.add(new ScoredExample(e, getDistance(example, e)));
		}
		
		Collections.sort(distances);
		
		HashMapCounter<Double> counter = new HashMapCounter<Double>();
		
		for( int i = 0; i < k; i++ ){
			counter.increment(distances.get(i).e.getLabel());
		}
		
		double maxCount = -1;
		double maxPrediction = 0.0;
		
		for( Double d: counter.keySet() ){
			int count = counter.get(d);
			
			if( count > maxCount ){
				maxCount = count;
				maxPrediction = d;
			}
		}
		
		double[] temp = new double[2];
		temp[0] = maxPrediction;
		temp[1] = maxCount/(double)k;
		return temp;
	}
	
	private double getDistance(Example e1, Example e2){
		double dist = 0.0;
		
		for( Integer featureNum: train.getAllFeatureIndices() ){
			double diff = e1.getFeature(featureNum) - e2.getFeature(featureNum);
			dist += diff*diff;
		}
		
		return Math.sqrt(dist);
	}
	
	public void setK(int k){
		this.k = k;
	}

	private class ScoredExample implements Comparable<ScoredExample>{
		public double distance;
		public Example e;
		
		public ScoredExample(Example e, double distance){
			this.e = e;
			this.distance = distance;
		}

		@Override
		public int compareTo(ScoredExample o) {
			// we want smallest first (i.e. closest distances)
			return Double.compare(distance, o.distance);
		}
	}
}
