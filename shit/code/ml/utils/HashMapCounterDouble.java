package ml.utils;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Class for counting occurrence of objects in a HashMap.
 * 
 * @author dkauchak
 *
 * @param <K>
 */
public class HashMapCounterDouble<K>{  // implements Map<K, Integer>{
	private HashMap<K, ChangeableDouble> map = new HashMap<K, ChangeableDouble>();
	
	/**
	 * Remove everything
	 */
	public void clear() {
		map.clear();
	}

	/**
	 * Check whether the key is contained in this map
	 * 
	 * @param key
	 * @return whether or not the key is in the map
	 */
	public boolean containsKey(Object key) {
		return map.containsKey(key);
	}

	/**
	 * Check whether the value is contained in this map
	 * 
	 * @param value
	 * @return
	 */
	public boolean containsValue(Object value) {
		return map.containsValue(value);
	}
	
	/**
	 * Get a sorted list of the entries in this map sorted by *values*
	 * 
	 * @return
	 */
	public ArrayList<Map.Entry<K, Double>> sortedEntrySet(){
		ArrayList<Map.Entry<K, Double>> list = new ArrayList<Map.Entry<K, Double>>();
		
		for( Map.Entry<K, ChangeableDouble> e: map.entrySet()){
			list.add(new AbstractMap.SimpleEntry<K, Double>(e.getKey(), e.getValue().getDouble()));
		}
		
		Collections.sort(list, new Comparator<Map.Entry<K, Double>>(){
			public int compare(Map.Entry<K, Double> e1, Map.Entry<K, Double> e2){
				return -e1.getValue().compareTo(e2.getValue());
			}
		});
		
		return list;
	}

	/**
	 * Get the count associated with this key
	 * 
	 * @param key
	 * @return
	 */
	public double get(Object key) {
		if( !map.containsKey(key) ){
			return 0;
		}else{
			return map.get(key).getDouble();
		}
	}

	public boolean isEmpty() {
		return map.isEmpty();
	}

	public Set<K> keySet() {
		return map.keySet();
	}

	/**
	 * Add the key/value pair
	 * 
	 * @param key
	 * @param value
	 */
	public void put(K key, double value) {
		map.put(key, new ChangeableDouble(value));
	}
	
	/**
	 * Increment the key by value.  If it doesn't exist, associate the key
	 * with the value.
	 * 
	 * @param key
	 * @param value
	 */
	public void increment(K key, double value){
		if( map.containsKey(key) ){
			map.get(key).increment(value);
		}else{
			map.put(key, new ChangeableDouble(value));
		}
	}

	/**
	 * Remove the entry associated with key
	 * 
	 * @param key
	 * @return
	 */
	public double remove(Object key) {
		return map.remove(key).getDouble();
	}

	/**
	 * @return number of elements in this map
	 */
	public int size() {
		return map.size();
	}
	
	/**
	 * Helper class supporting a double that you can increment and change
	 * 
	 * @author dkauchak
	 *
	 */
	private class ChangeableDouble{
		private double num;
		
		public ChangeableDouble(){
			num = 0;
		}
		
		public ChangeableDouble(double num){
			this.num = num;
		}
		
		public double getDouble(){
			return num;
		}
		
		public void increment(double value){
			num += value;
		}
		
		public void setDouble(double num){
			this.num = num;
		}
		
		public int compareTo(ChangeableDouble o){
			if( num < o.num ){
				return -1;
			}else if( num > o.num ){
				return 1;
			}else{
				return 0;
			}
		}
		
		public boolean equals(Object o){
			return num == ((ChangeableDouble)o).num;
		}
	}
}
