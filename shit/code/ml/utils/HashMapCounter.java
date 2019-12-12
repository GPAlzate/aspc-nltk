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
public class HashMapCounter<K>{  // implements Map<K, Integer>{
	private HashMap<K, ChangeableInteger> map = new HashMap<K, ChangeableInteger>();
	
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
	public ArrayList<Map.Entry<K, Integer>> sortedEntrySet(){
		ArrayList<Map.Entry<K, Integer>> list = new ArrayList<Map.Entry<K, Integer>>();
		
		for( Map.Entry<K, ChangeableInteger> e: map.entrySet()){
			list.add(new AbstractMap.SimpleEntry<K, Integer>(e.getKey(), e.getValue().getInt()));
		}
		
		Collections.sort(list, new Comparator<Map.Entry<K, Integer>>(){
			public int compare(Map.Entry<K, Integer> e1, Map.Entry<K, Integer> e2){
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
	public int get(Object key) {
		if( !map.containsKey(key) ){
			return 0;
		}else{
			return map.get(key).getInt();
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
	public void put(K key, int value) {
		map.put(key, new ChangeableInteger(value));
	}
	
	/**
	 * Increment the key by 1
	 * 
	 * @param key
	 */
	public void increment(K key){
		increment(key, 1);
	}
	
	/**
	 * Increment the key by value.  If it doesn't exist, associate the key
	 * with the value.
	 * 
	 * @param key
	 * @param value
	 */
	public void increment(K key, int value){
		if( map.containsKey(key) ){
			map.get(key).increment(value);
		}else{
			map.put(key, new ChangeableInteger(value));
		}
	}

	/**
	 * Remove the entry associated with key
	 * 
	 * @param key
	 * @return
	 */
	public int remove(Object key) {
		return map.remove(key).getInt();
	}

	/**
	 * @return number of elements in this map
	 */
	public int size() {
		return map.size();
	}
	
	/**
	 * Helper class supporting an int that you can increment and change
	 * 
	 * @author dkauchak
	 *
	 */
	private class ChangeableInteger{
		private int num;
		
		public ChangeableInteger(){
			num = 0;
		}
		
		public ChangeableInteger(int num){
			this.num = num;
		}
		
		public void increment( int val ){
			num += val;
		}
		
		public int getInt(){
			return num;
		}
		
		public void setInt(int num){
			this.num = num;
		}
		
		public int compareTo(ChangeableInteger o){
			if( num < o.num ){
				return -1;
			}else if( num > o.num ){
				return 1;
			}else{
				return 0;
			}
		}
		
		public boolean equals(Object o){
			return num == ((ChangeableInteger)o).num;
		}
	}
}
