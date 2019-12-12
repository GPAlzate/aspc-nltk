package ml.classifiers;

/**
 * A class for generating classifiers of a specific type.
 * 
 * @author dkauchak
 *
 */
public class ClassifierFactory {
	// constants to use for picking which classifier this factory generates
	public static final int DECISION_TREE = 0;
	public static final int PERCEPTRON = 1;
	public static final int KNN = 2;

	private int classifierType = 0;  // what classifier we want to instantiate
	private int classifierParameter = 0; // what hyperparameter to set for the classifier
	private boolean setClassifierParameter = false;

	/**
	 * Create a new classifier factory.
	 * 
	 * @param classifierType The type of classifier this factory should generate.  The
	 * number should be input from one of the classifier constants.
	 */
	public ClassifierFactory(int classifierType){
		this.classifierType = classifierType;
	}

	/**
	 * Create a new classifier factory
	 * 
	 * @param classifierType
	 * @param hyperParameter The hyperparameter value to set for the classifier
	 */
	public ClassifierFactory(int classifierType, int hyperParameter){
		this.classifierType = classifierType;
		classifierParameter = hyperParameter;
		setClassifierParameter = true;
	}

	/**
	 * Get a new classifier of whatever type the factory was instantiated to create.
	 * 
	 * @return
	 */
	public Classifier getClassifier(){
		if( classifierType == 0 ){
			DecisionTreeClassifier dt = new DecisionTreeClassifier();
			
			if( setClassifierParameter ){
				dt.setDepthLimit(classifierParameter);
			}
			
			return dt;
		}else if( classifierType == 1 ){
			AveragePerceptronClassifier p = new AveragePerceptronClassifier();
			
			if( setClassifierParameter ){
				p.setIterations(classifierParameter);
			}
			
			return p;
		}else if( classifierType == 2 ){
			KNNClassifier knn = new KNNClassifier();
			
			if( setClassifierParameter ){
				knn.setK(classifierParameter);
			}
			
			return knn;
		}else{
			throw new RuntimeException("Classifier type not found: " + classifierType);
		}
	}
}
