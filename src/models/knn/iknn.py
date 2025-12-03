import sys
sys.path.insert(0, '/home/hygo2025/Development/projects/fermi/session-rec-lib')

from algorithms.knn.iknn import ItemKNN as BaseItemKNN


class ItemKNN(BaseItemKNN):
    """
    Wrapper for ItemKNN to adapt fit() signature to session-rec-lib expectations.
    The base ItemKNN.fit() only accepts 'data' but run_config.py calls fit(train, test).
    """
    
    def fit(self, data, test=None):
        """
        Trains the predictor. Ignores test parameter to maintain compatibility.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data with session IDs, item IDs and timestamps.
        test: pandas.DataFrame (optional)
            Test data - ignored but accepted for compatibility with run_config.py
        """
        super().fit(data)
