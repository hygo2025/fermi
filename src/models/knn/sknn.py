import sys
sys.path.insert(0, '/home/hygo2025/Development/projects/fermi/session-rec-lib')

from algorithms.knn.sknn import ContextKNN as BaseContextKNN


class ContextKNN(BaseContextKNN):
    """
    Wrapper for ContextKNN to fix sessions_for_item() returning None.
    
    The base implementation returns None when item doesn't exist in item_session_map,
    which causes TypeError when using set union operator (|).
    
    This wrapper overrides sessions_for_item() to return empty set() instead of None.
    """
    
    def sessions_for_item(self, item_id):
        """
        Returns all sessions for an item, or empty set if item not found.
        
        Parameters
        --------
        item_id : int or string
            ID of the item
        
        Returns 
        --------
        out : set
            Set of session IDs that contain this item, or empty set if item not in training data
        """
        return self.item_session_map.get(item_id) if item_id in self.item_session_map else set()
