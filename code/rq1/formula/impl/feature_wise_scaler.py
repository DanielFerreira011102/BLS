from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class FeatureWiseScaler(BaseEstimator, TransformerMixin):
    """A scikit-learn compatible transformer that scales each feature independently
    and works with partial feature sets.
    
    This scaler only normalizes features that are present in the data. If a feature
    doesn't exist in the input data, it's simply ignored rather than causing an error.
    
    Parameters
    ----------
    scaler_class : scikit-learn scaler class, default=StandardScaler
        The scikit-learn scaler class to use for scaling individual features.
    scaler_kwargs : dict, default=None
        Arguments to pass to the scaler constructor.
    """
    
    def __init__(self, scaler_class=None, scaler_kwargs=None):
        from sklearn.preprocessing import StandardScaler
        self.scaler_class = scaler_class or StandardScaler
        self.scaler_kwargs = scaler_kwargs or {}
        self.feature_scalers_ = {}
        self.feature_names_ = None
    
    def fit(self, X, y=None):
        """Fit the scaler for each feature individually.
        
        Parameters
        ----------
        X : array-like or pandas DataFrame
            The data to fit the scaler to.
        y : None
            Ignored.
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Convert to DataFrame if not already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Store feature names for future reference
        self.feature_names_ = X.columns.tolist()
        
        # Create and fit a scaler for each feature
        for feature in self.feature_names_:
            scaler = self.scaler_class(**self.scaler_kwargs)
            # Reshape to 2D array as required by scikit-learn
            scaler.fit(X[[feature]].values.reshape(-1, 1))
            self.feature_scalers_[feature] = scaler
            
        return self
    
    def transform(self, X):
        """Transform X using the fitted scalers for features that are present.
        
        Parameters
        ----------
        X : array-like or pandas DataFrame
            The data to transform.
            
        Returns
        -------
        X_transformed : pandas DataFrame
            The transformed data.
        """
        # Convert to DataFrame if not already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        X_transformed = X.copy()
        
        # Apply transformation only to features that exist in both the fitted model
        # and the input data
        for feature in set(self.feature_names_).intersection(X.columns):
            feature_data = X[[feature]].values.reshape(-1, 1)
            scaler = self.feature_scalers_[feature]
            X_transformed[feature] = scaler.transform(feature_data).flatten()
            
        return X_transformed
    
    def fit_transform(self, X, y=None):
        """Fit to data, then transform it.
        
        Parameters
        ----------
        X : array-like or pandas DataFrame
            The data to fit and transform.
        y : None
            Ignored.
            
        Returns
        -------
        X_transformed : pandas DataFrame
            The transformed data.
        """
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X):
        """Transform X back to the original scale.
        
        Parameters
        ----------
        X : array-like or pandas DataFrame
            The data to inverse transform.
            
        Returns
        -------
        X_original : pandas DataFrame
            The inverse transformed data.
        """
        # Convert to DataFrame if not already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X_original = X.copy()
        
        # Apply inverse transformation only to features that exist in both
        # the fitted model and the input data
        for feature in set(self.feature_names_).intersection(X.columns):
            feature_data = X[[feature]].values.reshape(-1, 1)
            scaler = self.feature_scalers_[feature]
            X_original[feature] = scaler.inverse_transform(feature_data).flatten()
            
        return X_original
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.
        
        Parameters
        ----------
        input_features : array-like of str or None, default=None
            If input_features is None, then self.feature_names_ is returned.
            
        Returns
        -------
        feature_names_out : ndarray of str
            Output feature names.
        """
        if input_features is None:
            return np.array(self.feature_names_)
        return np.array(input_features)