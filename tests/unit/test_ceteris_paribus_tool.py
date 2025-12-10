"""
Tests for CeterisParibusTool

Based on usage patterns from ethics_compliance_pilot.ipynb
""" 
from __future__ import annotations
from tests.config import PROJECT_DIR
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from src.tools.xai import CeterisParibusTool
from sklearn.preprocessing import OneHotEncoder


@pytest.fixture
def sample_data():
    """Create sample dataset with mixed numeric and non-numeric features"""
    np.random.seed(42)
    n_samples = 100
    
    data = pd.DataFrame({
        'numeric_feature_1': np.random.randn(n_samples),
        'numeric_feature_2': np.random.randn(n_samples),
        'numeric_feature_3': np.random.randn(n_samples),
        'categorical_feature': np.random.choice(['A', 'B', 'C'], n_samples),
        'string_feature': [f'text_{i}' for i in range(n_samples)],
        'target': np.random.randn(n_samples)
    })
        
    # === Identify categorical columns ===
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = data.select_dtypes(exclude=['object', 'category']).columns.tolist()

    # === One-hot encode categorical features ===
    if categorical_cols:
        # Initialize OneHotEncoder
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # Fit on training data and transform both train and test
        data_cat_encoded = ohe.fit_transform(data[categorical_cols])
        
        # Create DataFrames with encoded features
        cat_feature_names = ohe.get_feature_names_out(categorical_cols)
        data_cat_df = pd.DataFrame(data_cat_encoded, columns=cat_feature_names, index=data.index)
        
        # Combine numerical and encoded categorical features
        data = pd.concat([data[numerical_cols].reset_index(drop=True), 
                                data_cat_df.reset_index(drop=True)], axis=1)
        
    return data


@pytest.fixture
def trained_model(sample_data):
    """Train a model on all features (numeric + encoded non-numeric)"""
    from sklearn.preprocessing import LabelEncoder
    
    # Prepare all features for model training
    X = sample_data.drop(columns=['target'])    
    y = sample_data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Return model and test indices to match with full dataset
    return model, X_test.index


@pytest.fixture
def tool(trained_model, sample_data):
    """Create a CeterisParibusTool instance with full dataset (including non-numeric features)"""
    model, test_indices = trained_model
    
    # Tool receives full dataset with all original features (mixed types)
    # The tool will filter to only explain numeric features - what it doesn't need is ignored
    X_test_full = sample_data.loc[test_indices, :].drop(columns=['target'])
    
    return CeterisParibusTool(
        model=model,
        data=X_test_full,  # Full dataset - tool uses what it needs, ignores the rest
        max_num_datapoints=10
    )


@pytest.fixture
def tool_large_max(trained_model, sample_data):
    """Create a CeterisParibusTool instance with large max_num_datapoints and full dataset"""
    model, test_indices = trained_model
    
    # Tool receives full dataset with all original features (mixed types)
    X_test_full = sample_data.loc[test_indices, :].drop(columns=['target'])
    
    return CeterisParibusTool(
        model=model,
        data=X_test_full,  # Full dataset - tool uses what it needs, ignores the rest
        max_num_datapoints=100  # Much larger than X_test size
    )


class TestCeterisParibusToolInitialization:
    """Test tool initialization and explainable features detection"""
    
    def test_tool_initialization(self, tool):
        """Test that tool can be initialized with model, data, and max_datapoints"""
        assert tool.model is not None
        assert tool.data is not None
        assert tool.max_num_datapoints == 10
    
    def test_explainable_features_property(self, tool):
        """Test that _explainable_features property identifies numeric features"""
        explainable = tool._explainable_features
        
        # Should only contain numeric features
        assert isinstance(explainable, list)
        assert len(explainable) > 0
        
        # All features should be numeric
        for feat in explainable:
            assert feat in tool.data.columns
            assert np.issubdtype(tool.data[feat].dtype, np.number)
    
    def test_explainable_features_excludes_non_numeric(self, tool):
        """Test that non-numeric features are excluded from explainable features"""
        explainable = tool._explainable_features
        
        # Should not include categorical or string features
        assert 'categorical_feature' not in explainable
        assert 'string_feature' not in explainable
        assert 'target' not in explainable
        
        # Should only include numeric features
        assert 'numeric_feature_1' in explainable
        assert 'numeric_feature_2' in explainable
        assert 'numeric_feature_3' in explainable
        
        # Verify tool.data contains non-numeric features (they're just not explainable)
        assert 'categorical_feature' in tool.data.columns
        assert 'string_feature' in tool.data.columns
    
    def test_explainable_features_caching(self, tool):
        """Test that _explainable_features is computed once and cached"""
        # First access
        features1 = tool._explainable_features
        
        # Second access should return same list (cached)
        features2 = tool._explainable_features
        
        assert features1 is features2
        assert features1 == features2


class TestCeterisParibusToolRun:
    """Test the _run method of the tool"""
    
    def test_run_with_valid_numeric_feature(self, tool):
        """Test running tool with a valid numeric feature"""
        explainable = tool._explainable_features
        # if not explainable:
        #     pytest.skip("No explainable features in test data")
        
        feature_name = explainable[0]
        img_bytes, df = tool._run(feature_name, num_datapoints=3)
        
        # Should return a dictionary
        assert isinstance(img_bytes, bytes)
        assert isinstance(df, pd.DataFrame)
        
        
        # Each entry should be a tuple of (bytes, DataFrame)
        assert len(df.columns) == 3  # row_id, feature_name, prediction
        assert 'row_id' in df.columns
        assert feature_name in df.columns
        assert 'prediction' in df.columns
    
    def test_run_with_invalid_feature_name(self, tool):
        """Test running tool with a feature that doesn't exist"""
        result = tool._run("nonexistent_feature", num_datapoints=3)
        
        # Should return an error message
        assert isinstance(result, str)
        assert "ERROR" in result
        assert "not found" in result.lower()
    
    def test_run_with_non_numeric_feature(self, tool, sample_data):
        """Test running tool with a non-numeric feature"""
        # Check if we have non-numeric features in the original sample_data
        non_numeric_cols = [
            col for col in sample_data.columns
            if not np.issubdtype(sample_data[col].dtype, np.number)
        ]
        
        # Use a non-numeric feature name (even if not in tool.data, it should be rejected)
        feature_name = non_numeric_cols[0]
        result = tool._run(feature_name, num_datapoints=3)
        
        # Should return an error message
        assert isinstance(result, str)
        assert "ERROR" in result
        # Either "not found" (if not in tool.data) or "cannot be explained" (if in tool.data but not numeric)
        assert "not found" in result.lower() or "cannot be explained" in result.lower() or "numeric" in result.lower()
    
    def test_run_caps_num_datapoints(self, tool):
        """Test that num_datapoints is capped at max_num_datapoints"""
        explainable = tool._explainable_features
        # if not explainable:
        #     pytest.skip("No explainable features in test data")
        
        feature_name = explainable[0]
        
        # Request more datapoints than max
        result = tool._run(feature_name, num_datapoints=100)
        
        # Should return tuple (bytes, DataFrame) or error string
        if isinstance(result, str):
            pytest.fail(f"Got error: {result}")
        
        img_bytes, df = result
        
        # Should still work (capped internally)
        assert isinstance(img_bytes, bytes)
        assert isinstance(df, pd.DataFrame)
        assert feature_name in df.columns
        
        # Check that the actual number of datapoints used is capped
        unique_row_ids = df['row_id'].unique()
        assert len(unique_row_ids) <= tool.max_num_datapoints
    
    def test_run_caps_num_datapoints_at_data_length(self, tool):
        """Test that num_datapoints is also capped at data length"""
        explainable = tool._explainable_features
        # if not explainable:
        #     pytest.skip("No explainable features in test data")
        
        feature_name = explainable[0]
        data_length = len(tool.data)
        
        # Request more datapoints than available data
        result = tool._run(feature_name, num_datapoints=data_length + 100)
        
        # Should return tuple (bytes, DataFrame) or error string
        if isinstance(result, str):
            pytest.fail(f"Got error: {result}")
        
        # Check actual number of datapoints used
        bytes_png, df = result
        assert isinstance(bytes_png, bytes)
        assert isinstance(df, pd.DataFrame)
        unique_row_ids = df['row_id'].unique()
        assert len(unique_row_ids) <= data_length
    
    def test_run_with_minimum_datapoints(self, tool):
        """Test that num_datapoints is at least 1"""
        explainable = tool._explainable_features
        # if not explainable:
        #     pytest.skip("No explainable features in test data")
        
        feature_name = explainable[0]
        
        # Request 0 or negative datapoints
        result = tool._run(feature_name, num_datapoints=0)
        
        # Should return tuple (bytes, DataFrame) or error string
        if isinstance(result, str):
            pytest.fail(f"Got error: {result}")
        
        bytes_png, df = result
        
        # Should still work (capped to 1)
        assert isinstance(bytes_png, bytes)
        assert isinstance(df, pd.DataFrame)
        assert feature_name in df.columns
    
    def test_run_uses_all_datapoints_when_data_smaller_than_max(self, tool_large_max):
        """Test that when dataset has fewer rows than max_num_datapoints, all datapoints are used"""
        explainable = tool_large_max._explainable_features
        # if not explainable:
        #     pytest.skip("No explainable features in test data")
        
        feature_name = explainable[0]
        data_length = len(tool_large_max.data)
        
        # Request more datapoints than available, but less than max
        result = tool_large_max._run(feature_name, num_datapoints=tool_large_max.max_num_datapoints)
        
        # Should return tuple (bytes, DataFrame) or error string
        if isinstance(result, str):
            pytest.fail(f"Got error: {result}")
        
        bytes_png, df = result
        
        # Verify all available datapoints are used
        unique_row_ids = df['row_id'].unique()
        assert len(unique_row_ids) == data_length, \
            f"Expected {data_length} datapoints, got {len(unique_row_ids)}"
        
        # Verify row_ids are 0 to (data_length - 1)
        assert set(unique_row_ids) == set(range(data_length)), \
            f"Row IDs should be 0 to {data_length - 1}, got {set(unique_row_ids)}"
        
        # Verify the DataFrame has the expected number of rows
        # (data_length datapoints * grid_points per datapoint)
        expected_rows = data_length * tool_large_max.grid_points
        assert len(df) == expected_rows, \
            f"Expected {expected_rows} rows (={data_length} datapoints * {tool_large_max.grid_points} grid_points), got {len(df)}"


class TestCeterisParibusToolAsync:
    """Test the async _arun method"""
    
    @pytest.mark.asyncio
    async def test_arun_with_valid_feature(self, tool):
        """Test async run with a valid feature"""
        explainable = tool._explainable_features
        # if not explainable:
        #     pytest.skip("No explainable features in test data")
        
        feature_name = explainable[0]
        result = await tool._arun(feature_name, num_datapoints=3)
        
        # Should return tuple (bytes, DataFrame) or error string
        if isinstance(result, str):
            pytest.fail(f"Got error: {result}")
        
        img_bytes, df = result
        assert isinstance(img_bytes, bytes)
        assert isinstance(df, pd.DataFrame)
        assert feature_name in df.columns
    
    @pytest.mark.asyncio
    async def test_arun_with_invalid_feature(self, tool):
        """Test async run with invalid feature"""
        result = await tool._arun("nonexistent_feature", num_datapoints=3)
        
        # Should return error message
        assert isinstance(result, str)
        assert "ERROR" in result


class TestCeterisParibusToolIntegration:
    """Integration tests with real data patterns from notebook"""
    
    def test_tool_with_boston_housing_pattern(self):
        """Test tool with data pattern similar to Boston housing dataset"""
        # Create synthetic data similar to Boston housing
        np.random.seed(42)
        n_samples = 50
        
        data = pd.DataFrame({
            'CRIM': np.random.rand(n_samples) * 10,
            'ZN': np.random.rand(n_samples) * 100,
            'INDUS': np.random.rand(n_samples) * 30,
            'CHAS': np.random.choice([0, 1], n_samples),  # Binary numeric
            'NOX': np.random.rand(n_samples),
            'RM': np.random.rand(n_samples) * 10,
            'AGE': np.random.rand(n_samples) * 100,
            'DIS': np.random.rand(n_samples) * 10,
            'RAD': np.random.randint(1, 25, n_samples),
            'TAX': np.random.randint(200, 800, n_samples),
            'PTRATIO': np.random.rand(n_samples) * 10,
            'B': np.random.rand(n_samples) * 400,
            'LSTAT': np.random.rand(n_samples) * 40,
        })
        
        # Train a simple model
        X_train, X_test, y_train, y_test = train_test_split(
            data, np.random.randn(len(data)), test_size=0.3, random_state=42
        )
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Create tool
        tool = CeterisParibusTool(
            model=model,
            data=X_test,
            max_num_datapoints=5
        )
        
        # Test with a feature from the dataset
        result = tool._run('CRIM', num_datapoints=3)
        
        # Should return tuple (bytes, DataFrame) or error string
        if isinstance(result, str):
            pytest.fail(f"Got error: {result}")
        
        # Check structure
        png_bytes, df = result
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0  # Should have some image data
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'CRIM' in df.columns
    
    def test_tool_error_handling(self, tool):
        """Test that tool handles errors gracefully"""
        explainable = tool._explainable_features
        # if not explainable:
        #     pytest.skip("No explainable features in test data")
        
        # Create a mock model that will raise an error
        class BadModel:
            def predict(self, X):
                raise ValueError("Model error")
        
        bad_tool = CeterisParibusTool(
            model=BadModel(),
            data=tool.data,
            max_num_datapoints=5
        )
        
        result = bad_tool._run(explainable[0], num_datapoints=3)
        
        # Should return error message, not raise exception
        assert isinstance(result, str)
        assert "Error" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

