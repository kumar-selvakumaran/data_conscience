from langchain_core.tools import BaseTool
from typing import Dict, Tuple, Any, List, Optional
import pandas as pd
import numpy as np
from pydantic import Field
from functools import cached_property

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO



class CeterisParibusTool(BaseTool):
    """
    Generate Ceteris Paribus (ICE) plots for a specified feature and number of datapoints.
    Returns a dictionary mapping feature names to tuples of (PNG bytes, DataFrame).
    Only numeric features can be explained using ICE plots.
    """
    name: str = "ceteris_paribus_plot"
    
    description: str = (
        "Generate Ceteris Paribus (ICE) plots for a specified feature. "
        "Takes a feature name from the dataset and number of datapoints. "
        "Only numeric features can be explained using ICE plots. "
        "Returns a dictionary with feature names as keys and tuples of (PNG image bytes, DataFrame) as values."
    )

    model: Any = Field(..., description="The fitted model")
    data: pd.DataFrame = Field(..., description="The dataset")
    max_num_datapoints: int = Field(..., description="Maximum number of datapoints allowed")
    grid_points: int = Field(50, description="Grid resolution per feature")
    
    @cached_property
    def _explainable_features(self) -> List[str]:
        """
        This property creates the '_explainable_features' attribute upon first access
        and initializes it with the list of features that can be explained using Ceteris Paribus (ICE) technique (numeric features only).
        On the first call, it determines which columns of the dataset are numeric and stores them as the attribute.
        
        Returns:
            List[str]: The names of numeric features that can be explained.
        """
        numeric_features = [
            col for col in self.data.columns 
            if np.issubdtype(self.data[col].dtype, np.number)
        ]
        return numeric_features
    
    
    def ceteris_paribus_bytes(
        self,
        feature_name: str,
        num_datapoints: int = 1,
    ) -> Dict[str, Tuple[bytes, pd.DataFrame]]:
        """
        Generate a Ceteris Paribus (ICE) plot for the given numeric feature using up to
        'num_datapoints' samples from the dataset. Plots are returned as PNG image bytes
        along with the underlying DataFrame used for plotting.

        Args:
            feature_name (str): The name of the numeric feature to plot.
            num_datapoints (int): Number of rows from the top of the dataset to use for ICE computation.

        Returns:
            Tuple[bytes, pd.DataFrame]: A tuple containing (PNG image bytes, DataFrame with columns
            ['row_id', feature_name, 'prediction']) for the specified feature.
        """
        num_datapoints = max(1, min(num_datapoints, len(self.data)))
        X_targets = self.data.iloc[:num_datapoints].copy()
        # Binary probability if available
        proba_fn = getattr(self.model, "predict_proba", None)
        is_binary = False
        if proba_fn is not None:
            classes_ = getattr(self.model, "classes_", None)
            is_binary = classes_ is not None and len(classes_) == 2

        if feature_name not in self._explainable_features:
            raise ValueError(f"Feature '{feature_name}' is not numeric")

        grid = np.linspace(self.data[feature_name].min(), self.data[feature_name].max(), self.grid_points)

        base = np.repeat(X_targets.values, self.grid_points, axis=0)
        base_df = pd.DataFrame(base, columns=self.data.columns)
        base_df[feature_name] = np.tile(grid, num_datapoints)

        if proba_fn is not None and is_binary:
            preds = self.model.predict_proba(base_df)[:, 1]
        else:
            preds = self.model.predict(base_df)
        
        row_ids = np.repeat(np.arange(num_datapoints), self.grid_points)
        df_plot = pd.DataFrame(
            {"row_id": row_ids, feature_name: base_df[feature_name].to_numpy(), "prediction": preds.astype(float)}
        )

        # Plot one line per row_id + dashed mean curve
        fig = plt.figure()
        ax = fig.gca()
        for rid in range(num_datapoints):
            sub = df_plot[df_plot["row_id"] == rid]
            ax.plot(sub[feature_name], sub["prediction"], alpha=0.9, linewidth=1.6)
        mean_curve = df_plot.groupby(feature_name, as_index=False)["prediction"].mean()
        ax.plot(mean_curve[feature_name], mean_curve["prediction"], linestyle="--", linewidth=2.0)

        ax.set_xlabel("feature value")
        ax.set_ylabel("predicted probability for target" if is_binary else "target value")
        ax.set_title(f"Ceteris Paribus Plot (n={num_datapoints})- Feature Name : '{feature_name}'")
        ax.grid(True, alpha=0.3)

        # Serialize figure to PNG bytes
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
        plt.close(fig)
        png_bytes = buf.getvalue()
        buf.close()
        
        return png_bytes, df_plot

    def _run(self, feature_name: str, num_datapoints: int) -> Dict[str, Tuple[bytes, pd.DataFrame]]:
        """
        Generate Ceteris Paribus plots.
        
        Args:
            feature_name: Name of the feature to analyze (must be a numeric feature in the dataset)
            num_datapoints: Number of datapoints for which curves need to be drawn (capped at max_num_datapoints)
        
        Returns:
            Dictionary mapping feature names to tuples of (PNG bytes, DataFrame)
        """
        # Validate feature name is explainable
        if feature_name not in self._explainable_features:
            if feature_name not in self.data.columns:
                return f"ERROR: Feature '{feature_name}' not found in dataset. Available features: {list(self.data.columns)}"
            else:
                return (
                    f"ERROR: Feature '{feature_name}' cannot be explained using ICE plots. "
                    f"Only numeric features are supported. Explainable features: {self._explainable_features}"
                )
        
        # Cap num_datapoints at the maximum allowed
        num_datapoints = max(1, min(num_datapoints, self.max_num_datapoints, len(self.data)))
        
        try:
            # Call the function
            result = self.ceteris_paribus_bytes(
                feature_name=feature_name,
                num_datapoints=num_datapoints
            )
            
            return result
            
        except Exception as e:
            return f"Error generating Ceteris Paribus plots: {e}"
    
    async def _arun(self, feature_name: str, num_datapoints: int) -> Dict[str, Tuple[bytes, pd.DataFrame]]:
        """Async version - simple shim"""
        return self._run(feature_name, num_datapoints)