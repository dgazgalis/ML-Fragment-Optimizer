"""
Biological assay data processing utilities.

This module provides functions for processing plate reader outputs,
quality control, replicate handling, normalization, and hit calling.
"""

from typing import Optional, List, Dict, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from dataclasses import dataclass
from enum import Enum

try:
    from scipy import stats
    from scipy.optimize import curve_fit
except ImportError:
    raise ImportError("SciPy is required. Install with: pip install scipy")


class WellFormat(Enum):
    """Plate format types."""
    PLATE_96 = 96
    PLATE_384 = 384
    PLATE_1536 = 1536


@dataclass
class PlateQCMetrics:
    """Quality control metrics for a plate."""
    z_factor: float
    z_prime_factor: float
    signal_to_noise: float
    signal_to_background: float
    cv_positive: float
    cv_negative: float
    n_outliers: int
    passed: bool


class PlateReaderParser:
    """Parse various plate reader output formats."""

    @staticmethod
    def parse_envision(filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Parse Envision plate reader output.

        Args:
            filepath: Path to Envision output file

        Returns:
            DataFrame with well data
        """
        # This is a simplified parser - real implementations would need
        # to handle various Envision formats
        df = pd.read_csv(filepath, skiprows=2)
        return df

    @staticmethod
    def parse_generic_matrix(
        filepath: Union[str, Path],
        plate_format: WellFormat = WellFormat.PLATE_96
    ) -> pd.DataFrame:
        """
        Parse generic matrix format (rows = A-H, cols = 1-12).

        Args:
            filepath: Path to file
            plate_format: Plate format

        Returns:
            DataFrame in long format
        """
        # Read matrix
        matrix_df = pd.read_csv(filepath, index_col=0)

        # Convert to long format
        data = []
        for row_letter in matrix_df.index:
            for col_num in matrix_df.columns:
                value = matrix_df.loc[row_letter, col_num]
                well = f"{row_letter}{col_num}"
                data.append({
                    'well': well,
                    'row': row_letter,
                    'column': int(col_num),
                    'value': value
                })

        return pd.DataFrame(data)

    @staticmethod
    def parse_long_format(
        filepath: Union[str, Path],
        well_col: str = "well",
        value_col: str = "value"
    ) -> pd.DataFrame:
        """
        Parse long format file.

        Args:
            filepath: Path to file
            well_col: Column name for wells
            value_col: Column name for values

        Returns:
            DataFrame with well data
        """
        df = pd.read_csv(filepath)

        # Ensure required columns
        if well_col not in df.columns or value_col not in df.columns:
            raise ValueError(f"File must contain '{well_col}' and '{value_col}' columns")

        # Standardize column names
        df = df.rename(columns={well_col: 'well', value_col: 'value'})

        # Parse well positions
        df['row'] = df['well'].str[0]
        df['column'] = df['well'].str[1:].astype(int)

        return df


class AssayNormalizer:
    """Normalize assay data using various methods."""

    @staticmethod
    def percent_inhibition(
        values: np.ndarray,
        positive_control: float,
        negative_control: float
    ) -> np.ndarray:
        """
        Calculate percent inhibition.

        Args:
            values: Raw values
            positive_control: Positive control mean (100% inhibition)
            negative_control: Negative control mean (0% inhibition)

        Returns:
            Percent inhibition values
        """
        return 100 * (values - negative_control) / (positive_control - negative_control)

    @staticmethod
    def percent_activation(
        values: np.ndarray,
        positive_control: float,
        negative_control: float
    ) -> np.ndarray:
        """
        Calculate percent activation.

        Args:
            values: Raw values
            positive_control: Positive control mean (100% activation)
            negative_control: Negative control mean (0% activation)

        Returns:
            Percent activation values
        """
        return 100 * (values - negative_control) / (positive_control - negative_control)

    @staticmethod
    def z_score(values: np.ndarray) -> np.ndarray:
        """
        Calculate Z-scores.

        Args:
            values: Raw values

        Returns:
            Z-scores
        """
        return (values - np.mean(values)) / np.std(values)

    @staticmethod
    def robust_z_score(values: np.ndarray) -> np.ndarray:
        """
        Calculate robust Z-scores using median and MAD.

        Args:
            values: Raw values

        Returns:
            Robust Z-scores
        """
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        return 0.6745 * (values - median) / mad

    @staticmethod
    def b_score(
        plate_df: pd.DataFrame,
        value_col: str = "value"
    ) -> np.ndarray:
        """
        Calculate B-scores (corrects for row/column effects).

        Args:
            plate_df: DataFrame with 'row', 'column', and value columns
            value_col: Column name for values

        Returns:
            B-scores
        """
        values = plate_df[value_col].values
        rows = plate_df['row'].values
        columns = plate_df['column'].values

        # Calculate row and column medians
        row_medians = plate_df.groupby('row')[value_col].median()
        col_medians = plate_df.groupby('column')[value_col].median()
        overall_median = plate_df[value_col].median()

        # Calculate B-scores
        b_scores = []
        for val, row, col in zip(values, rows, columns):
            row_effect = row_medians[row] - overall_median
            col_effect = col_medians[col] - overall_median
            b_score = val - row_effect - col_effect
            b_scores.append(b_score)

        return np.array(b_scores)


class QualityController:
    """Perform quality control on assay data."""

    @staticmethod
    def calculate_z_factor(
        positive_values: np.ndarray,
        negative_values: np.ndarray
    ) -> float:
        """
        Calculate Z-factor.

        Z-factor = 1 - (3 * (σp + σn) / |μp - μn|)

        Args:
            positive_values: Positive control values
            negative_values: Negative control values

        Returns:
            Z-factor (1 = ideal, 0.5-1 = excellent, 0-0.5 = acceptable)
        """
        mean_pos = np.mean(positive_values)
        mean_neg = np.mean(negative_values)
        std_pos = np.std(positive_values)
        std_neg = np.std(negative_values)

        z_factor = 1 - (3 * (std_pos + std_neg) / abs(mean_pos - mean_neg))

        return z_factor

    @staticmethod
    def calculate_z_prime_factor(
        positive_values: np.ndarray,
        negative_values: np.ndarray
    ) -> float:
        """
        Calculate Z'-factor (same as Z-factor but for controls only).

        Args:
            positive_values: Positive control values
            negative_values: Negative control values

        Returns:
            Z'-factor
        """
        return QualityController.calculate_z_factor(positive_values, negative_values)

    @staticmethod
    def calculate_signal_to_noise(
        signal: np.ndarray,
        background: np.ndarray
    ) -> float:
        """
        Calculate signal-to-noise ratio.

        Args:
            signal: Signal values
            background: Background values

        Returns:
            Signal-to-noise ratio
        """
        mean_signal = np.mean(signal)
        std_background = np.std(background)

        return mean_signal / std_background

    @staticmethod
    def calculate_signal_to_background(
        signal: np.ndarray,
        background: np.ndarray
    ) -> float:
        """
        Calculate signal-to-background ratio.

        Args:
            signal: Signal values
            background: Background values

        Returns:
            Signal-to-background ratio
        """
        mean_signal = np.mean(signal)
        mean_background = np.mean(background)

        return mean_signal / mean_background

    @staticmethod
    def detect_outliers(
        values: np.ndarray,
        method: str = "iqr",
        threshold: float = 1.5
    ) -> np.ndarray:
        """
        Detect outliers.

        Args:
            values: Array of values
            method: Detection method ('iqr', 'zscore', 'mad')
            threshold: Threshold for outlier detection

        Returns:
            Boolean array (True = outlier)
        """
        if method == "iqr":
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            return (values < lower) | (values > upper)

        elif method == "zscore":
            z_scores = np.abs(stats.zscore(values))
            return z_scores > threshold

        elif method == "mad":
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            modified_z = 0.6745 * (values - median) / mad
            return np.abs(modified_z) > threshold

        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def assess_plate_quality(
        plate_df: pd.DataFrame,
        positive_wells: List[str],
        negative_wells: List[str],
        value_col: str = "value"
    ) -> PlateQCMetrics:
        """
        Assess overall plate quality.

        Args:
            plate_df: DataFrame with plate data
            positive_wells: List of positive control wells
            negative_wells: List of negative control wells
            value_col: Column name for values

        Returns:
            PlateQCMetrics object
        """
        # Extract control values
        pos_values = plate_df[plate_df['well'].isin(positive_wells)][value_col].values
        neg_values = plate_df[plate_df['well'].isin(negative_wells)][value_col].values

        # Calculate metrics
        z_factor = QualityController.calculate_z_factor(pos_values, neg_values)
        z_prime = QualityController.calculate_z_prime_factor(pos_values, neg_values)
        snr = QualityController.calculate_signal_to_noise(pos_values, neg_values)
        sbr = QualityController.calculate_signal_to_background(pos_values, neg_values)

        cv_pos = 100 * np.std(pos_values) / np.mean(pos_values)
        cv_neg = 100 * np.std(neg_values) / np.mean(neg_values)

        # Detect outliers in sample wells
        sample_wells = ~plate_df['well'].isin(positive_wells + negative_wells)
        sample_values = plate_df[sample_wells][value_col].values
        outliers = QualityController.detect_outliers(sample_values)
        n_outliers = np.sum(outliers)

        # Determine pass/fail
        passed = (
            z_prime >= 0.5 and  # Excellent assay
            cv_pos < 20 and  # Low variability
            cv_neg < 20
        )

        return PlateQCMetrics(
            z_factor=z_factor,
            z_prime_factor=z_prime,
            signal_to_noise=snr,
            signal_to_background=sbr,
            cv_positive=cv_pos,
            cv_negative=cv_neg,
            n_outliers=n_outliers,
            passed=passed
        )


class ReplicateHandler:
    """Handle replicate measurements."""

    @staticmethod
    def aggregate_replicates(
        df: pd.DataFrame,
        group_cols: List[str],
        value_col: str,
        method: str = "mean",
        remove_outliers: bool = True,
        outlier_threshold: float = 2.0
    ) -> pd.DataFrame:
        """
        Aggregate replicate measurements.

        Args:
            df: DataFrame with replicates
            group_cols: Columns to group by (e.g., ['compound_id'])
            value_col: Column with values to aggregate
            method: Aggregation method ('mean', 'median', 'trimmed_mean')
            remove_outliers: Remove outliers before aggregation
            outlier_threshold: Z-score threshold for outliers

        Returns:
            DataFrame with aggregated values
        """
        def aggregate_group(group):
            values = group[value_col].values

            if remove_outliers and len(values) > 2:
                # Remove outliers using Z-score
                z_scores = np.abs(stats.zscore(values))
                values = values[z_scores < outlier_threshold]

            if len(values) == 0:
                return pd.Series({
                    f'{value_col}_mean': np.nan,
                    f'{value_col}_std': np.nan,
                    f'{value_col}_sem': np.nan,
                    f'{value_col}_n': 0
                })

            if method == "mean":
                agg_value = np.mean(values)
            elif method == "median":
                agg_value = np.median(values)
            elif method == "trimmed_mean":
                agg_value = stats.trim_mean(values, 0.1)  # Trim 10% from each end
            else:
                raise ValueError(f"Unknown method: {method}")

            return pd.Series({
                f'{value_col}_mean': agg_value,
                f'{value_col}_std': np.std(values),
                f'{value_col}_sem': stats.sem(values),
                f'{value_col}_n': len(values)
            })

        result = df.groupby(group_cols).apply(aggregate_group).reset_index()

        return result


class HitCaller:
    """Call hits based on statistical thresholds."""

    @staticmethod
    def call_hits_by_threshold(
        values: np.ndarray,
        threshold: float,
        direction: str = "greater"
    ) -> np.ndarray:
        """
        Call hits by simple threshold.

        Args:
            values: Array of values
            threshold: Threshold value
            direction: 'greater' or 'less'

        Returns:
            Boolean array (True = hit)
        """
        if direction == "greater":
            return values > threshold
        elif direction == "less":
            return values < threshold
        else:
            raise ValueError(f"Unknown direction: {direction}")

    @staticmethod
    def call_hits_by_zscore(
        values: np.ndarray,
        threshold: float = 3.0,
        direction: str = "greater"
    ) -> np.ndarray:
        """
        Call hits by Z-score.

        Args:
            values: Array of values
            threshold: Z-score threshold
            direction: 'greater' or 'less'

        Returns:
            Boolean array (True = hit)
        """
        z_scores = stats.zscore(values)

        if direction == "greater":
            return z_scores > threshold
        elif direction == "less":
            return z_scores < -threshold
        else:
            raise ValueError(f"Unknown direction: {direction}")

    @staticmethod
    def call_hits_by_mad(
        values: np.ndarray,
        threshold: float = 3.0,
        direction: str = "greater"
    ) -> np.ndarray:
        """
        Call hits by MAD (Median Absolute Deviation).

        Args:
            values: Array of values
            threshold: MAD threshold
            direction: 'greater' or 'less'

        Returns:
            Boolean array (True = hit)
        """
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        mad_scores = (values - median) / (1.4826 * mad)  # 1.4826 makes MAD consistent with std

        if direction == "greater":
            return mad_scores > threshold
        elif direction == "less":
            return mad_scores < -threshold
        else:
            raise ValueError(f"Unknown direction: {direction}")

    @staticmethod
    def classify_activity(
        values: np.ndarray,
        labels: Optional[List[str]] = None,
        thresholds: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Classify activity into multiple categories.

        Args:
            values: Array of values
            labels: Category labels (default: ['inactive', 'weak', 'moderate', 'strong'])
            thresholds: Threshold values (default: [25, 50, 75] for percent inhibition)

        Returns:
            Array of category labels
        """
        if labels is None:
            labels = ['inactive', 'weak', 'moderate', 'strong']

        if thresholds is None:
            thresholds = [25, 50, 75]

        if len(labels) != len(thresholds) + 1:
            raise ValueError("Number of labels must be one more than number of thresholds")

        categories = np.empty(len(values), dtype=object)
        categories[:] = labels[0]

        for i, threshold in enumerate(thresholds):
            categories[values >= threshold] = labels[i + 1]

        return categories


def process_dose_response_plate(
    df: pd.DataFrame,
    compound_col: str = "compound_id",
    concentration_col: str = "concentration",
    value_col: str = "value",
    positive_wells: Optional[List[str]] = None,
    negative_wells: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Process dose-response plate data.

    Args:
        df: DataFrame with plate data
        compound_col: Column name for compound IDs
        concentration_col: Column name for concentrations
        value_col: Column name for values
        positive_wells: Positive control wells
        negative_wells: Negative control wells

    Returns:
        Processed DataFrame with normalized values
    """
    df = df.copy()

    # Calculate controls if specified
    if positive_wells and negative_wells:
        pos_mean = df[df['well'].isin(positive_wells)][value_col].mean()
        neg_mean = df[df['well'].isin(negative_wells)][value_col].mean()

        # Normalize to percent inhibition
        normalizer = AssayNormalizer()
        df['percent_inhibition'] = normalizer.percent_inhibition(
            df[value_col].values,
            pos_mean,
            neg_mean
        )

    # Calculate Z-scores
    normalizer = AssayNormalizer()
    df['z_score'] = normalizer.z_score(df[value_col].values)

    return df


if __name__ == "__main__":
    print("Assay Processing Utilities")
    print("=" * 50)

    # Example: Generate synthetic plate data
    np.random.seed(42)

    # 96-well plate
    rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    cols = list(range(1, 13))

    data = []
    for row in rows:
        for col in cols:
            well = f"{row}{col:02d}"

            # Positive controls (column 1)
            if col == 1:
                value = np.random.normal(10000, 500)
                well_type = "positive"
            # Negative controls (column 12)
            elif col == 12:
                value = np.random.normal(1000, 100)
                well_type = "negative"
            # Samples
            else:
                value = np.random.normal(5000, 1000)
                well_type = "sample"

            data.append({
                'well': well,
                'row': row,
                'column': col,
                'value': value,
                'well_type': well_type
            })

    plate_df = pd.DataFrame(data)

    print("\nPlate data:")
    print(plate_df.head(10))

    # Calculate QC metrics
    print("\n" + "=" * 50)
    print("Quality Control Metrics")

    positive_wells = [f"{row}01" for row in rows]
    negative_wells = [f"{row}12" for row in rows]

    qc_metrics = QualityController.assess_plate_quality(
        plate_df,
        positive_wells,
        negative_wells
    )

    print(f"\nZ'-factor: {qc_metrics.z_prime_factor:.3f}")
    print(f"Signal-to-Noise: {qc_metrics.signal_to_noise:.2f}")
    print(f"CV (Positive): {qc_metrics.cv_positive:.2f}%")
    print(f"CV (Negative): {qc_metrics.cv_negative:.2f}%")
    print(f"Outliers: {qc_metrics.n_outliers}")
    print(f"Passed: {qc_metrics.passed}")

    # Normalize data
    print("\n" + "=" * 50)
    print("Normalization")

    pos_mean = plate_df[plate_df['well'].isin(positive_wells)]['value'].mean()
    neg_mean = plate_df[plate_df['well'].isin(negative_wells)]['value'].mean()

    normalizer = AssayNormalizer()
    plate_df['percent_inhibition'] = normalizer.percent_inhibition(
        plate_df['value'].values,
        pos_mean,
        neg_mean
    )

    print("\nNormalized data:")
    print(plate_df[plate_df['well_type'] == 'sample'][['well', 'value', 'percent_inhibition']].head())

    # Call hits
    print("\n" + "=" * 50)
    print("Hit Calling")

    sample_data = plate_df[plate_df['well_type'] == 'sample']
    hits = HitCaller.call_hits_by_threshold(
        sample_data['percent_inhibition'].values,
        threshold=50,
        direction="greater"
    )

    print(f"\nHits identified: {np.sum(hits)}/{len(hits)}")
