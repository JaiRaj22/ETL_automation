import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import f1_score, make_scorer
import io
import base64
import datetime
import re

def main():
    st.title("Enhanced Automated Data Analysis System")
    st.write("Upload a CSV file to automatically clean, analyze, and visualize your data.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read data
        df = read_data(uploaded_file)
        
        if df is not None:
            # Display raw data
            st.subheader("Raw Data Preview")
            st.write(df.head())
            
            # Data cleaning and processing
            st.subheader("Data Cleaning and Processing")
            
            # Display basic info
            show_basic_info(df)
            
            # Ask for target variable first
            
            
            # Remove ID columns
            df_no_id = identify_and_remove_id_columns(df)
            
            # Detect and convert datetime columns
            df_no_id = detect_and_convert_datetime(df_no_id)
            
            # Clean data
            df_cleaned = clean_data(df_no_id)
            
            # Encode categorical variables
            df_encoded = encode_categorical(df_cleaned)
            
            all_columns = df.columns.tolist()
            target_col = st.selectbox("Select target variable for analysis:", all_columns)
            
            # Remove columns with low correlation to target
            if target_col in df_encoded.columns:
                df_encoded = remove_low_correlation_features(df_encoded, target_col)
            
            # Handle outliers
            with st.expander("Outlier Handling Options"):
                outlier_method = st.radio(
                    "Select method for handling outliers:",
                    ("Cap (Winsorize)", "Remove")
                )
            
            df_no_outliers = handle_outliers(df_encoded, method=outlier_method)
            
            # Display processed data
            st.subheader("Processed Data Preview")
            st.write(df_no_outliers.head())
            
            # Download processed data
            st.download_button(
                label="Download processed data as CSV",
                data=convert_df_to_csv(df_no_outliers),
                file_name='processed_data.csv',
                mime='text/csv',
            )
            
            # Data analysis and visualization
            st.subheader("Data Analysis and Visualization")
            
            # Correlation analysis
            correlation_analysis(df_no_outliers)
            
            # PCA analysis
            perform_pca_analysis(df_no_outliers)
            
            # Generate relevant plots
            generate_plots(df_no_outliers)
            df_ml_ready = df_no_outliers.copy()
            # ML algorithm suggestions
            suggest_ml_algorithms(df_ml_ready, target_col)

def read_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def show_basic_info(df):
    # Display shape
    st.write(f"Dataset shape: {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Display data types
    st.write("Data types:")
    st.write(df.dtypes)
    
    # Display missing values
    missing_values = df.isnull().sum()
    st.write("Missing values:")
    st.write(missing_values)

def identify_and_remove_id_columns(df):
    st.write("Identifying and removing ID columns...")
    
    # Make a copy of the dataframe
    df_no_id = df.copy()
    
    # Common ID column names (case-insensitive)
    id_patterns = [
        r'^id$', r'^_?id_?$', r'.*_id$', r'^id_.*$', r'.*_id_.*$',
        r'^key$', r'^row_?num.*$', r'^index$', r'^record_?num.*$',
        r'.*identifier$', r'^uuid$', r'^guid$'
    ]
    
    # Columns with unique values can also indicate they are IDs
    unique_cols = []
    
    # Check if there's any column that closely resembles an ID column
    id_cols_removed = []
    
    for col in df.columns:
        # Check if column name matches ID patterns
        is_id_by_name = any(re.match(pattern, col, re.IGNORECASE) for pattern in id_patterns)
        
        # Check if column has mostly unique values (>95% unique)
        unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
        is_unique = unique_ratio > 0.95 and len(df) > 10
        
        # Check if column is integer or string with consistent length
        is_potential_id = False
        if pd.api.types.is_integer_dtype(df[col]):
            is_potential_id = True
        elif pd.api.types.is_string_dtype(df[col]):
            # Check for consistent length in string IDs (take a sample for efficiency)
            sample = df[col].dropna().sample(min(100, len(df[col].dropna()))).astype(str)
            if len(sample) > 0 and sample.str.len().var() < 1:  # Low variance in length
                is_potential_id = True
        
        # If it looks like an ID column, add to removal list
        if (is_id_by_name or (is_unique and is_potential_id)):
            id_cols_removed.append((col, f"{'Name match' if is_id_by_name else ''} {'Unique values' if is_unique else ''} {unique_ratio:.2%} unique"))
            unique_cols.append(col)
    
    # Display ID columns identified
    if id_cols_removed:
        st.write("The following ID-like columns were identified:")
        for col, reason in id_cols_removed:
            st.write(f"- '{col}': {reason}")
        
        # Let user confirm which columns to remove
        columns_to_remove = st.multiselect(
            "Select ID columns to remove:",
            options=unique_cols,
            default=unique_cols
        )
        
        # Remove selected ID columns
        if columns_to_remove:
            df_no_id = df_no_id.drop(columns=columns_to_remove)
            st.write(f"Removed {len(columns_to_remove)} ID columns.")
    else:
        st.write("No ID-like columns were identified.")
    
    return df_no_id

def remove_low_correlation_features(df, target_col):
    st.write(f"Analyzing correlation with target variable '{target_col}'...")
    
    # Make a copy
    df_filtered = df.copy()
    
    # Skip if target variable is no longer in dataframe
    if target_col not in df.columns:
        st.warning(f"Target variable '{target_col}' not found in dataframe. Skipping correlation analysis.")
        return df_filtered
    
    try:
        # Calculate correlation with target for numerical features
        target_is_numeric = pd.api.types.is_numeric_dtype(df[target_col])
        features_to_analyze = []
        correlation_dict = {}
        
        if target_is_numeric:
            # For numeric target, get correlation with other numeric features
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != target_col]
            
            if numeric_cols:
                correlations = df[numeric_cols + [target_col]].corr()[target_col].abs().sort_values(ascending=True)
                # Remove self-correlation
                correlations = correlations.drop(target_col, errors='ignore')
                
                # Convert to dictionary for display
                correlation_dict = correlations.to_dict()
                features_to_analyze = numeric_cols
        else:
            # For categorical target, we need a different approach 
            # Use One-Hot Encoding for target
            encoded_target = pd.get_dummies(df[target_col])
            
            # Calculate correlations for numeric features with each target class
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                # For each numeric feature, get max correlation with any target class
                max_correlations = {}
                for col in numeric_cols:
                    if col != target_col:
                        col_correlations = []
                        for target_class in encoded_target.columns:
                            corr = df[col].corr(encoded_target[target_class], method='pearson')
                            col_correlations.append(abs(corr))
                        max_correlations[col] = max(col_correlations) if col_correlations else 0
                
                # Sort by correlation (ascending)
                sorted_correlations = {k: v for k, v in sorted(max_correlations.items(), key=lambda item: item[1])}
                correlation_dict = sorted_correlations
                features_to_analyze = list(sorted_correlations.keys())
        
        # Display correlations
        if correlation_dict:
            st.write("Correlation with target variable (absolute values):")
            for feature, corr in correlation_dict.items():
                st.write(f"- {feature}: {corr:.4f}")
            
            # Let user select correlation threshold
            if features_to_analyze:
                corr_threshold = st.slider(
                    "Select correlation threshold for feature removal (lower = remove more features):",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.05,
                    step=0.01
                )
                
                # Identify features below threshold
                low_corr_features = [feat for feat, corr in correlation_dict.items() if corr < corr_threshold]
                
                # Let user confirm which low-correlated features to remove
                if low_corr_features:
                    st.write(f"Found {len(low_corr_features)} features with correlation below {corr_threshold}:")
                    for feat in low_corr_features:
                        st.write(f"- {feat}: {correlation_dict[feat]:.4f}")
                    
                    features_to_remove = st.multiselect(
                        "Select low-correlation features to remove:",
                        options=low_corr_features,
                        default=low_corr_features
                    )
                    
                    # Remove selected features
                    if features_to_remove:
                        df_filtered = df_filtered.drop(columns=features_to_remove)
                        st.write(f"Removed {len(features_to_remove)} low-correlation features.")
                else:
                    st.write(f"No features found with correlation below {corr_threshold}.")
        else:
            st.write("No features available for correlation analysis with target.")
        
    except Exception as e:
        st.error(f"Error in correlation analysis: {e}")
        st.write("Skipping feature removal based on correlation.")
    
    return df_filtered

def detect_and_convert_datetime(df):
    st.write("Detecting and converting datetime columns...")
    
    df_copy = df.copy()
    datetime_patterns = [
        # Common date formats
        r'\d{4}-\d{1,2}-\d{1,2}',                      # YYYY-MM-DD
        r'\d{1,2}-\d{1,2}-\d{4}',                      # DD-MM-YYYY or MM-DD-YYYY
        r'\d{1,2}/\d{1,2}/\d{4}',                      # DD/MM/YYYY or MM/DD/YYYY
        r'\d{4}/\d{1,2}/\d{1,2}',                      # YYYY/MM/DD
        r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}',  # DD Mon YYYY
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}', # Mon DD, YYYY
        
        # Date formats with time
        r'\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{1,2}(:\d{1,2})?',  # YYYY-MM-DD HH:MM(:SS)
        r'\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{1,2}(:\d{1,2})?',  # MM/DD/YYYY HH:MM(:SS)
    ]
    
    datetime_cols = []
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if column contains datetime strings
            sample_values = df[col].dropna().sample(min(5, len(df[col].dropna()))).tolist()
            
            is_datetime = False
            for sample in sample_values:
                if isinstance(sample, str):
                    for pattern in datetime_patterns:
                        if re.search(pattern, sample):
                            is_datetime = True
                            break
                if is_datetime:
                    break
            
            if is_datetime:
                try:
                    df_copy[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
                    if not df_copy[col].isna().all():  # Only keep conversion if not all values became NaN
                        datetime_cols.append(col)
                        st.write(f"Converted '{col}' to datetime format")
                except Exception as e:
                    st.write(f"Failed to convert '{col}' to datetime: {e}")
    
    return df_copy

def clean_data(df):
    st.write("Cleaning data...")
    
    # Make a copy
    df_cleaned = df.copy()
    
    # Handle missing values
    missing_values = df.isnull().sum()
    columns_with_missing = missing_values[missing_values > 0].index.tolist()
    
    if columns_with_missing:
        st.write("Handling missing values...")
        
        for col in columns_with_missing:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            
            if missing_pct > 50:
                # Drop columns with more than 50% missing values
                st.write(f"Dropping column '{col}' with {missing_pct:.2f}% missing values")
                df_cleaned = df_cleaned.drop(col, axis=1)
            else:
                # Fill missing values based on data type
                if pd.api.types.is_datetime64_dtype(df[col]):
                    # For datetime, fill with median date
                    median_date = df[col].median()
                    df_cleaned[col] = df_cleaned[col].fillna(median_date)
                    st.write(f"Filled missing values in '{col}' with median date: {median_date}")
                elif df[col].dtype == 'object':
                    # For categorical columns, fill with mode
                    mode_value = df[col].mode()[0]
                    df_cleaned[col] = df_cleaned[col].fillna(mode_value)
                    st.write(f"Filled missing values in '{col}' with mode: {mode_value}")
                else:
                    # For numerical columns, fill with median
                    median_value = df[col].median()
                    df_cleaned[col] = df_cleaned[col].fillna(median_value)
                    st.write(f"Filled missing values in '{col}' with median: {median_value}")
    
    return df_cleaned

def encode_categorical(df):
    st.write("Encoding categorical variables...")
    
    # Make a copy
    df_encoded = df.copy()
    
    # Find categorical columns with <= 10 unique values
    for col in df.columns:
        if df[col].dtype == 'object' or (df[col].dtype.name == 'category'):
            unique_values = df[col].nunique()
            
            if unique_values <= 10:
                st.write(f"Encoding column '{col}' with {unique_values} unique values")
                
                # Use Label Encoder for encoding
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
                
                # Display mapping
                mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                st.write(f"Encoding mapping for '{col}': {mapping}")
    
    return df_encoded

def handle_outliers(df, method="Cap (Winsorize)"):
    st.write(f"Detecting and handling outliers using {method} method...")
    
    # Make a copy
    df_processed = df.copy()
    
    # Find numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create a placeholder for outlier plots
    if numerical_cols:
        outlier_fig, outlier_axes = plt.subplots(len(numerical_cols), 1, figsize=(10, 3*len(numerical_cols)))
        
        # Handle case when there's only one numerical column
        if len(numerical_cols) == 1:
            outlier_axes = [outlier_axes]
    
    # Process each numerical column
    for i, col in enumerate(numerical_cols):
        # Calculate IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Get outlier mask
        outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outliers_count = outliers_mask.sum()
        
        if outliers_count > 0:
            st.write(f"Found {outliers_count} outliers in '{col}'")
            
            # Create boxplot
            if numerical_cols:
                sns.boxplot(x=df[col], ax=outlier_axes[i])
                outlier_axes[i].set_title(f"Boxplot of {col}")
                outlier_axes[i].set_xlabel(col)
            
            if method == "Cap (Winsorize)":
                # Cap outliers to boundaries
                df_processed.loc[df_processed[col] < lower_bound, col] = lower_bound
                df_processed.loc[df_processed[col] > upper_bound, col] = upper_bound
                st.write(f"Capped outliers in '{col}' to range: [{lower_bound:.2f}, {upper_bound:.2f}]")
            elif method == "Remove":
                # Create temporary mask to preserve indexes for display
                temp_mask = outliers_mask.copy()
                # Display info about removed outliers
                st.write(f"Removing {outliers_count} outliers from '{col}'")
                # Remove outliers
                df_processed = df_processed[~outliers_mask]
                # Reset index after removal
                df_processed = df_processed.reset_index(drop=True)
    
    # Display outlier plots
    if numerical_cols:
        plt.tight_layout()
        st.pyplot(outlier_fig)
    
    return df_processed

def correlation_analysis(df):
    st.write("Performing correlation analysis...")
    
    # Find numerical columns for correlation
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) >= 2:
        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr()
        
        # Display correlation matrix
        st.write("Correlation Matrix:")
        st.write(corr_matrix)
        
        # Plot heatmap
        st.write("Correlation Heatmap:")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)
        
        # Identify highly correlated features
        st.write("Highly correlated features:")
        
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) >= 0.7:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        if high_corr_pairs:
            for col1, col2, corr in high_corr_pairs:
                st.write(f"{col1} and {col2}: {corr:.2f}")
                
                # Create scatter plot for highly correlated features
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(df[col1], df[col2], alpha=0.5)
                ax.set_xlabel(col1)
                ax.set_ylabel(col2)
                ax.set_title(f"Scatter plot: {col1} vs {col2} (correlation: {corr:.2f})")
                st.pyplot(fig)
        else:
            st.write("No highly correlated features found (correlation >= 0.7)")
    else:
        st.write("Need at least two numerical columns for correlation analysis.")

def perform_pca_analysis(df):
    st.write("Performing PCA analysis...")
    
    # Find numerical columns for PCA
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) >= 2:
        # Create a copy of numerical data
        numerical_data = df[numerical_cols].copy()
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numerical_data)
        
        # Determine number of components based on data dimensions
        n_components = min(len(numerical_cols), 10)  # Max 10 components
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)
        
        # Calculate explained variance
        explained_variance = pca.explained_variance_ratio_ * 100
        cumulative_variance = np.cumsum(explained_variance)
        
        # Plot explained variance
        st.write("PCA Explained Variance:")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(1, n_components + 1), explained_variance, alpha=0.7, label='Individual')
        ax.step(range(1, n_components + 1), cumulative_variance, where='mid', label='Cumulative', color='red')
        ax.set_xlabel('Principal Components')
        ax.set_ylabel('Explained Variance (%)')
        ax.set_title('Explained Variance by Principal Components')
        ax.axhline(y=80, color='gray', linestyle='--', label='80% Threshold')
        ax.legend()
        st.pyplot(fig)
        
        # Determine optimal number of components (80% variance)
        optimal_components = np.where(cumulative_variance >= 80)[0]
        if len(optimal_components) > 0:
            optimal_n = optimal_components[0] + 1
            st.write(f"Optimal number of components for 80% variance: {optimal_n}")
        else:
            optimal_n = n_components
            st.write(f"All {n_components} components explain {cumulative_variance[-1]:.2f}% of variance")
        
        # Create dataframe with PCA results
        pca_df = pd.DataFrame(
            data=pca_result[:, :2],
            columns=['PC1', 'PC2']
        )
        
        # Plot first two principal components
        st.write("First Two Principal Components:")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)
        ax.set_xlabel(f'PC1 ({explained_variance[0]:.2f}%)')
        ax.set_ylabel(f'PC2 ({explained_variance[1]:.2f}%)')
        ax.set_title('PCA: First two principal components')
        plt.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)
        
        # Display component loadings
        st.write("PCA Component Loadings:")
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=numerical_cols
        )
        st.write(loadings)
        
        # Loadings heatmap
        st.write("Loadings Heatmap:")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(loadings.iloc[:, :min(5, n_components)], annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('PCA Loadings Heatmap (First 5 Components)')
        st.pyplot(fig)
    else:
        st.write("Need at least two numerical columns for PCA analysis.")

def generate_plots(df):
    st.write("Generating relevant plots based on data types...")
    
    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Add encoded columns with <= 10 unique values as categorical
    for col in df.columns:
        if col not in numerical_cols and col not in categorical_cols and col not in datetime_cols:
            if df[col].nunique() <= 10:
                categorical_cols.append(col)
    
    # Generate distributions for numerical variables
    if numerical_cols:
        st.write("Distribution of numerical variables:")
        
        # Create histograms
        for i in range(0, len(numerical_cols), 2):
            cols_subset = numerical_cols[i:i+2]
            fig, axes = plt.subplots(1, len(cols_subset), figsize=(12, 5))
            
            # Handle case when there's only one column left
            if len(cols_subset) == 1:
                axes = [axes]
            
            for j, col in enumerate(cols_subset):
                sns.histplot(df[col], kde=True, ax=axes[j])
                axes[j].set_title(f"Distribution of {col}")
                axes[j].set_xlabel(col)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # Generate bar charts for categorical variables
    if categorical_cols:
        st.write("Distribution of categorical variables:")
        
        for col in categorical_cols:
            if df[col].nunique() <= 10:  # Only plot if 10 or fewer categories
                fig, ax = plt.subplots(figsize=(10, 6))
                df[col].value_counts().plot(kind='bar', ax=ax)
                ax.set_title(f"Distribution of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
    
    # Generate time series plots for datetime columns
    if datetime_cols:
        st.write("Time series analysis:")
        
        for date_col in datetime_cols:
            # Check if we have numerical columns to plot against time
            if numerical_cols:
                for num_col in numerical_cols[:3]:  # Limit to first 3 numerical columns
                    # Create time series plot
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Sort by date
                    temp_df = df.sort_values(by=date_col)
                    
                    # Plot time series
                    ax.plot(temp_df[date_col], temp_df[num_col])
                    ax.set_title(f"{num_col} over time ({date_col})")
                    ax.set_xlabel(date_col)
                    ax.set_ylabel(num_col)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
    
    # Generate pairplot for numerical variables (limited to prevent memory issues)
    if len(numerical_cols) >= 2 and len(numerical_cols) <= 5:
        st.write("Pairplot of numerical variables:")
        fig = sns.pairplot(df[numerical_cols])
        st.pyplot(fig)
    elif len(numerical_cols) > 5:
        st.write("Pairplot of selected numerical variables (limited to 5):")
        # Select top 5 columns with highest variance
        var_cols = df[numerical_cols].var().sort_values(ascending=False).index[:5].tolist()
        fig = sns.pairplot(df[var_cols])
        st.pyplot(fig)

def convert_datetime_features(df):
    """Convert datetime columns to useful numeric features for ML algorithms"""
    df_processed = df.copy()
    
    # Find datetime columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    for col in datetime_cols:
        # Extract useful components from datetime
        df_processed[f"{col}_year"] = df[col].dt.year
        df_processed[f"{col}_month"] = df[col].dt.month
        df_processed[f"{col}_day"] = df[col].dt.day
        df_processed[f"{col}_dayofweek"] = df[col].dt.dayofweek
        
        # Drop the original datetime column
        df_processed = df_processed.drop(columns=[col])
        
    return df_processed

def suggest_ml_algorithms(df, target_col=None):
    st.subheader("Machine Learning Algorithm Suggestions")
    df = convert_datetime_features(df)
    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Add categorical columns with <= 10 unique values
    for col in df.columns:
        if col not in numerical_cols and col not in categorical_cols:
            if df[col].nunique() <= 10:
                categorical_cols.append(col)
    
    if len(df.columns) <= 1:
        st.write("Not enough columns for ML algorithm suggestions.")
        return
    
    # Use passed target_col if available, otherwise let user select
    if target_col is None or target_col not in df.columns:
        all_columns = df.columns.tolist()
        target_col = st.selectbox("Select target variable for ML suggestions:", all_columns)
    else:
        st.write(f"Using '{target_col}' as target variable for ML suggestions.")
    
    if target_col:
        # Remove target from feature list
        features = [col for col in df.columns if col != target_col]
        
        # Check target variable type
        is_classification = False
        n_classes = df[target_col].nunique()
        
        if target_col in categorical_cols or n_classes <= 10:
            task_type = "Classification"
            is_classification = True
            st.write(f"Detected task type: Classification with {n_classes} classes")
        else:
            task_type = "Regression"
            st.write("Detected task type: Regression")
        
        # Split data for testing
        X = df[features]
        y = df[target_col]
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Suggest and evaluate algorithms
            st.write("Evaluating ML algorithms for best F1 score...")
            
            # Define algorithms based on task type
            if is_classification:
                algorithms = {
                    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                    "SVM": SVC(probability=True, random_state=42),
                    "KNN": KNeighborsClassifier()
                }
                
                # For binary classification, use binary F1
                if n_classes == 2:
                    scorer = make_scorer(f1_score, average='binary')
                else:
                    scorer = make_scorer(f1_score, average='weighted')
                
            else:  # Regression
                algorithms = {
                    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                    "Linear Regression": LinearRegression(),
                    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                    "SVR": SVR(),
                    "KNN": KNeighborsRegressor()
                }
                # For regression, use negative mean squared error instead of F1
                from sklearn.metrics import mean_squared_error
                scorer = make_scorer(mean_squared_error, greater_is_better=False)
            
            # Evaluate algorithms
            results = {}
            
            for name, algorithm in algorithms.items():
                try:
                    # Use cross-validation for more robust evaluation
                    scores = cross_val_score(algorithm, X_train, y_train, cv=5, scoring=scorer)
                    
                    if is_classification:
                        avg_score = np.mean(scores)
                        results[name] = avg_score
                        st.write(f"{name}: Average F1 Score = {avg_score:.4f}")
                    else:
                        # For regression, converting negative MSE to positive RMSE for interpretability
                        avg_rmse = np.sqrt(-np.mean(scores))
                        results[name] = -avg_rmse  # Keep negative for sorting
                        st.write(f"{name}: Average RMSE = {avg_rmse:.4f}")
                except Exception as e:
                    st.write(f"Error evaluating {name}: {e}")
            
            # Find best algorithm
            if results:
                best_algo = max(results.items(), key=lambda x: x[1])[0]
                
                st.write(f"\n**Recommended Algorithm**: {best_algo}")
                
                if is_classification:
                    st.write(f"This algorithm achieved the highest F1 score on your dataset.")
                    
                    # Additional recommendations
                    if n_classes > 2:
                        st.write("For multi-class classification:")
                        st.write("- Consider using class weights if classes are imbalanced")
                        st.write("- Feature engineering might help improve performance")
                    else:
                        st.write("For binary classification:")
                        st.write("- Consider using ROC AUC as an additional metric")
                        st.write("- Threshold tuning might improve precision/recall balance")
                else:
                    st.write(f"This algorithm achieved the lowest Root Mean Squared Error (RMSE) on your dataset.")
                    st.write("For regression tasks:")
                    st.write("- Feature scaling might improve performance for algorithms like SVR")
                    st.write("- Consider feature engineering or polynomial features for complex relationships")
        
        except Exception as e:
            st.error(f"Error in ML algorithm evaluation: {e}")
            st.write("Unable to evaluate ML algorithms. Please ensure your data is properly formatted and contains sufficient samples.")

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

if __name__ == "__main__":
    main()