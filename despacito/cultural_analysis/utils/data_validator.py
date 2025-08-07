"""
Data Validation Module for Cultural Analysis Project
Comprehensive data quality checks and validation procedures
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Comprehensive data validation for cultural analysis datasets"""
    
    def __init__(self):
        self.validation_results = {}
        
    def validate_dataset(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Run comprehensive validation on dataset
        
        Args:
            df: DataFrame to validate
            dataset_name: Name for reporting
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'dataset_name': dataset_name,
            'total_rows': len(df),
            'issues': [],
            'warnings': [],
            'quality_score': 0
        }
        
        # Basic structure checks
        results.update(self._check_basic_structure(df))
        
        # Comment quality checks
        if 'original_comment' in df.columns:
            results.update(self._check_comment_quality(df))
        
        # Timestamp validation
        if 'published_at' in df.columns:
            results.update(self._check_timestamps(df))
        
        # Spam/emotion label checks
        results.update(self._check_labels(df))
        
        # Calculate overall quality score
        results['quality_score'] = self._calculate_quality_score(results)
        
        self.validation_results[dataset_name] = results
        return results
    
    def _check_basic_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check basic DataFrame structure"""
        issues = []
        
        # Check for completely empty DataFrame
        if df.empty:
            issues.append("Dataset is completely empty")
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate rows")
        
        # Check for columns with all missing values
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            issues.append(f"Columns with all missing values: {empty_cols}")
        
        return {
            'duplicates': duplicates,
            'empty_columns': empty_cols,
            'basic_issues': issues
        }
    
    def _check_comment_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate comment text quality"""
        issues = []
        warnings = []
        
        comment_col = 'original_comment'
        
        # Missing comments
        missing_comments = df[comment_col].isnull().sum()
        if missing_comments > 0:
            issues.append(f"{missing_comments} missing comments")
        
        # Empty or very short comments
        valid_comments = df[comment_col].dropna()
        short_comments = (valid_comments.str.len() < 3).sum()
        if short_comments > 0:
            warnings.append(f"{short_comments} very short comments (< 3 characters)")
        
        # Extremely long comments (potential spam)
        long_comments = (valid_comments.str.len() > 1000).sum()
        if long_comments > 0:
            warnings.append(f"{long_comments} very long comments (> 1000 characters)")
        
        # Comments with excessive repeated characters
        repeated_pattern = r'(.)\1{5,}'
        excessive_repeats = valid_comments.str.contains(repeated_pattern, na=False).sum()
        if excessive_repeats > 0:
            warnings.append(f"{excessive_repeats} comments with excessive repeated characters")
        
        # Comments that are mostly numbers/symbols
        mostly_nonalpha = valid_comments.apply(
            lambda x: len(re.sub(r'[a-zA-Z\s]', '', str(x))) / len(str(x)) > 0.7 if x else False
        ).sum()
        if mostly_nonalpha > 0:
            warnings.append(f"{mostly_nonalpha} comments mostly non-alphabetic")
        
        return {
            'comment_issues': issues,
            'comment_warnings': warnings,
            'missing_comments': missing_comments,
            'short_comments': short_comments,
            'long_comments': long_comments
        }
    
    def _check_timestamps(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate timestamp data"""
        issues = []
        warnings = []
        
        timestamp_col = 'published_at'
        
        # Convert to datetime and check for parsing errors
        timestamps = pd.to_datetime(df[timestamp_col], errors='coerce')
        parsing_errors = timestamps.isnull().sum() - df[timestamp_col].isnull().sum()
        
        if parsing_errors > 0:
            issues.append(f"{parsing_errors} timestamps could not be parsed")
        
        # Check for future dates
        now = pd.Timestamp.now(tz='UTC')
        # Convert timestamps to UTC if they have timezone info
        if timestamps.dt.tz is not None:
            future_dates = (timestamps > now).sum()
        else:
            future_dates = (timestamps > now.tz_localize(None)).sum()
        if future_dates > 0:
            warnings.append(f"{future_dates} timestamps are in the future")
        
        # Check for very old dates (before YouTube existed)
        youtube_start = pd.Timestamp('2005-01-01')
        if timestamps.dt.tz is not None:
            youtube_start = youtube_start.tz_localize('UTC')
            too_old = (timestamps < youtube_start).sum()
        else:
            too_old = (timestamps < youtube_start).sum()
        if too_old > 0:
            warnings.append(f"{too_old} timestamps before YouTube existed")
        
        # Check temporal distribution
        valid_timestamps = timestamps.dropna()
        if len(valid_timestamps) > 0:
            date_range = valid_timestamps.max() - valid_timestamps.min()
            if date_range.days < 30:
                warnings.append("All comments span less than 30 days")
        
        return {
            'timestamp_issues': issues,
            'timestamp_warnings': warnings,
            'parsing_errors': parsing_errors,
            'future_dates': future_dates,
            'too_old_dates': too_old
        }
    
    def _check_labels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check spam and emotion labels"""
        issues = []
        warnings = []
        
        # Check spam labels
        if 'spam_label' in df.columns:
            spam_values = df['spam_label'].value_counts()
            if 'unknown' in spam_values and spam_values['unknown'] > len(df) * 0.8:
                warnings.append("More than 80% of spam labels are 'unknown'")
        
        # Check emotion labels
        if 'emotion_label' in df.columns:
            emotion_values = df['emotion_label'].value_counts()
            if 'unknown' in emotion_values and emotion_values['unknown'] > len(df) * 0.8:
                warnings.append("More than 80% of emotion labels are 'unknown'")
        
        # Check detected language
        if 'detected_language' in df.columns:
            lang_values = df['detected_language'].value_counts()
            if len(lang_values) == 1:
                warnings.append("All comments have the same detected language")
        
        return {
            'label_issues': issues,
            'label_warnings': warnings
        }
    
    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-100)"""
        score = 100.0
        
        # Deduct for issues
        score -= len(results.get('basic_issues', [])) * 10
        score -= len(results.get('comment_issues', [])) * 15
        score -= len(results.get('timestamp_issues', [])) * 10
        score -= len(results.get('label_issues', [])) * 5
        
        # Deduct for warnings (less severe)
        score -= len(results.get('comment_warnings', [])) * 3
        score -= len(results.get('timestamp_warnings', [])) * 2
        score -= len(results.get('label_warnings', [])) * 2
        
        # Ensure score doesn't go below 0
        return max(0.0, score)
    
    def generate_validation_report(self, output_path: Path = None) -> str:
        """Generate comprehensive validation report"""
        if not self.validation_results:
            return "No validation results available"
        
        report_lines = ["# Data Validation Report\n"]
        
        for dataset_name, results in self.validation_results.items():
            report_lines.append(f"## {dataset_name}")
            report_lines.append(f"- **Total Rows**: {results['total_rows']:,}")
            report_lines.append(f"- **Quality Score**: {results['quality_score']:.1f}/100")
            
            # Issues
            all_issues = (results.get('basic_issues', []) + 
                         results.get('comment_issues', []) + 
                         results.get('timestamp_issues', []) + 
                         results.get('label_issues', []))
            
            if all_issues:
                report_lines.append("\n### Issues:")
                for issue in all_issues:
                    report_lines.append(f"- ❌ {issue}")
            
            # Warnings
            all_warnings = (results.get('comment_warnings', []) + 
                           results.get('timestamp_warnings', []) + 
                           results.get('label_warnings', []))
            
            if all_warnings:
                report_lines.append("\n### Warnings:")
                for warning in all_warnings:
                    report_lines.append(f"- ⚠️ {warning}")
            
            report_lines.append("\n")
        
        report = "\n".join(report_lines)
        
        if output_path:
            output_path.write_text(report)
            logger.info(f"Validation report saved to {output_path}")
        
        return report
    
    def get_recommendations(self) -> List[str]:
        """Get data quality improvement recommendations"""
        recommendations = []
        
        for dataset_name, results in self.validation_results.items():
            if results['quality_score'] < 80:
                recommendations.append(f"Dataset '{dataset_name}' needs quality improvements")
            
            if results.get('duplicates', 0) > 0:
                recommendations.append(f"Remove duplicate rows from '{dataset_name}'")
            
            if results.get('missing_comments', 0) > 0:
                recommendations.append(f"Handle missing comments in '{dataset_name}'")
            
            if results.get('parsing_errors', 0) > 0:
                recommendations.append(f"Fix timestamp parsing errors in '{dataset_name}'")
        
        return recommendations

def validate_all_datasets(data_dir: Path) -> DataValidator:
    """Validate all datasets in the project"""
    validator = DataValidator()
    
    # Main input files
    input_files = [
        data_dir / "despa_kJQP7kiw5Fk_comments_spam_emotion.csv",
        data_dir / "seeyou_RgKAFK5djSk_comments_spam_emotion.csv"
    ]
    
    for file_path in input_files:
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                validator.validate_dataset(df, file_path.stem)
            except Exception as e:
                logger.error(f"Error validating {file_path}: {e}")
    
    return validator 