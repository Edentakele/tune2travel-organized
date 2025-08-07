#!/usr/bin/env python3
"""
Enhanced Task 1: Geographic Analysis with Advanced NLP
Uses the new modular architecture with improved location extraction and validation
"""

import pandas as pd
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# Import our new modules
from config import INPUT_FILES, TASK1_CONFIG, get_file_prefix, get_output_path
from utils import (
    load_and_validate_data, safe_save_csv, create_regex_pattern, 
    extract_matching_comments, analyze_temporal_distribution, 
    log_performance, create_progress_tracker
)
from data_validator import DataValidator

logger = logging.getLogger(__name__)

class EnhancedGeographicAnalyzer:
    """Enhanced geographic analysis with advanced NLP and validation"""
    
    def __init__(self):
        self.validator = DataValidator()
        self.results = {}
        
    @log_performance
    def analyze_single_dataset(self, file_path: Path, prefix: str) -> Dict:
        """
        Analyze a single dataset with enhanced geographic extraction
        
        Args:
            file_path: Path to input CSV file
            prefix: File prefix (despa/seeyou)
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Starting enhanced analysis for {prefix}")
        
        # Load and validate data
        required_cols = ['original_comment', 'published_at', 'spam_label', 'emotion_label']
        df = load_and_validate_data(file_path, required_cols)
        
        if df.empty:
            logger.error(f"No valid data loaded from {file_path}")
            return {}
        
        # Validate data quality
        validation_results = self.validator.validate_dataset(df, prefix)
        logger.info(f"Data quality score for {prefix}: {validation_results['quality_score']:.1f}/100")
        
        # Progress tracking
        total_comments = len(df)
        progress = create_progress_tracker(total_comments, f"Processing {prefix}")
        
        # Enhanced geographic extraction
        results = {
            'total_comments': total_comments,
            'quality_score': validation_results['quality_score'],
            'extracted_data': {}
        }
        
        # 1. Enhanced keyword extraction
        keyword_comments = self._extract_keyword_comments(df)
        results['extracted_data']['keywords'] = len(keyword_comments)
        progress(len(keyword_comments))
        
        # 2. Enhanced flag emoji extraction  
        flag_comments = self._extract_flag_emoji_comments(df)
        results['extracted_data']['flags'] = len(flag_comments)
        progress(len(keyword_comments) + len(flag_comments))
        
        # 3. Advanced location extraction
        location_comments = self._extract_location_comments_advanced(df)
        results['extracted_data']['locations'] = len(location_comments)
        progress(len(keyword_comments) + len(flag_comments) + len(location_comments))
        
        # 4. Combine all extracted comments
        all_extracted = self._combine_extracted_comments(
            keyword_comments, flag_comments, location_comments
        )
        results['extracted_data']['total_unique'] = len(all_extracted)
        
        # 5. Temporal analysis with enhanced insights
        temporal_analysis = self._enhanced_temporal_analysis(all_extracted)
        results['temporal_insights'] = temporal_analysis
        
        # 6. Geographic insights
        geographic_insights = self._extract_geographic_insights(all_extracted)
        results['geographic_insights'] = geographic_insights
        
        # Save enhanced results
        self._save_enhanced_results(prefix, {
            'keyword_comments': keyword_comments,
            'flag_comments': flag_comments, 
            'location_comments': location_comments,
            'all_extracted': all_extracted,
            'temporal_analysis': temporal_analysis,
            'geographic_insights': geographic_insights
        })
        
        logger.info(f"Enhanced analysis complete for {prefix}")
        return results
    
    def _extract_keyword_comments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced keyword extraction with confidence scoring"""
        keywords = TASK1_CONFIG['keywords']
        pattern = create_regex_pattern(keywords, case_sensitive=False)
        
        matches = extract_matching_comments(df, pattern)
        
        if not matches.empty:
            # Add confidence scoring based on keyword strength
            matches = matches.copy()
            matches['extraction_method'] = 'keyword'
            matches['confidence_score'] = matches['original_comment'].apply(
                self._calculate_keyword_confidence
            )
            matches['matched_keywords'] = matches['original_comment'].apply(
                lambda x: self._find_matched_keywords(x, keywords)
            )
        
        return matches
    
    def _extract_flag_emoji_comments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced flag emoji extraction with country identification"""
        pattern = re.compile(TASK1_CONFIG['flag_emoji_pattern'])
        matches = df[df['original_comment'].str.contains(pattern, na=False)].copy()
        
        if not matches.empty:
            matches['extraction_method'] = 'flag_emoji'
            matches['confidence_score'] = 0.9  # High confidence for flag emojis
            matches['extracted_flags'] = matches['original_comment'].apply(
                self._extract_flag_emojis
            )
            matches['estimated_countries'] = matches['extracted_flags'].apply(
                self._flags_to_countries
            )
        
        return matches
    
    def _extract_location_comments_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced location extraction with NER-style approach"""
        # Enhanced location pattern with more countries and cities
        enhanced_locations = [
            # Major countries
            'India', 'Turkey', 'USA', 'United States', 'UK', 'United Kingdom', 
            'Canada', 'Australia', 'Germany', 'France', 'Brazil', 'Japan', 
            'China', 'Russia', 'Italy', 'Spain', 'Mexico', 'Argentina', 
            'Egypt', 'Nigeria', 'Kenya', 'Pakistan', 'Bangladesh',
            'Indonesia', 'Thailand', 'Philippines', 'Vietnam', 'South Korea',
            'Netherlands', 'Sweden', 'Norway', 'Denmark', 'Finland',
            
            # Major cities with city indicators
            r'[A-Za-z]+[Cc]ity', 'New York', 'Los Angeles', 'London', 'Paris', 
            'Tokyo', 'Beijing', 'Mumbai', 'Delhi', 'Istanbul', 'Moscow',
            'Berlin', 'Madrid', 'Rome', 'Sydney', 'Toronto', 'Montreal'
        ]
        
        location_pattern = r'\b(' + '|'.join(re.escape(loc) for loc in enhanced_locations) + r')\b'
        pattern = re.compile(location_pattern, re.IGNORECASE)
        
        matches = df[df['original_comment'].str.contains(pattern, na=False)].copy()
        
        if not matches.empty:
            matches['extraction_method'] = 'location_mention'
            matches['extracted_locations'] = matches['original_comment'].apply(
                lambda x: self._extract_all_locations(x, pattern)
            )
            matches['confidence_score'] = matches['extracted_locations'].apply(
                self._calculate_location_confidence
            )
        
        return matches
    
    def _combine_extracted_comments(self, *dfs) -> pd.DataFrame:
        """Combine multiple extraction results and remove duplicates"""
        if not any(len(df) > 0 for df in dfs):
            return pd.DataFrame()
        
        # Combine all non-empty dataframes
        non_empty_dfs = [df for df in dfs if len(df) > 0]
        if not non_empty_dfs:
            return pd.DataFrame()
        
        combined = pd.concat(non_empty_dfs, ignore_index=True)
        
        # Remove duplicates based on comment content, keep highest confidence
        if 'confidence_score' in combined.columns:
            combined = combined.sort_values('confidence_score', ascending=False)
        
        combined = combined.drop_duplicates(subset=['original_comment'], keep='first')
        
        return combined.reset_index(drop=True)
    
    def _enhanced_temporal_analysis(self, df: pd.DataFrame) -> Dict:
        """Enhanced temporal analysis with insights"""
        if df.empty:
            return {}
        
        # Basic temporal distribution
        temporal_dist = analyze_temporal_distribution(df, freq='M')
        
        if temporal_dist.empty:
            return {}
        
        # Enhanced insights
        insights = {
            'total_periods': len(temporal_dist),
            'peak_period': str(temporal_dist.idxmax()),
            'peak_count': int(temporal_dist.max()),
            'average_per_period': float(temporal_dist.mean()),
            'temporal_trend': self._calculate_temporal_trend(temporal_dist),
            'seasonal_patterns': self._detect_seasonal_patterns(temporal_dist)
        }
        
        return insights
    
    def _extract_geographic_insights(self, df: pd.DataFrame) -> Dict:
        """Extract geographic insights from comments"""
        if df.empty:
            return {}
        
        insights = {
            'extraction_methods': df['extraction_method'].value_counts().to_dict(),
            'average_confidence': float(df['confidence_score'].mean()) if 'confidence_score' in df.columns else 0,
            'high_confidence_comments': int((df['confidence_score'] > 0.8).sum()) if 'confidence_score' in df.columns else 0
        }
        
        # Extract top countries if available
        if 'estimated_countries' in df.columns:
            all_countries = []
            for countries_list in df['estimated_countries'].dropna():
                if isinstance(countries_list, list):
                    all_countries.extend(countries_list)
            
            if all_countries:
                country_counts = pd.Series(all_countries).value_counts()
                insights['top_countries'] = country_counts.head(10).to_dict()
        
        return insights
    
    def _save_enhanced_results(self, prefix: str, data: Dict) -> None:
        """Save enhanced analysis results"""
        base_path = Path(f"{prefix}_outputs") / "enhanced"
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual extraction results
        for data_type, df in data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                file_path = base_path / f"{prefix}_enhanced_{data_type}.csv"
                safe_save_csv(df, file_path)
        
        # Save insights as JSON
        insights_data = {
            'temporal_analysis': data.get('temporal_analysis', {}),
            'geographic_insights': data.get('geographic_insights', {})
        }
        
        import json
        insights_path = base_path / f"{prefix}_enhanced_insights.json"
        try:
            with open(insights_path, 'w') as f:
                json.dump(insights_data, f, indent=2, default=str)
            logger.info(f"Saved insights to {insights_path}")
        except Exception as e:
            logger.error(f"Error saving insights: {e}")
    
    # Helper methods for enhanced analysis
    def _calculate_keyword_confidence(self, text: str) -> float:
        """Calculate confidence score for keyword matches"""
        if not isinstance(text, str):
            return 0.0
        
        text_lower = text.lower()
        keywords = TASK1_CONFIG['keywords']
        
        # Base confidence for any match
        confidence = 0.5
        
        # Boost for multiple keywords
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        confidence += matches * 0.1
        
        # Boost for specific phrases
        if 'from' in text_lower:
            confidence += 0.2
        
        # Reduce for very short comments
        if len(text) < 10:
            confidence -= 0.2
        
        return min(1.0, max(0.0, confidence))
    
    def _find_matched_keywords(self, text: str, keywords: List[str]) -> List[str]:
        """Find which keywords matched in the text"""
        if not isinstance(text, str):
            return []
        
        text_lower = text.lower()
        return [kw for kw in keywords if kw in text_lower]
    
    def _extract_flag_emojis(self, text: str) -> List[str]:
        """Extract flag emojis from text"""
        if not isinstance(text, str):
            return []
        
        pattern = re.compile(TASK1_CONFIG['flag_emoji_pattern'])
        return pattern.findall(text)
    
    def _flags_to_countries(self, flags: List[str]) -> List[str]:
        """Convert flag emojis to country names (simplified mapping)"""
        # This is a simplified implementation
        # In practice, you'd use a comprehensive flag-to-country mapping
        flag_mapping = {
            '🇺🇸': 'United States', '🇮🇳': 'India', '🇹🇷': 'Turkey',
            '🇬🇧': 'United Kingdom', '🇨🇦': 'Canada', '🇦🇺': 'Australia',
            '🇩🇪': 'Germany', '🇫🇷': 'France', '🇧🇷': 'Brazil',
            '🇯🇵': 'Japan', '🇨🇳': 'China', '🇷🇺': 'Russia'
        }
        
        return [flag_mapping.get(flag, f'Unknown_{flag}') for flag in flags]
    
    def _extract_all_locations(self, text: str, pattern: re.Pattern) -> List[str]:
        """Extract all location matches from text"""
        if not isinstance(text, str):
            return []
        
        return pattern.findall(text)
    
    def _calculate_location_confidence(self, locations: List[str]) -> float:
        """Calculate confidence for location extractions"""
        if not locations:
            return 0.0
        
        # Higher confidence for more specific locations
        base_confidence = 0.6
        
        # Boost for multiple locations
        if len(locations) > 1:
            base_confidence += 0.1
        
        # Boost for known major cities/countries
        major_locations = ['India', 'USA', 'UK', 'Canada', 'Australia', 'New York', 'London']
        if any(loc in major_locations for loc in locations):
            base_confidence += 0.2
        
        return min(1.0, base_confidence)
    
    def _calculate_temporal_trend(self, temporal_dist: pd.Series) -> str:
        """Calculate overall temporal trend"""
        if len(temporal_dist) < 2:
            return "insufficient_data"
        
        # Simple linear trend calculation
        x = range(len(temporal_dist))
        y = temporal_dist.values
        
        # Calculate correlation coefficient as trend indicator
        correlation = np.corrcoef(x, y)[0, 1]
        
        if correlation > 0.3:
            return "increasing"
        elif correlation < -0.3:
            return "decreasing" 
        else:
            return "stable"
    
    def _detect_seasonal_patterns(self, temporal_dist: pd.Series) -> Dict:
        """Detect seasonal patterns in temporal data"""
        # This is a simplified implementation
        # Could be enhanced with proper seasonal decomposition
        
        if len(temporal_dist) < 12:
            return {"pattern": "insufficient_data"}
        
        # Group by month if we have enough data
        try:
            monthly_avg = temporal_dist.groupby(temporal_dist.index.month).mean()
            peak_month = monthly_avg.idxmax()
            
            return {
                "pattern": "detected",
                "peak_month": int(peak_month),
                "seasonal_variation": float(monthly_avg.std())
            }
        except:
            return {"pattern": "analysis_failed"}

def main():
    """Main execution function"""
    logger.info("Starting Enhanced Geographic Analysis (Task 1)")
    
    analyzer = EnhancedGeographicAnalyzer()
    
    # Process both datasets
    for dataset_key, file_path in INPUT_FILES.items():
        if file_path.exists():
            prefix = get_file_prefix(str(file_path))
            results = analyzer.analyze_single_dataset(file_path, prefix)
            analyzer.results[prefix] = results
        else:
            logger.warning(f"Input file not found: {file_path}")
    
    # Generate summary report
    generate_summary_report(analyzer.results)
    
    logger.info("Enhanced Geographic Analysis completed successfully!")

def generate_summary_report(results: Dict) -> None:
    """Generate summary report of enhanced analysis"""
    report_lines = ["# Enhanced Geographic Analysis Report\n"]
    
    for prefix, data in results.items():
        if not data:
            continue
            
        report_lines.append(f"## {prefix.title()} Dataset")
        report_lines.append(f"- **Total Comments**: {data['total_comments']:,}")
        report_lines.append(f"- **Data Quality Score**: {data['quality_score']:.1f}/100")
        
        extracted = data.get('extracted_data', {})
        if extracted:
            report_lines.append(f"- **Geographic Comments Found**: {extracted.get('total_unique', 0):,}")
            report_lines.append(f"  - Keywords: {extracted.get('keywords', 0):,}")
            report_lines.append(f"  - Flag Emojis: {extracted.get('flags', 0):,}")
            report_lines.append(f"  - Location Mentions: {extracted.get('locations', 0):,}")
        
        temporal = data.get('temporal_insights', {})
        if temporal:
            report_lines.append(f"- **Peak Period**: {temporal.get('peak_period', 'N/A')}")
            report_lines.append(f"- **Temporal Trend**: {temporal.get('temporal_trend', 'N/A')}")
        
        geographic = data.get('geographic_insights', {})
        if geographic:
            report_lines.append(f"- **Average Confidence**: {geographic.get('average_confidence', 0):.2f}")
            if 'top_countries' in geographic:
                top_countries = list(geographic['top_countries'].keys())[:3]
                report_lines.append(f"- **Top Countries**: {', '.join(top_countries)}")
        
        report_lines.append("\n")
    
    # Save report
    report_path = Path("enhanced_geographic_analysis_report.md")
    try:
        report_path.write_text("\n".join(report_lines))
        logger.info(f"Summary report saved to {report_path}")
    except Exception as e:
        logger.error(f"Error saving report: {e}")

if __name__ == "__main__":
    main() 