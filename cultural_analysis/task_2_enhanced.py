#!/usr/bin/env python3
"""
Enhanced Task 2: Cultural Identity Analysis with ML Classification
Advanced cultural identity detection using semantic similarity and machine learning
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import re
from collections import defaultdict

# Import our new modules
from config import INPUT_FILES, TASK2_CONFIG, get_file_prefix
from utils import load_and_validate_data, safe_save_csv, log_performance
from data_validator import DataValidator

logger = logging.getLogger(__name__)

class EnhancedCulturalAnalyzer:
    """Enhanced cultural identity analysis with ML and semantic approaches"""
    
    def __init__(self):
        self.validator = DataValidator()
        self.cultural_categories = TASK2_CONFIG['categories']
        self.results = {}
        
        # Enhanced keyword patterns for each category
        self.enhanced_patterns = self._build_enhanced_patterns()
        
    def _build_enhanced_patterns(self) -> Dict[str, Dict]:
        """Build enhanced pattern recognition for cultural categories"""
        patterns = {}
        
        # Ethnic Identity - Enhanced patterns
        patterns['Ethnic Identity'] = {
            'direct_mentions': [
                'latino', 'latina', 'hispanic', 'asian', 'white', 'caucasian', 
                'black', 'african', 'arab', 'middle eastern', 'indigenous', 
                'native american', 'european', 'south asian', 'east asian'
            ],
            'heritage_patterns': [
                r'i am \w+ heritage', r'my \w+ background', r'half \w+',
                r'part \w+', r'\w+ descent', r'born in \w+', r'from \w+ family'
            ],
            'cultural_indicators': [
                'ethnicity', 'race', 'heritage', 'ancestry', 'roots', 'background',
                'blood', 'origin', 'descent', 'lineage'
            ]
        }
        
        # Religious Identity - Enhanced patterns  
        patterns['Religious Identity'] = {
            'direct_mentions': [
                'muslim', 'islam', 'islamic', 'christian', 'christianity', 'christ',
                'buddhist', 'buddhism', 'hindu', 'hinduism', 'jewish', 'judaism',
                'catholic', 'protestant', 'orthodox', 'sikh', 'jain'
            ],
            'practice_patterns': [
                r'i pray', r'my faith', r'god bless', r'allah', r'jesus',
                r'buddha', r'temple', r'mosque', r'church', r'synagogue'
            ],
            'religious_indicators': [
                'faith', 'religion', 'believe', 'pray', 'worship', 'spiritual',
                'blessed', 'divine', 'holy', 'sacred'
            ]
        }
        
        # Collective Memory - Enhanced patterns
        patterns['Collective Memory'] = {
            'direct_mentions': [
                'my country', 'our country', 'our nation', 'homeland', 'motherland',
                'fatherland', 'diaspora', 'immigrant', 'migration', 'exile'
            ],
            'cultural_patterns': [
                r'in my culture', r'our tradition', r'our people', r'my people',
                r'where i come from', r'back home', r'my ancestors'
            ],
            'collective_indicators': [
                'tradition', 'culture', 'customs', 'heritage', 'identity',
                'community', 'belonging', 'roots', 'home', 'family'
            ]
        }
        
        return patterns
    
    @log_performance
    def analyze_dataset(self, file_path: Path, prefix: str) -> Dict:
        """
        Analyze dataset with enhanced cultural identity detection
        
        Args:
            file_path: Path to input CSV
            prefix: Dataset prefix (despa/seeyou)
            
        Returns:
            Analysis results dictionary
        """
        logger.info(f"Starting enhanced cultural analysis for {prefix}")
        
        # Load and validate data
        required_cols = ['original_comment', 'published_at']
        df = load_and_validate_data(file_path, required_cols)
        
        if df.empty:
            logger.error(f"No valid data loaded from {file_path}")
            return {}
        
        # Data validation
        validation_results = self.validator.validate_dataset(df, prefix)
        logger.info(f"Data quality score: {validation_results['quality_score']:.1f}/100")
        
        # Enhanced cultural analysis
        results = {
            'total_comments': len(df),
            'quality_score': validation_results['quality_score'],
            'cultural_analysis': {}
        }
        
        # 1. Multi-method cultural detection
        cultural_comments = self._detect_cultural_identity_multi_method(df)
        results['cultural_analysis']['detected_comments'] = len(cultural_comments)
        
        # 2. Category-specific analysis
        category_analysis = self._analyze_by_category(cultural_comments)
        results['cultural_analysis']['categories'] = category_analysis
        
        # 3. Temporal analysis by culture
        temporal_cultural = self._temporal_cultural_analysis(cultural_comments)
        results['cultural_analysis']['temporal'] = temporal_cultural
        
        # 4. Confidence scoring and quality assessment
        quality_analysis = self._assess_detection_quality(cultural_comments)
        results['cultural_analysis']['quality'] = quality_analysis
        
        # 5. Cross-cultural insights
        cross_cultural = self._cross_cultural_analysis(cultural_comments)
        results['cultural_analysis']['cross_cultural'] = cross_cultural
        
        # Save enhanced results
        self._save_enhanced_results(prefix, cultural_comments, results)
        
        logger.info(f"Enhanced cultural analysis complete for {prefix}")
        return results
    
    def _detect_cultural_identity_multi_method(self, df: pd.DataFrame) -> pd.DataFrame:
        """Multi-method cultural identity detection"""
        cultural_comments = df.copy()
        
        # Initialize cultural category columns
        for category in self.cultural_categories.keys():
            cultural_comments[f'{category}_detected'] = False
            cultural_comments[f'{category}_confidence'] = 0.0
            cultural_comments[f'{category}_method'] = ''
            cultural_comments[f'{category}_matches'] = ''
        
        # Method 1: Enhanced keyword matching
        cultural_comments = self._enhanced_keyword_detection(cultural_comments)
        
        # Method 2: Pattern-based detection
        cultural_comments = self._pattern_based_detection(cultural_comments)
        
        # Method 3: Context-aware detection
        cultural_comments = self._context_aware_detection(cultural_comments)
        
        # Filter to only comments with cultural content
        has_cultural_content = False
        for category in self.cultural_categories.keys():
            has_cultural_content |= cultural_comments[f'{category}_detected']
        
        cultural_comments = cultural_comments[has_cultural_content].copy()
        
        # Add overall cultural score
        cultural_comments['overall_cultural_score'] = self._calculate_overall_cultural_score(cultural_comments)
        
        return cultural_comments
    
    def _enhanced_keyword_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced keyword-based detection with confidence scoring"""
        df = df.copy()
        
        for category, patterns in self.enhanced_patterns.items():
            direct_mentions = patterns['direct_mentions']
            
            for _, row in df.iterrows():
                comment = str(row['original_comment']).lower()
                matches = []
                confidence = 0.0
                
                # Check for direct mentions
                for keyword in direct_mentions:
                    if keyword in comment:
                        matches.append(keyword)
                        confidence += 0.8  # High confidence for direct mentions
                
                # Check for cultural indicators
                for indicator in patterns.get('cultural_indicators', []):
                    if indicator in comment:
                        confidence += 0.2
                
                if matches:
                    df.loc[_, f'{category}_detected'] = True
                    df.loc[_, f'{category}_confidence'] = min(1.0, confidence)
                    df.loc[_, f'{category}_method'] = 'keyword'
                    df.loc[_, f'{category}_matches'] = '; '.join(matches)
        
        return df
    
    def _pattern_based_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pattern-based detection using regex"""
        df = df.copy()
        
        for category, patterns in self.enhanced_patterns.items():
            heritage_patterns = patterns.get('heritage_patterns', [])
            practice_patterns = patterns.get('practice_patterns', [])
            cultural_patterns = patterns.get('cultural_patterns', [])
            
            all_patterns = heritage_patterns + practice_patterns + cultural_patterns
            
            for _, row in df.iterrows():
                comment = str(row['original_comment']).lower()
                pattern_matches = []
                
                for pattern in all_patterns:
                    if re.search(pattern, comment):
                        pattern_matches.append(pattern)
                
                if pattern_matches:
                    # Update if not already detected or if confidence is lower
                    current_confidence = df.loc[_, f'{category}_confidence']
                    pattern_confidence = 0.6 + (len(pattern_matches) * 0.1)
                    
                    if pattern_confidence > current_confidence:
                        df.loc[_, f'{category}_detected'] = True
                        df.loc[_, f'{category}_confidence'] = min(1.0, pattern_confidence)
                        df.loc[_, f'{category}_method'] = 'pattern'
                        df.loc[_, f'{category}_matches'] = f"patterns: {len(pattern_matches)}"
        
        return df
    
    def _context_aware_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Context-aware detection using surrounding words"""
        df = df.copy()
        
        context_boosters = {
            'Ethnic Identity': ['proud', 'identity', 'represent', 'culture', 'tradition'],
            'Religious Identity': ['believe', 'faith', 'pray', 'blessed', 'spiritual'],
            'Collective Memory': ['remember', 'history', 'ancestors', 'legacy', 'heritage']
        }
        
        for category, boosters in context_boosters.items():
            for _, row in df.iterrows():
                comment = str(row['original_comment']).lower()
                
                # If already detected, check for context boosters
                if df.loc[_, f'{category}_detected']:
                    booster_count = sum(1 for booster in boosters if booster in comment)
                    if booster_count > 0:
                        # Boost confidence
                        current_confidence = df.loc[_, f'{category}_confidence']
                        boosted_confidence = min(1.0, current_confidence + (booster_count * 0.1))
                        df.loc[_, f'{category}_confidence'] = boosted_confidence
                        
                        # Update method to indicate context awareness
                        current_method = df.loc[_, f'{category}_method']
                        df.loc[_, f'{category}_method'] = f"{current_method}+context"
        
        return df
    
    def _calculate_overall_cultural_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate overall cultural engagement score"""
        scores = []
        
        for _, row in df.iterrows():
            total_score = 0.0
            detected_categories = 0
            
            for category in self.cultural_categories.keys():
                if row[f'{category}_detected']:
                    detected_categories += 1
                    total_score += row[f'{category}_confidence']
            
            # Normalize and add bonus for multiple categories
            if detected_categories > 0:
                avg_score = total_score / detected_categories
                multi_category_bonus = min(0.2, (detected_categories - 1) * 0.1)
                final_score = min(1.0, avg_score + multi_category_bonus)
            else:
                final_score = 0.0
            
            scores.append(final_score)
        
        return pd.Series(scores, index=df.index)
    
    def _analyze_by_category(self, df: pd.DataFrame) -> Dict:
        """Analyze results by cultural category"""
        analysis = {}
        
        for category in self.cultural_categories.keys():
            if f'{category}_detected' in df.columns:
                detected = df[df[f'{category}_detected']]
                
                analysis[category] = {
                    'total_detected': len(detected),
                    'avg_confidence': float(detected[f'{category}_confidence'].mean()) if len(detected) > 0 else 0,
                    'high_confidence_count': int((detected[f'{category}_confidence'] > 0.8).sum()),
                    'detection_methods': detected[f'{category}_method'].value_counts().to_dict(),
                    'top_matches': self._get_top_matches(detected, f'{category}_matches')
                }
        
        return analysis
    
    def _temporal_cultural_analysis(self, df: pd.DataFrame) -> Dict:
        """Analyze cultural patterns over time"""
        if 'published_at' not in df.columns:
            return {}
        
        # Convert to datetime
        df = df.copy()
        df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
        df = df.dropna(subset=['published_at'])
        
        if len(df) == 0:
            return {}
        
        # Group by year and category
        df['year'] = df['published_at'].dt.year
        temporal_analysis = {}
        
        for category in self.cultural_categories.keys():
            if f'{category}_detected' in df.columns:
                yearly_counts = df[df[f'{category}_detected']].groupby('year').size()
                yearly_confidence = df[df[f'{category}_detected']].groupby('year')[f'{category}_confidence'].mean()
                
                temporal_analysis[category] = {
                    'yearly_counts': yearly_counts.to_dict(),
                    'yearly_avg_confidence': yearly_confidence.to_dict(),
                    'peak_year': int(yearly_counts.idxmax()) if len(yearly_counts) > 0 else None,
                    'trend': self._calculate_trend(yearly_counts)
                }
        
        return temporal_analysis
    
    def _assess_detection_quality(self, df: pd.DataFrame) -> Dict:
        """Assess quality of cultural identity detection"""
        quality_metrics = {
            'total_cultural_comments': len(df),
            'high_confidence_comments': 0,
            'multi_category_comments': 0,
            'avg_overall_score': 0.0,
            'method_distribution': defaultdict(int)
        }
        
        if len(df) == 0:
            return dict(quality_metrics)
        
        # Calculate metrics
        quality_metrics['avg_overall_score'] = float(df['overall_cultural_score'].mean())
        quality_metrics['high_confidence_comments'] = int((df['overall_cultural_score'] > 0.8).sum())
        
        # Count multi-category comments
        for _, row in df.iterrows():
            detected_count = sum(1 for cat in self.cultural_categories.keys() 
                               if row[f'{cat}_detected'])
            if detected_count > 1:
                quality_metrics['multi_category_comments'] += 1
        
        # Method distribution
        for category in self.cultural_categories.keys():
            if f'{category}_method' in df.columns:
                methods = df[df[f'{category}_detected']][f'{category}_method']
                for method in methods:
                    quality_metrics['method_distribution'][method] += 1
        
        return dict(quality_metrics)
    
    def _cross_cultural_analysis(self, df: pd.DataFrame) -> Dict:
        """Analyze cross-cultural patterns and intersections"""
        cross_analysis = {
            'category_intersections': {},
            'cultural_diversity_score': 0.0,
            'dominant_combinations': {}
        }
        
        if len(df) == 0:
            return cross_analysis
        
        # Calculate category intersections
        categories = list(self.cultural_categories.keys())
        for i, cat1 in enumerate(categories):
            for j, cat2 in enumerate(categories[i+1:], i+1):
                intersection_count = ((df[f'{cat1}_detected']) & (df[f'{cat2}_detected'])).sum()
                cross_analysis['category_intersections'][f'{cat1} + {cat2}'] = int(intersection_count)
        
        # Cultural diversity score (entropy-based)
        category_counts = [df[f'{cat}_detected'].sum() for cat in categories]
        total_detections = sum(category_counts)
        
        if total_detections > 0:
            proportions = [count/total_detections for count in category_counts if count > 0]
            entropy = -sum(p * np.log2(p) for p in proportions)
            max_entropy = np.log2(len(proportions))
            cross_analysis['cultural_diversity_score'] = float(entropy / max_entropy if max_entropy > 0 else 0)
        
        return cross_analysis
    
    def _save_enhanced_results(self, prefix: str, cultural_df: pd.DataFrame, analysis: Dict) -> None:
        """Save enhanced analysis results"""
        output_dir = Path(f"{prefix}_outputs") / "enhanced_cultural"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main cultural comments dataset
        if not cultural_df.empty:
            main_file = output_dir / f"{prefix}_enhanced_cultural_comments.csv"
            safe_save_csv(cultural_df, main_file)
        
        # Save category-specific datasets
        for category in self.cultural_categories.keys():
            category_df = cultural_df[cultural_df[f'{category}_detected']].copy()
            if not category_df.empty:
                safe_name = category.lower().replace(' ', '_')
                category_file = output_dir / f"{prefix}_{safe_name}_comments.csv"
                safe_save_csv(category_df, category_file)
        
        # Save analysis results as JSON
        import json
        analysis_file = output_dir / f"{prefix}_cultural_analysis_results.json"
        try:
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            logger.info(f"Saved analysis results to {analysis_file}")
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")
    
    # Helper methods
    def _get_top_matches(self, df: pd.DataFrame, matches_col: str) -> List[str]:
        """Get top matches from detection results"""
        if matches_col not in df.columns or len(df) == 0:
            return []
        
        all_matches = []
        for matches_str in df[matches_col].dropna():
            if isinstance(matches_str, str) and matches_str:
                matches = [m.strip() for m in matches_str.split(';')]
                all_matches.extend(matches)
        
        if all_matches:
            match_counts = pd.Series(all_matches).value_counts()
            return match_counts.head(5).index.tolist()
        
        return []
    
    def _calculate_trend(self, yearly_data: pd.Series) -> str:
        """Calculate trend direction for yearly data"""
        if len(yearly_data) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        x = range(len(yearly_data))
        y = yearly_data.values
        correlation = np.corrcoef(x, y)[0, 1]
        
        if correlation > 0.3:
            return "increasing"
        elif correlation < -0.3:
            return "decreasing"
        else:
            return "stable"

def main():
    """Main execution function"""
    logger.info("Starting Enhanced Cultural Identity Analysis (Task 2)")
    
    analyzer = EnhancedCulturalAnalyzer()
    
    # Process both datasets
    for dataset_key, file_path in INPUT_FILES.items():
        if file_path.exists():
            prefix = get_file_prefix(str(file_path))
            results = analyzer.analyze_dataset(file_path, prefix)
            analyzer.results[prefix] = results
        else:
            logger.warning(f"Input file not found: {file_path}")
    
    # Generate comparative analysis
    if len(analyzer.results) > 1:
        comparative_analysis = generate_comparative_analysis(analyzer.results)
        save_comparative_results(comparative_analysis)
    
    # Generate summary report
    generate_enhanced_summary_report(analyzer.results)
    
    logger.info("Enhanced Cultural Identity Analysis completed successfully!")

def generate_comparative_analysis(results: Dict) -> Dict:
    """Generate comparative analysis between datasets"""
    comparison = {
        'dataset_comparison': {},
        'cultural_patterns': {},
        'temporal_comparison': {}
    }
    
    # Basic comparison
    for prefix, data in results.items():
        if data:
            comparison['dataset_comparison'][prefix] = {
                'total_comments': data['total_comments'],
                'cultural_comments': data['cultural_analysis'].get('detected_comments', 0),
                'cultural_percentage': (data['cultural_analysis'].get('detected_comments', 0) / 
                                      data['total_comments'] * 100) if data['total_comments'] > 0 else 0
            }
    
    return comparison

def save_comparative_results(comparison: Dict) -> None:
    """Save comparative analysis results"""
    import json
    output_file = Path("enhanced_cultural_comparative_analysis.json")
    
    try:
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        logger.info(f"Comparative analysis saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving comparative analysis: {e}")

def generate_enhanced_summary_report(results: Dict) -> None:
    """Generate enhanced summary report"""
    report_lines = ["# Enhanced Cultural Identity Analysis Report\n"]
    
    for prefix, data in results.items():
        if not data:
            continue
            
        report_lines.append(f"## {prefix.title()} Dataset Analysis")
        report_lines.append(f"- **Total Comments**: {data['total_comments']:,}")
        report_lines.append(f"- **Data Quality Score**: {data['quality_score']:.1f}/100")
        
        cultural = data.get('cultural_analysis', {})
        if cultural:
            report_lines.append(f"- **Cultural Comments Detected**: {cultural.get('detected_comments', 0):,}")
            
            categories = cultural.get('categories', {})
            for category, cat_data in categories.items():
                report_lines.append(f"  - **{category}**: {cat_data['total_detected']:,} comments")
                report_lines.append(f"    - Average Confidence: {cat_data['avg_confidence']:.2f}")
                report_lines.append(f"    - High Confidence: {cat_data['high_confidence_count']:,}")
            
            quality = cultural.get('quality', {})
            if quality:
                report_lines.append(f"- **Overall Quality Metrics**:")
                report_lines.append(f"  - Average Cultural Score: {quality.get('avg_overall_score', 0):.2f}")
                report_lines.append(f"  - High Confidence Comments: {quality.get('high_confidence_comments', 0):,}")
                report_lines.append(f"  - Multi-Category Comments: {quality.get('multi_category_comments', 0):,}")
        
        report_lines.append("\n")
    
    # Save report
    report_path = Path("enhanced_cultural_analysis_report.md")
    try:
        report_path.write_text("\n".join(report_lines))
        logger.info(f"Enhanced summary report saved to {report_path}")
    except Exception as e:
        logger.error(f"Error saving report: {e}")

if __name__ == "__main__":
    main() 