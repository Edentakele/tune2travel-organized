#!/usr/bin/env python3
"""
Demo Script: Enhanced Cultural Analysis Features
Demonstrates the new capabilities with real data
"""

import pandas as pd
import logging
from pathlib import Path
import json

# Import our enhanced modules
from config import INPUT_FILES, TASK1_CONFIG, TASK2_CONFIG
from utils import load_and_validate_data, create_regex_pattern, extract_matching_comments
from data_validator import DataValidator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_data_validation():
    """Demonstrate data validation capabilities"""
    logger.info("🔍 DEMO: Data Validation")
    logger.info("-" * 40)
    
    validator = DataValidator()
    
    for dataset_key, file_path in INPUT_FILES.items():
        if file_path.exists():
            logger.info(f"Validating {dataset_key} dataset...")
            
            # Load a sample of data for validation
            df_sample = pd.read_csv(file_path, nrows=10000)  # Sample for demo
            results = validator.validate_dataset(df_sample, dataset_key)
            
            logger.info(f"✅ {dataset_key} Analysis Complete:")
            logger.info(f"   📊 Total rows analyzed: {results['total_rows']:,}")
            logger.info(f"   🎯 Quality Score: {results['quality_score']:.1f}/100")
            
            # Show some specific insights
            if results.get('comment_warnings'):
                logger.info(f"   ⚠️ Comment warnings: {len(results['comment_warnings'])}")
            if results.get('parsing_errors', 0) > 0:
                logger.info(f"   ❌ Timestamp parsing errors: {results['parsing_errors']}")
        else:
            logger.warning(f"File not found: {file_path}")
    
    logger.info("\n")

def demo_enhanced_geographic_analysis():
    """Demonstrate enhanced geographic analysis"""
    logger.info("🌍 DEMO: Enhanced Geographic Analysis")
    logger.info("-" * 40)
    
    for dataset_key, file_path in INPUT_FILES.items():
        if file_path.exists():
            logger.info(f"Analyzing geographic patterns in {dataset_key}...")
            
            # Load sample data
            df = pd.read_csv(file_path, nrows=5000)  # Sample for demo
            logger.info(f"   📄 Loaded {len(df):,} comments for analysis")
            
            # Enhanced keyword detection
            keywords = TASK1_CONFIG['keywords']
            keyword_pattern = create_regex_pattern(keywords, case_sensitive=False)
            keyword_matches = extract_matching_comments(df, keyword_pattern)
            
            logger.info(f"   🔍 Keyword matches: {len(keyword_matches):,}")
            
            # Enhanced location detection
            location_pattern = create_regex_pattern(['India', 'USA', 'UK', 'Canada', 'Turkey', 'Brazil'], case_sensitive=False)
            location_matches = extract_matching_comments(df, location_pattern)
            
            logger.info(f"   📍 Location mentions: {len(location_matches):,}")
            
            # Flag emoji detection
            flag_pattern = r'🇮🇳|🇺🇸|🇬🇧|🇨🇦|🇹🇷|🇧🇷'
            flag_matches = df[df['original_comment'].str.contains(flag_pattern, na=False)]
            
            logger.info(f"   🏳️ Flag emoji comments: {len(flag_matches):,}")
            
            # Calculate geographic coverage percentage
            all_geographic = pd.concat([keyword_matches, location_matches, flag_matches]).drop_duplicates()
            coverage_percentage = (len(all_geographic) / len(df)) * 100
            
            logger.info(f"   📈 Geographic coverage: {coverage_percentage:.2f}% of comments")
            
        else:
            logger.warning(f"File not found: {file_path}")
    
    logger.info("\n")

def demo_enhanced_cultural_analysis():
    """Demonstrate enhanced cultural identity analysis"""
    logger.info("🎭 DEMO: Enhanced Cultural Identity Analysis")
    logger.info("-" * 40)
    
    # Enhanced cultural patterns
    cultural_patterns = {
        'Ethnic Identity': ['latino', 'hispanic', 'asian', 'african', 'white', 'black'],
        'Religious Identity': ['muslim', 'christian', 'hindu', 'buddhist', 'jewish'],
        'Collective Memory': ['my country', 'our culture', 'homeland', 'tradition']
    }
    
    for dataset_key, file_path in INPUT_FILES.items():
        if file_path.exists():
            logger.info(f"Analyzing cultural patterns in {dataset_key}...")
            
            # Load sample data
            df = pd.read_csv(file_path, nrows=5000)  # Sample for demo
            
            category_results = {}
            
            for category, keywords in cultural_patterns.items():
                pattern = create_regex_pattern(keywords, case_sensitive=False)
                matches = extract_matching_comments(df, pattern)
                category_results[category] = len(matches)
                
                logger.info(f"   {category}: {len(matches):,} comments")
                
                # Show top matching keywords if any found
                if len(matches) > 0:
                    top_keywords = []
                    for keyword in keywords:
                        keyword_count = matches['original_comment'].str.lower().str.contains(keyword).sum()
                        if keyword_count > 0:
                            top_keywords.append(f"{keyword}({keyword_count})")
                    
                    if top_keywords:
                        logger.info(f"      Top terms: {', '.join(top_keywords[:3])}")
            
            # Calculate cultural diversity
            total_cultural = sum(category_results.values())
            cultural_percentage = (total_cultural / len(df)) * 100
            
            logger.info(f"   📊 Total cultural comments: {total_cultural:,}")
            logger.info(f"   📈 Cultural coverage: {cultural_percentage:.2f}% of comments")
            
        else:
            logger.warning(f"File not found: {file_path}")
    
    logger.info("\n")

def demo_temporal_analysis():
    """Demonstrate temporal analysis capabilities"""
    logger.info("📅 DEMO: Temporal Analysis")
    logger.info("-" * 40)
    
    for dataset_key, file_path in INPUT_FILES.items():
        if file_path.exists():
            logger.info(f"Analyzing temporal patterns in {dataset_key}...")
            
            # Load sample data
            df = pd.read_csv(file_path, nrows=5000)
            
            # Convert timestamp and analyze by year
            df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
            df = df.dropna(subset=['published_at'])
            df['year'] = df['published_at'].dt.year
            
            yearly_counts = df.groupby('year').size()
            
            logger.info(f"   📅 Data spans from {yearly_counts.index.min()} to {yearly_counts.index.max()}")
            logger.info(f"   📊 Peak year: {yearly_counts.idxmax()} ({yearly_counts.max():,} comments)")
            logger.info(f"   📈 Average per year: {yearly_counts.mean():.0f} comments")
            
            # Show top 3 years
            top_years = yearly_counts.nlargest(3)
            logger.info(f"   🏆 Top years: {dict(top_years)}")
            
        else:
            logger.warning(f"File not found: {file_path}")
    
    logger.info("\n")

def demo_advanced_insights():
    """Generate advanced insights combining multiple analysis methods"""
    logger.info("🧠 DEMO: Advanced Insights Generation")
    logger.info("-" * 40)
    
    insights = {
        'datasets_analyzed': 0,
        'total_comments_processed': 0,
        'geographic_patterns': {},
        'cultural_patterns': {},
        'quality_metrics': {}
    }
    
    for dataset_key, file_path in INPUT_FILES.items():
        if file_path.exists():
            logger.info(f"Generating insights for {dataset_key}...")
            
            # Load sample data
            df = pd.read_csv(file_path, nrows=3000)  # Smaller sample for demo
            insights['datasets_analyzed'] += 1
            insights['total_comments_processed'] += len(df)
            
            # Geographic insights
            geo_keywords = TASK1_CONFIG['keywords']
            geo_pattern = create_regex_pattern(geo_keywords, case_sensitive=False)
            geo_matches = extract_matching_comments(df, geo_pattern)
            geo_percentage = (len(geo_matches) / len(df)) * 100
            
            insights['geographic_patterns'][dataset_key] = {
                'total_geographic': len(geo_matches),
                'percentage': geo_percentage
            }
            
            # Cultural insights
            cultural_total = 0
            for category, keywords in TASK2_CONFIG['categories'].items():
                pattern = create_regex_pattern(keywords, case_sensitive=False)
                matches = extract_matching_comments(df, pattern)
                cultural_total += len(matches)
            
            cultural_percentage = (cultural_total / len(df)) * 100
            insights['cultural_patterns'][dataset_key] = {
                'total_cultural': cultural_total,
                'percentage': cultural_percentage
            }
            
            # Quality metrics
            comment_lengths = df['original_comment'].str.len()
            insights['quality_metrics'][dataset_key] = {
                'avg_comment_length': float(comment_lengths.mean()),
                'total_comments': len(df)
            }
    
    # Display summary insights
    logger.info("📋 SUMMARY INSIGHTS:")
    logger.info(f"   📊 Total datasets: {insights['datasets_analyzed']}")
    logger.info(f"   📝 Comments processed: {insights['total_comments_processed']:,}")
    
    # Geographic comparison
    if len(insights['geographic_patterns']) > 1:
        geo_data = insights['geographic_patterns']
        datasets = list(geo_data.keys())
        logger.info(f"   🌍 Geographic engagement:")
        for dataset in datasets:
            percentage = geo_data[dataset]['percentage']
            logger.info(f"      {dataset}: {percentage:.2f}%")
    
    # Cultural comparison
    if len(insights['cultural_patterns']) > 1:
        cultural_data = insights['cultural_patterns']
        logger.info(f"   🎭 Cultural engagement:")
        for dataset in datasets:
            percentage = cultural_data[dataset]['percentage']
            logger.info(f"      {dataset}: {percentage:.2f}%")
    
    # Save insights
    try:
        with open('demo_insights.json', 'w') as f:
            json.dump(insights, f, indent=2, default=str)
        logger.info("   💾 Insights saved to demo_insights.json")
    except Exception as e:
        logger.error(f"   ❌ Error saving insights: {e}")

def main():
    """Run the complete demonstration"""
    logger.info("🚀 Cultural Analysis Enhancement Demonstration")
    logger.info("=" * 60)
    logger.info("This demo showcases the new enhanced analysis capabilities")
    logger.info("=" * 60)
    
    # Run all demonstrations
    demo_data_validation()
    demo_enhanced_geographic_analysis()
    demo_enhanced_cultural_analysis()
    demo_temporal_analysis()
    demo_advanced_insights()
    
    logger.info("✨ DEMONSTRATION COMPLETE!")
    logger.info("=" * 60)
    logger.info("🎯 What's New:")
    logger.info("   ✅ Enhanced data validation with quality scoring")
    logger.info("   ✅ Improved geographic detection with multiple methods")
    logger.info("   ✅ Advanced cultural identity analysis")
    logger.info("   ✅ Comprehensive temporal pattern analysis")
    logger.info("   ✅ Cross-dataset comparative insights")
    logger.info("")
    logger.info("🚀 Next Steps:")
    logger.info("   1. Review the demo_insights.json file")
    logger.info("   2. Check SETUP_COMPLETION_REPORT.md for full roadmap")
    logger.info("   3. Try the enhanced scripts on your full dataset")
    logger.info("   4. Customize the config.py for your specific needs")

if __name__ == "__main__":
    main() 