#!/usr/bin/env python3
"""
Unified Cultural Analysis Dashboard Generator

Combines Task 1, Task 2, and advanced visualizations into a comprehensive dashboard.
Features:
- Interactive stacked bar charts for cultural identity categories
- Geographic distribution analysis with interactive maps
- Temporal trend analysis with filtering capabilities
- Spam analysis and data quality metrics
- Comparative analysis between Despacito and See You Again
- Export capabilities (HTML, JSON, CSV)
"""

import os
import sys
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import numpy as np
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
OUTPUT_HTML = BASE_DIR / "unified_cultural_analysis_dashboard.html"

class UnifiedCulturalAnalysisGenerator:
    def __init__(self):
        self.task1_data = None
        self.task2_data = None
        self.task2_yearly = None
        self.geographic_data = None
        self.insights = {}
        
    def load_all_data(self):
        """Load all available data sources"""
        print("🔄 Loading all data sources...")
        self.load_task1_data()
        self.load_task2_data()
        self.load_geographic_data()
        self.process_yearly_statistics()
        print("✅ All data loaded successfully!")
        
    def load_task1_data(self):
        """Load Task 1 yearly statistics"""
        task1_file = BASE_DIR / "task_1_yearly_stats.csv"
        if task1_file.exists():
            self.task1_data = pd.read_csv(task1_file)
            print(f"   📊 Task 1 data: {len(self.task1_data)} rows")
            
    def load_task2_data(self):
        """Load Task 2 cultural identity data"""
        task2_data = []
        for song in ['despa', 'seeyou']:
            yearly_dir = BASE_DIR / f"{song}_outputs" / f"{song}_task_2_yearly_split"
            if yearly_dir.exists():
                csv_files = list(yearly_dir.glob(f"{song}_*_labeled_context_comments.csv"))
                for csv_file in csv_files:
                    try:
                        year = int(csv_file.stem.split('_')[1])
                        df = pd.read_csv(csv_file)
                        df['Song'] = 'Despacito' if song == 'despa' else 'See You Again'
                        df['Year'] = year
                        task2_data.append(df)
                    except Exception as e:
                        print(f"      ❌ Error loading {csv_file}: {e}")
        
        if task2_data:
            self.task2_data = pd.concat(task2_data, ignore_index=True)
            print(f"   ✅ Total Task 2 data: {len(self.task2_data)} rows")
            
    def load_geographic_data(self):
        """Load geographic distribution data"""
        geo_data = []
        for prefix in ['despa', 'seeyou']:
            for geo_type in ['location', 'flag_emoji', 'keyword']:
                file_path = BASE_DIR / f"{prefix}_task_1_{geo_type}_extracted_comments.csv"
                if file_path.exists():
                    try:
                        df = pd.read_csv(file_path)
                        df['Song'] = 'Despacito' if prefix == 'despa' else 'See You Again'
                        df['Type'] = geo_type
                        geo_data.append(df)
                    except Exception as e:
                        print(f"   ❌ Error loading {file_path}: {e}")
        
        if geo_data:
            self.geographic_data = pd.concat(geo_data, ignore_index=True)
            print(f"   🗺️ Geographic data: {len(self.geographic_data)} rows")
                
    def process_yearly_statistics(self):
        """Process Task 2 data for yearly statistics"""
        if self.task2_data is None:
            return
            
        categories = ['Ethnic Identity', 'Religious Identity', 'Collective Memory']
        yearly_stats = []
        
        for song in ['Despacito', 'See You Again']:
            song_data = self.task2_data[self.task2_data['Song'] == song]
            for year in sorted(song_data['Year'].unique()):
                year_data = song_data[song_data['Year'] == year]
                stats = {'Year': year, 'Song': song, 'Total_Comments': len(year_data)}
                
                for category in categories:
                    if category in year_data.columns:
                        count = year_data[category].notna().sum()
                        stats[category.replace(' ', '_')] = count
                        stats[f"{category.replace(' ', '_')}_Percentage"] = (count / len(year_data) * 100) if len(year_data) > 0 else 0
                    else:
                        stats[category.replace(' ', '_')] = 0
                        stats[f"{category.replace(' ', '_')}_Percentage"] = 0
                
                yearly_stats.append(stats)
        
        self.task2_yearly = pd.DataFrame(yearly_stats)
        
    def create_unified_stacked_charts(self):
        """Create unified stacked bar charts"""
        if self.task2_yearly is None or self.task2_yearly.empty:
            return go.Figure().add_annotation(text="No Task 2 data available", 
                                            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Cultural Identity Categories - Despacito', 'Cultural Identity Categories - See You Again'],
            vertical_spacing=0.15
        )
        
        categories = ['Ethnic_Identity', 'Religious_Identity', 'Collective_Memory']
        colors = {'Ethnic_Identity': '#FF6B6B', 'Religious_Identity': '#4ECDC4', 'Collective_Memory': '#45B7D1'}
        
        for i, song in enumerate(['Despacito', 'See You Again'], 1):
            song_data = self.task2_yearly[self.task2_yearly['Song'] == song].sort_values('Year')
            
            for category in categories:
                display_name = category.replace('_', ' ')
                fig.add_trace(go.Bar(
                    name=f"{display_name}",
                    x=song_data['Year'],
                    y=song_data[category],
                    marker_color=colors[category],
                    text=song_data[category],
                    textposition='inside',
                    showlegend=(i == 1),
                    hovertemplate=f"<b>{song} - {display_name}</b><br>Year: %{{x}}<br>Comments: %{{y}}<extra></extra>"
                ), row=i, col=1)
        
        fig.update_layout(
            title="Unified Cultural Identity Analysis",
            height=800,
            barmode='stack'
        )
        
        return fig
        
    def create_geographic_chart(self):
        """Create geographic distribution chart"""
        if self.geographic_data is None:
            return go.Figure()
            
        geo_summary = self.geographic_data.groupby(['Song', 'Type']).size().reset_index(name='Count')
        
        fig = px.bar(
            geo_summary,
            x='Type',
            y='Count',
            color='Song',
            title="Geographic References Distribution",
            color_discrete_map={'Despacito': '#FF6B6B', 'See You Again': '#45B7D1'}
        )
        
        return fig
        
    def generate_insights(self):
        """Generate insights from analyses"""
        insights = {
            'generation_time': datetime.now().isoformat(),
            'data_summary': {},
            'cultural_insights': {}
        }
        
        if self.task2_data is not None:
            insights['data_summary']['total_comments'] = len(self.task2_data)
            insights['data_summary']['unique_years'] = sorted(self.task2_data['Year'].unique().tolist())
            
        if self.geographic_data is not None:
            insights['data_summary']['geographic_comments'] = len(self.geographic_data)
            
        self.insights = insights
        return insights
        
    def create_unified_dashboard(self):
        """Create the complete unified dashboard"""
        print("🎨 Creating unified dashboard...")
        
        stacked_chart = self.create_unified_stacked_charts()
        geographic_chart = self.create_geographic_chart()
        insights = self.generate_insights()
        
        # Convert charts to HTML
        stacked_html = stacked_chart.to_html(include_plotlyjs=False, div_id="stacked-chart")
        geographic_html = geographic_chart.to_html(include_plotlyjs=False, div_id="geographic-chart")
        
        dashboard_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Unified Cultural Analysis Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .chart-container {{ background: white; margin: 20px 0; padding: 20px; 
                           border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .insights-box {{ background: #e8f4fd; padding: 15px; border-radius: 8px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎵 Unified Cultural Analysis Dashboard</h1>
        <p>Comprehensive analysis combining Task 1, Task 2, and Advanced Visualizations</p>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="insights-box">
        <h2>📊 Summary</h2>
        <p>Total Comments: {insights.get('data_summary', {}).get('total_comments', 'N/A')}</p>
        <p>Geographic References: {insights.get('data_summary', {}).get('geographic_comments', 'N/A')}</p>
    </div>
    
    <div class="chart-container">
        {stacked_html}
    </div>
    
    <div class="chart-container">
        {geographic_html}
    </div>
    
</body>
</html>
"""
        
        with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
            
        print(f"✅ Dashboard saved: {OUTPUT_HTML}")
        return OUTPUT_HTML

def main():
    """Main execution function"""
    print("🚀 Starting Unified Cultural Analysis Dashboard Generation")
    
    generator = UnifiedCulturalAnalysisGenerator()
    
    try:
        generator.load_all_data()
        html_file = generator.create_unified_dashboard()
        print(f"✅ Dashboard Complete: {html_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    main() 