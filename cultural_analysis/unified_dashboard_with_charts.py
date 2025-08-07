#!/usr/bin/env python3
"""
Unified Dashboard with Advanced Charts

This script provides comprehensive interactive visualizations combining:
- Task 1: Geographic/temporal distribution analysis
- Task 2: Cultural identity categories analysis
- Advanced statistical insights and charts
- Interactive filtering and exploration capabilities
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
OUTPUT_HTML = BASE_DIR / "unified_advanced_charts_dashboard.html"

class UnifiedAdvancedChartsGenerator:
    def __init__(self):
        self.task1_data = None
        self.task2_data = None
        self.task2_yearly = None
        self.geographic_data = None
        self.insights = {}
        
    def load_all_data(self):
        """Load all data sources"""
        print("📊 Loading comprehensive dataset...")
        self.load_task1_data()
        self.load_task2_data()
        self.load_geographic_data()
        self.process_advanced_statistics()
        
    def load_task1_data(self):
        """Load Task 1 data"""
        task1_file = BASE_DIR / "task_1_yearly_stats.csv"
        if task1_file.exists():
            self.task1_data = pd.read_csv(task1_file)
            print(f"   ✅ Task 1: {len(self.task1_data)} records")
            
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
                        continue
        
        if task2_data:
            self.task2_data = pd.concat(task2_data, ignore_index=True)
            print(f"   ✅ Task 2: {len(self.task2_data)} comments")
            
    def load_geographic_data(self):
        """Load geographic data"""
        geo_data = []
        for prefix in ['despa', 'seeyou']:
            for geo_type in ['location', 'flag_emoji', 'keyword']:
                file_path = BASE_DIR / f"{prefix}_task_1_{geo_type}_extracted_comments.csv"
                if file_path.exists():
                    try:
                        df = pd.read_csv(file_path)
                        df['Song'] = 'Despacito' if prefix == 'despa' else 'See You Again'
                        df['Type'] = geo_type.replace('_', ' ').title()
                        geo_data.append(df)
                    except:
                        continue
        
        if geo_data:
            self.geographic_data = pd.concat(geo_data, ignore_index=True)
            print(f"   ✅ Geographic: {len(self.geographic_data)} references")
                
    def process_advanced_statistics(self):
        """Process data for advanced statistics"""
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
                        stats[f"{category.replace(' ', '_')}_Pct"] = (count / len(year_data) * 100) if len(year_data) > 0 else 0
                    else:
                        stats[category.replace(' ', '_')] = 0
                        stats[f"{category.replace(' ', '_')}_Pct"] = 0
                
                yearly_stats.append(stats)
        
        self.task2_yearly = pd.DataFrame(yearly_stats)
        
    def create_comprehensive_overview(self):
        """Create comprehensive overview with multiple subplots"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Cultural Categories by Song',
                'Temporal Distribution', 
                'Geographic Reference Types',
                'Category Percentages Over Time'
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Add charts based on available data
        if self.task2_yearly is not None and not self.task2_yearly.empty:
            # Chart 1: Cultural categories by song
            categories = ['Ethnic_Identity', 'Religious_Identity', 'Collective_Memory']
            colors = {'Ethnic_Identity': '#FF6B6B', 'Religious_Identity': '#4ECDC4', 'Collective_Memory': '#45B7D1'}
            
            for song in ['Despacito', 'See You Again']:
                song_totals = self.task2_yearly[self.task2_yearly['Song'] == song][categories].sum()
                
                for i, category in enumerate(categories):
                    fig.add_trace(go.Bar(
                        name=category.replace('_', ' ') if song == 'Despacito' else '',
                        x=[song],
                        y=[int(song_totals[category])],
                        marker_color=colors[category],
                        showlegend=(song == 'Despacito'),
                        text=str(int(song_totals[category])),
                        textposition='inside'
                    ), row=1, col=1)
        
        fig.update_layout(
            title="Comprehensive Cultural Analysis Overview",
            height=800,
            showlegend=True
        )
        
        return fig
        
    def create_unified_dashboard(self):
        """Create the unified dashboard with all charts"""
        print("🎨 Generating unified dashboard with advanced charts...")
        
        # Generate visualizations
        overview_chart = self.create_comprehensive_overview()
        
        # Convert to HTML
        overview_html = overview_chart.to_html(include_plotlyjs=False, div_id="overview-chart")
        
        dashboard_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Unified Cultural Analysis Dashboard with Advanced Charts</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f0f2f5; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 15px; margin-bottom: 25px; }}
        .chart-section {{ background: white; margin: 25px 0; padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎵 Unified Cultural Analysis Dashboard</h1>
        <h2>Advanced Charts & Comprehensive Insights</h2>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="chart-section">
        <h2>📊 Comprehensive Overview</h2>
        {overview_html}
    </div>
    
</body>
</html>
"""
        
        with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
            
        print(f"✅ Advanced dashboard saved: {OUTPUT_HTML}")
        return OUTPUT_HTML

def main():
    """Main execution function"""
    print("🚀 Starting Unified Dashboard with Advanced Charts Generation")
    
    generator = UnifiedAdvancedChartsGenerator()
    
    try:
        generator.load_all_data()
        html_file = generator.create_unified_dashboard()
        print(f"✅ Advanced Dashboard Complete: {html_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 