#!/usr/bin/env python3
"""
Combined Interactive Visualization for Task 1 and Task 2 Cultural Analysis

This script creates interactive visualizations that combine:
1. Task 1: Geographic/temporal distribution data
2. Task 2: Cultural identity categories (Ethnic Identity, Religious Identity, Collective Memory)

Features:
- Stacked bar charts for yearly comment counts by cultural categories
- Interactive switching between datasets
- Comparative view between Despacito and See You Again
"""

import os
import sys
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import glob
from pathlib import Path

# Constants
BASE_DIR = Path(__file__).parent
OUTPUT_HTML = BASE_DIR / "combined_task_1_2_interactive_dashboard.html"

class CulturalAnalysisVisualizer:
    def __init__(self):
        self.task1_data = None
        self.task2_data = None
        
    def load_task1_data(self):
        """Load Task 1 yearly statistics data"""
        task1_file = BASE_DIR / "task_1_yearly_stats.csv"
        if task1_file.exists():
            self.task1_data = pd.read_csv(task1_file)
            print(f"Loaded Task 1 data: {len(self.task1_data)} rows")
        else:
            print(f"Warning: Task 1 file not found at {task1_file}")
            
    def load_task2_data(self):
        """Load Task 2 yearly cultural identity data"""
        task2_data = []
        
        # Load data for both songs
        for song in ['despa', 'seeyou']:
            yearly_dir = BASE_DIR / f"{song}_outputs" / f"{song}_task_2_yearly_split"
            
            if yearly_dir.exists():
                csv_files = list(yearly_dir.glob(f"{song}_*_labeled_context_comments.csv"))
                print(f"Found {len(csv_files)} yearly files for {song}")
                
                for csv_file in csv_files:
                    try:
                        # Extract year from filename
                        year = int(csv_file.stem.split('_')[1])
                        
                        # Load CSV
                        df = pd.read_csv(csv_file)
                        
                        # Add metadata
                        df['Song'] = 'Despacito' if song == 'despa' else 'See You Again'
                        df['Year'] = year
                        
                        task2_data.append(df)
                        print(f"Loaded {len(df)} comments for {song} {year}")
                        
                    except Exception as e:
                        print(f"Error loading {csv_file}: {e}")
            else:
                print(f"Warning: Directory not found: {yearly_dir}")
        
        if task2_data:
            self.task2_data = pd.concat(task2_data, ignore_index=True)
            print(f"Total Task 2 data loaded: {len(self.task2_data)} rows")
        else:
            print("Warning: No Task 2 data loaded")
            
    def process_task2_yearly_stats(self):
        """Process Task 2 data to create yearly statistics by cultural categories"""
        if self.task2_data is None:
            return None
            
        # Define cultural categories
        categories = ['Ethnic Identity', 'Religious Identity', 'Collective Memory']
        
        yearly_stats = []
        
        for song in ['Despacito', 'See You Again']:
            song_data = self.task2_data[self.task2_data['Song'] == song]
            
            for year in sorted(song_data['Year'].unique()):
                year_data = song_data[song_data['Year'] == year]
                
                stats = {
                    'Year': year,
                    'Song': song,
                    'Total_Comments': len(year_data)
                }
                
                # Count non-empty values for each category
                for category in categories:
                    if category in year_data.columns:
                        count = year_data[category].notna().sum()
                        stats[category.replace(' ', '_')] = count
                    else:
                        stats[category.replace(' ', '_')] = 0
                
                yearly_stats.append(stats)
        
        return pd.DataFrame(yearly_stats)
    
    def create_stacked_bar_chart(self, data, title_suffix=""):
        """Create stacked bar chart for cultural identity categories"""
        if data is None or data.empty:
            return go.Figure().add_annotation(text="No data available", 
                                            xref="paper", yref="paper",
                                            x=0.5, y=0.5, showarrow=False)
        
        # Prepare data for stacked bar chart
        categories = ['Ethnic_Identity', 'Religious_Identity', 'Collective_Memory']
        colors = {
            'Ethnic_Identity': '#FF6B6B',     # Red
            'Religious_Identity': '#4ECDC4',   # Teal  
            'Collective_Memory': '#45B7D1'     # Blue
        }
        
        fig = go.Figure()
        
        for song in data['Song'].unique():
            song_data = data[data['Song'] == song].sort_values('Year')
            
            # Add traces for each category
            for category in categories:
                display_name = category.replace('_', ' ')
                
                fig.add_trace(go.Bar(
                    name=f"{song} - {display_name}",
                    x=song_data['Year'],
                    y=song_data[category],
                    marker_color=colors[category],
                    opacity=0.8 if song == 'Despacito' else 0.6,
                    offsetgroup=song,
                    legendgroup=f"{song}_{category}",
                    text=song_data[category],
                    textposition='inside',
                    textfont=dict(color='white', size=10),
                    hovertemplate=f"<b>{song} - {display_name}</b><br>" +
                                "Year: %{x}<br>" +
                                "Comments: %{y}<br>" +
                                "<extra></extra>"
                ))
        
        fig.update_layout(
            title=f"Cultural Identity Categories by Year{title_suffix}",
            xaxis_title="Year",
            yaxis_title="Number of Comments",
            barmode='stack',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=600,
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        return fig
    
    def create_task1_comparison_chart(self):
        """Create comparison chart for Task 1 geographic data"""
        if self.task1_data is None:
            return go.Figure().add_annotation(text="Task 1 data not available", 
                                            xref="paper", yref="paper",
                                            x=0.5, y=0.5, showarrow=False)
        
        fig = go.Figure()
        
        # Group by Video and Year
        for video in self.task1_data['Video'].unique():
            video_data = self.task1_data[self.task1_data['Video'] == video].sort_values('Year')
            video_name = 'Despacito' if video == 'Despa' else 'See You Again'
            
            fig.add_trace(go.Scatter(
                x=video_data['Year'],
                y=video_data['Total Comments'],
                mode='lines+markers',
                name=video_name,
                line=dict(width=3),
                marker=dict(size=8),
                hovertemplate=f"<b>{video_name}</b><br>" +
                            "Year: %{x}<br>" +
                            "Total Comments: %{y:,}<br>" +
                            "Unique Countries: %{customdata[0]}<br>" +
                            "Top Country: %{customdata[1]} (%{customdata[2]:,})<br>" +
                            "<extra></extra>",
                customdata=list(zip(
                    video_data['Unique Countries'],
                    video_data['Top 1 Country'],
                    video_data['Top 1 Count']
                ))
            ))
        
        fig.update_layout(
            title="Task 1: Geographic Comment Distribution Over Time",
            xaxis_title="Year",
            yaxis_title="Total Comments with Geographic Information",
            height=500,
            font=dict(size=12),
            legend=dict(x=0.02, y=0.98),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        return fig
    
    def create_category_breakdown_chart(self, data):
        """Create detailed breakdown of cultural categories"""
        if data is None or data.empty:
            return go.Figure().add_annotation(text="No data available", 
                                            xref="paper", yref="paper",
                                            x=0.5, y=0.5, showarrow=False)
        
        # Calculate percentage distribution
        categories = ['Ethnic_Identity', 'Religious_Identity', 'Collective_Memory']
        
        fig_data = []
        for song in data['Song'].unique():
            song_data = data[data['Song'] == song]
            
            totals = {}
            for category in categories:
                totals[category] = song_data[category].sum()
            
            total_all = sum(totals.values())
            
            if total_all > 0:
                fig_data.append({
                    'Song': song,
                    'Categories': [cat.replace('_', ' ') for cat in categories],
                    'Values': [totals[cat] for cat in categories],
                    'Percentages': [round(totals[cat]/total_all*100, 1) for cat in categories]
                })
        
        # Create subplots for pie charts
        fig = make_subplots(
            rows=1, cols=len(fig_data),
            specs=[[{'type': 'domain'}] * len(fig_data)],
            subplot_titles=[item['Song'] for item in fig_data]
        )
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, song_data in enumerate(fig_data):
            fig.add_trace(go.Pie(
                labels=song_data['Categories'],
                values=song_data['Values'],
                name=song_data['Song'],
                marker_colors=colors,
                textinfo='label+percent',
                hovertemplate="<b>%{label}</b><br>" +
                            "Comments: %{value}<br>" +
                            "Percentage: %{percent}<br>" +
                            "<extra></extra>"
            ), row=1, col=i+1)
        
        fig.update_layout(
            title="Cultural Identity Categories Distribution",
            height=400,
            font=dict(size=12)
        )
        
        return fig
    
    def create_combined_dashboard(self):
        """Create the main interactive dashboard"""
        print("Creating combined dashboard...")
        
        # Load data
        self.load_task1_data()
        self.load_task2_data()
        
        # Process Task 2 data
        task2_yearly = self.process_task2_yearly_stats()
        
        # Create individual charts
        task1_chart = self.create_task1_comparison_chart()
        task2_stacked = self.create_stacked_bar_chart(task2_yearly)
        category_breakdown = self.create_category_breakdown_chart(task2_yearly)
        
        # Calculate statistics
        if task2_yearly is not None:
            total_task2_comments = task2_yearly[['Ethnic_Identity', 'Religious_Identity', 'Collective_Memory']].sum().sum()
            years_covered = len(task2_yearly['Year'].unique())
        else:
            total_task2_comments = 0
            years_covered = 0
            
        if self.task1_data is not None:
            total_task1_comments = self.task1_data['Total Comments'].sum()
            unique_countries = self.task1_data['Unique Countries'].max()
        else:
            total_task1_comments = 0
            unique_countries = 0
        
        # Create HTML with multiple charts
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Cultural Analysis Dashboard - Tasks 1 & 2 Combined</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .dashboard-header {{
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .chart-container {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .chart-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .stat-number {{
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }}
        .info-panel {{
            background: #e8f4f8;
            border-left: 4px solid #4ECDC4;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1>Cultural Analysis Dashboard</h1>
        <p>Interactive Visualization of Task 1 (Geographic) and Task 2 (Cultural Identity) Analysis</p>
        <p><em>Comparing "Despacito" vs "See You Again" YouTube Comments</em></p>
    </div>
    
    <div class="info-panel">
        <h3>📊 Dashboard Overview</h3>
        <p><strong>Task 1:</strong> Geographic distribution of comments mentioning countries, cities, or containing flag emojis</p>
        <p><strong>Task 2:</strong> Cultural identity analysis focusing on three categories:</p>
        <ul style="margin: 10px 0; padding-left: 30px;">
            <li><strong>Ethnic Identity:</strong> latino, hispanic, asian, white, caucasian, black</li>
            <li><strong>Religious Identity:</strong> muslim, islam, christian, christ</li>
            <li><strong>Collective Memory:</strong> my country, our country, in my culture, homeland, diaspora, immigrant</li>
        </ul>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-number">{total_task1_comments:,}</div>
            <div class="stat-label">Task 1 Comments<br>Geographic Data</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{total_task2_comments:,}</div>
            <div class="stat-label">Task 2 Comments<br>Cultural Identity</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{unique_countries}</div>
            <div class="stat-label">Unique Countries<br>Identified</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{years_covered}</div>
            <div class="stat-label">Years of Data<br>Available</div>
        </div>
    </div>
    
    <div class="chart-container">
        <div class="chart-title">Task 1: Geographic Comment Distribution Timeline</div>
        <div id="task1-chart">{task1_chart.to_html(include_plotlyjs=False, div_id="task1-chart")}</div>
    </div>
    
    <div class="chart-container">
        <div class="chart-title">Task 2: Cultural Identity Categories - Yearly Stacked View</div>
        <div id="task2-stacked">{task2_stacked.to_html(include_plotlyjs=False, div_id="task2-stacked")}</div>
    </div>
    
    <div class="chart-container">
        <div class="chart-title">Task 2: Cultural Categories Distribution Overview</div>
        <div id="category-breakdown">{category_breakdown.to_html(include_plotlyjs=False, div_id="category-breakdown")}</div>
    </div>
    
    <div class="info-panel">
        <h3>🔍 Analysis Notes</h3>
        <p><strong>Data Sources:</strong></p>
        <ul style="margin: 10px 0; padding-left: 30px;">
            <li>Task 1 data from combined geographic extraction (keywords, flag emojis, location mentions)</li>
            <li>Task 2 data from cultural keyword filtering and context analysis</li>
        </ul>
        <p><strong>Methodology:</strong> Comments are processed yearly and categorized based on keyword matching. Multiple categories can be present in a single comment.</p>
    </div>
    
</body>
</html>"""
        
        # Save HTML file
        with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Dashboard saved to: {OUTPUT_HTML}")
        return str(OUTPUT_HTML)
    
    def generate_summary_statistics(self):
        """Generate and save summary statistics"""
        print("Generating summary statistics...")
        
        # Load data if not already loaded
        if self.task2_data is None:
            self.load_task2_data()
        if self.task1_data is None:
            self.load_task1_data()
        
        summary_stats = {}
        
        # Task 1 summary
        if self.task1_data is not None:
            summary_stats['Task1'] = {
                'total_geographic_comments': int(self.task1_data['Total Comments'].sum()),
                'unique_countries_max': int(self.task1_data['Unique Countries'].max()),
                'years_covered': sorted(list(self.task1_data['Year'].unique())),
                'videos_analyzed': list(self.task1_data['Video'].unique())
            }
        
        # Task 2 summary
        if self.task2_data is not None:
            summary_stats['Task2'] = {
                'total_cultural_comments': int(self.task2_data.shape[0]),
                'ethnic_identity_total': int(self.task2_data['Ethnic Identity'].notna().sum()),
                'religious_identity_total': int(self.task2_data['Religious Identity'].notna().sum()),
                'collective_memory_total': int(self.task2_data['Collective Memory'].notna().sum()),
                'years_covered': sorted(list(self.task2_data['Year'].unique())),
                'songs_analyzed': list(self.task2_data['Song'].unique())
            }
        
        # Save summary statistics
        summary_file = BASE_DIR / "combined_analysis_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Summary statistics saved to: {summary_file}")
        
        return summary_stats

def main():
    """Main execution function"""
    print("🎵 Starting Combined Cultural Analysis Visualization")
    print("=" * 60)
    
    # Create visualizer instance
    visualizer = CulturalAnalysisVisualizer()
    
    try:
        # Generate dashboard
        dashboard_path = visualizer.create_combined_dashboard()
        
        # Generate summary statistics
        stats = visualizer.generate_summary_statistics()
        
        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETE!")
        print(f"📊 Interactive Dashboard: {dashboard_path}")
        print(f"📈 Summary Statistics: {BASE_DIR}/combined_analysis_summary.json")
        
        # Print quick stats
        if 'Task1' in stats:
            print(f"\n📍 Task 1 - Geographic Analysis:")
            print(f"   Total comments: {stats['Task1']['total_geographic_comments']:,}")
            print(f"   Countries identified: {stats['Task1']['unique_countries_max']}")
        
        if 'Task2' in stats:
            print(f"\n🏛️ Task 2 - Cultural Identity Analysis:")
            print(f"   Total comments: {stats['Task2']['total_cultural_comments']:,}")
            print(f"   Ethnic identity: {stats['Task2']['ethnic_identity_total']:,}")
            print(f"   Religious identity: {stats['Task2']['religious_identity_total']:,}")
            print(f"   Collective memory: {stats['Task2']['collective_memory_total']:,}")
        
        print(f"\n🌟 Open the dashboard file in your browser to explore the interactive visualizations!")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        raise

if __name__ == "__main__":
    main() 