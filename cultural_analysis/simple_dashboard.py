#!/usr/bin/env python3
"""
Simple Cultural Analysis Dashboard Generator
Creates individual chart files and a main dashboard
"""

from advanced_cultural_analysis_dashboard import AdvancedCulturalAnalysisVisualizer
import plotly.offline as pyo
from pathlib import Path
from datetime import datetime

def create_simple_dashboard():
    """Create a simple dashboard with individual chart files"""
    print("🎵 Creating Simple Cultural Analysis Dashboard...")
    
    # Initialize visualizer
    visualizer = AdvancedCulturalAnalysisVisualizer()
    visualizer.load_all_data()
    
    # Create individual charts
    print("📊 Generating individual charts...")
    
    # Generate charts
    task1_chart = visualizer.create_task1_enhanced_chart()
    stacked_chart = visualizer.create_enhanced_stacked_bar_chart()
    temporal_chart = visualizer.create_temporal_trend_analysis()
    comparative_chart = visualizer.create_comparative_metrics_chart()
    
    # Save individual chart files
    chart_files = {}
    
    chart_files['task1'] = 'task1_geographic_analysis.html'
    pyo.plot(task1_chart, filename=chart_files['task1'], auto_open=False)
    print(f"✅ Task 1 chart saved: {chart_files['task1']}")
    
    chart_files['stacked'] = 'task2_stacked_cultural_categories.html'
    pyo.plot(stacked_chart, filename=chart_files['stacked'], auto_open=False)
    print(f"✅ Stacked chart saved: {chart_files['stacked']}")
    
    chart_files['temporal'] = 'temporal_trends_analysis.html'
    pyo.plot(temporal_chart, filename=chart_files['temporal'], auto_open=False)
    print(f"✅ Temporal chart saved: {chart_files['temporal']}")
    
    chart_files['comparative'] = 'comparative_metrics_analysis.html'
    pyo.plot(comparative_chart, filename=chart_files['comparative'], auto_open=False)
    print(f"✅ Comparative chart saved: {chart_files['comparative']}")
    
    # Create main dashboard HTML
    insights = visualizer.generate_insights_summary()
    
    # Calculate statistics
    total_geographic = insights.get('task1_insights', {}).get('total_geographic_comments', 0)
    total_cultural = insights.get('task2_insights', {}).get('total_cultural_comments', 0)
    max_countries = insights.get('task1_insights', {}).get('max_countries', 0)
    years_coverage = len(insights.get('task2_insights', {}).get('yearly_coverage', []))
    
    # Get cultural category totals
    category_totals = insights.get('task2_insights', {}).get('category_totals', {})
    ethnic_total = category_totals.get('ethnic_identity', 0)
    religious_total = category_totals.get('religious_identity', 0)
    collective_total = category_totals.get('collective_memory', 0)
    
    # Get dominant categories
    despacito_dominant = insights.get('task2_insights', {}).get('dominant_category_despa', {})
    seeyou_dominant = insights.get('task2_insights', {}).get('dominant_category_seeyou', {})
    
    dashboard_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Simple Cultural Analysis Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }}
        .dashboard-header {{
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }}
        .chart-container {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border: 1px solid rgba(255,255,255,0.2);
        }}
        .chart-link {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border-radius: 25px;
            text-decoration: none;
            font-weight: 600;
            transition: transform 0.2s ease;
            margin: 10px;
        }}
        .chart-link:hover {{
            transform: translateY(-2px);
            color: white;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border: 1px solid rgba(255,255,255,0.2);
            transition: transform 0.2s ease;
        }}
        .stat-card:hover {{
            transform: translateY(-2px);
        }}
        .stat-number {{
            font-size: 36px;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .stat-label {{
            font-size: 14px;
            color: #718096;
            margin-top: 8px;
            font-weight: 500;
        }}
        .insights-panel {{
            background: linear-gradient(135deg, #fef5e7 0%, #fff5e6 100%);
            border-left: 5px solid #ed8936;
            padding: 20px;
            margin: 25px 0;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        .insights-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }}
        .insight-item {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }}
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1>🎵 Simple Cultural Analysis Dashboard</h1>
        <div>Comprehensive Analysis of Cultural Identity in YouTube Comments</div>
        <div><em>"Despacito" vs "See You Again" - Cross-Cultural Comparative Study</em></div>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-number">{total_geographic:,}</div>
            <div class="stat-label">Geographic Comments<br>Task 1 Analysis</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{total_cultural:,}</div>
            <div class="stat-label">Cultural Identity Mentions<br>Task 2 Analysis</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{max_countries}</div>
            <div class="stat-label">Unique Countries<br>Identified</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{years_coverage}</div>
            <div class="stat-label">Years Analyzed<br>Temporal Coverage</div>
        </div>
    </div>
    
    <div class="chart-container">
        <h2>📊 Interactive Visualizations</h2>
        <p>Click on the links below to view detailed interactive charts:</p>
        
        <div style="text-align: center; margin: 20px 0;">
            <a href="{chart_files['task1']}" class="chart-link" target="_blank">
                📈 Task 1: Geographic Analysis
            </a>
            <a href="{chart_files['stacked']}" class="chart-link" target="_blank">
                📊 Task 2: Cultural Categories (Stacked)
            </a>
            <a href="{chart_files['temporal']}" class="chart-link" target="_blank">
                📉 Temporal Trends Analysis
            </a>
            <a href="{chart_files['comparative']}" class="chart-link" target="_blank">
                ⚖️ Comparative Metrics
            </a>
        </div>
    </div>
    
    <div class="insights-panel">
        <h3>💡 Key Insights</h3>
        <div class="insights-grid">
            <div class="insight-item">
                <strong>🌍 Geographic Engagement:</strong><br>
                Total comments with geographic references: {total_geographic:,}<br>
                Maximum countries in a year: {max_countries}
            </div>
            <div class="insight-item">
                <strong>🏛️ Cultural Categories:</strong><br>
                Ethnic Identity: {ethnic_total:,} mentions<br>
                Religious Identity: {religious_total:,} mentions<br>
                Collective Memory: {collective_total:,} mentions
            </div>
            <div class="insight-item">
                <strong>🎵 Despacito Dominant Category:</strong><br>
                {despacito_dominant.get('category', 'N/A')} ({despacito_dominant.get('total', 0):,} mentions)<br>
                Average: {despacito_dominant.get('average_percentage', 0):.1f}% per year
            </div>
            <div class="insight-item">
                <strong>🎭 See You Again Dominant Category:</strong><br>
                {seeyou_dominant.get('category', 'N/A')} ({seeyou_dominant.get('total', 0):,} mentions)<br>
                Average: {seeyou_dominant.get('average_percentage', 0):.1f}% per year
            </div>
        </div>
    </div>
    
    <div class="insights-panel">
        <h3>🔬 Analysis Notes</h3>
        <ul>
            <li><strong>Task 1:</strong> Geographic distribution analysis using regex patterns for countries, cities, and flag emojis</li>
            <li><strong>Task 2:</strong> Cultural identity keyword matching across three categories: Ethnic, Religious, and Collective Memory</li>
            <li><strong>Data Coverage:</strong> Analysis spans {years_coverage} years with {total_cultural:,} total cultural identity mentions</li>
            <li><strong>Methodology:</strong> Comments can belong to multiple categories; percentages calculated yearly</li>
        </ul>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>"""
    
    # Save main dashboard
    dashboard_file = 'simple_cultural_analysis_dashboard.html'
    with open(dashboard_file, 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    
    print(f"✅ Main dashboard saved: {dashboard_file}")
    print(f"🎉 All charts created successfully!")
    print(f"📁 Files created:")
    print(f"   📊 Main Dashboard: {dashboard_file}")
    for name, file in chart_files.items():
        print(f"   📈 {name.title()} Chart: {file}")
    
    return dashboard_file, chart_files

if __name__ == "__main__":
    create_simple_dashboard() 