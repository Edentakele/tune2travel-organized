# Unified Cultural Analysis Dashboard System

This comprehensive dashboard system combines all cultural analysis tasks into a single powerful visualization and analysis platform.

## 🎯 Overview

The **Unified Cultural Analysis Dashboard** integrates:

- **Task 1**: Geographic and temporal distribution analysis
- **Task 2**: Cultural identity categories (Ethnic Identity, Religious Identity, Collective Memory)
- **Advanced Visualizations**: Interactive charts, statistical insights, and comparative analysis
- **Data Quality Analysis**: Spam filtering and data validation
- **Export Capabilities**: HTML dashboards, JSON insights, CSV exports

## 📁 File Structure

### Core Dashboard Files

- `unified_dashboard_generator.py` - Main dashboard generator with comprehensive functionality
- `unified_dashboard_with_charts.py` - Advanced charts and detailed visualizations
- `COMBINED_ANALYSIS_README.md` - This documentation file

### Generated Outputs

- `unified_cultural_analysis_dashboard.html` - Main dashboard output
- `unified_advanced_charts_dashboard.html` - Advanced charts dashboard
- `unified_analysis_insights.json` - Machine-readable insights data

## 🚀 Quick Start

### 1. Generate Main Dashboard

```bash
python unified_dashboard_generator.py
```

### 2. Generate Advanced Charts Dashboard

```bash
python unified_dashboard_with_charts.py
```

### 3. View Results

Open the generated HTML files in your web browser:
- `unified_cultural_analysis_dashboard.html` - Main dashboard
- `unified_advanced_charts_dashboard.html` - Advanced charts

## 📊 Dashboard Features

### Main Dashboard (`unified_dashboard_generator.py`)

**Core Visualizations:**
- Unified stacked bar charts for cultural identity categories
- Geographic distribution analysis
- Temporal trend analysis
- Summary statistics and insights

**Key Metrics:**
- Total comments analyzed across all tasks
- Geographic references by type (location, flag emoji, keyword)
- Cultural category distribution by song and year
- Peak activity periods and dominant categories

### Advanced Charts Dashboard (`unified_dashboard_with_charts.py`)

**Advanced Visualizations:**
- Comprehensive overview with 4-panel layout
- Detailed cultural identity analysis (absolute counts + percentages)
- Geographic insights with pie charts and comparisons
- Interactive filtering and exploration

**Statistical Analysis:**
- Percentage distributions over time
- Song-by-song comparisons
- Reference type breakdowns
- Temporal trend analysis

## 🎵 Data Sources

### Task 1 Data (Geographic Analysis)
- `task_1_yearly_stats.csv` - Yearly statistics summary
- `{prefix}_task_1_location_extracted_comments.csv` - Location mentions
- `{prefix}_task_1_flag_emoji_extracted_comments.csv` - Flag emoji references
- `{prefix}_task_1_keyword_extracted_comments.csv` - Keyword-based extraction

### Task 2 Data (Cultural Identity)
- `{song}_outputs/{song}_task_2_yearly_split/` - Yearly split data
- `{song}_{year}_labeled_context_comments.csv` - Cultural identity annotations

**Where `{prefix}` = `despa` (Despacito) or `seeyou` (See You Again)**

## 📈 Generated Insights

### Data Summary
- Total comments processed
- Unique years covered
- Songs analyzed
- Geographic references found

### Cultural Analysis
- Distribution of cultural identity categories
- Dominant categories by song
- Temporal evolution of cultural references

### Geographic Analysis
- Reference types distribution
- Song-specific geographic patterns
- Temporal geographic trends

### Quality Metrics
- Spam distribution analysis
- Data completeness statistics
- Validation results

## 🔧 Technical Requirements

### Dependencies
```bash
pip install pandas plotly numpy pathlib datetime json
```

### System Requirements
- Python 3.7+
- 8GB RAM (recommended for large datasets)
- Modern web browser for viewing dashboards

## 📋 Usage Examples

### Basic Dashboard Generation

```python
from unified_dashboard_generator import UnifiedCulturalAnalysisGenerator

# Initialize generator
generator = UnifiedCulturalAnalysisGenerator()

# Load all data
generator.load_all_data()

# Create dashboard
html_file, json_file = generator.create_unified_dashboard()
```

### Advanced Charts Generation

```python
from unified_dashboard_with_charts import UnifiedAdvancedChartsGenerator

# Initialize advanced generator
generator = UnifiedAdvancedChartsGenerator()

# Load data and create dashboard
generator.load_all_data()
html_file = generator.create_unified_dashboard()
```

### Accessing Insights Programmatically

```python
# Load insights from JSON
import json
with open('unified_analysis_insights.json', 'r') as f:
    insights = json.load(f)

# Access specific metrics
total_comments = insights['data_summary']['total_comments_task2']
peak_year = insights['temporal_insights']['peak_year']
```

## 🎨 Customization

### Adding New Visualizations

1. **Create new chart method** in the generator class:
```python
def create_custom_chart(self):
    # Your custom visualization logic
    fig = go.Figure()
    # ... add traces and layout
    return fig
```

2. **Integrate into dashboard** in `create_unified_dashboard()`:
```python
custom_chart = self.create_custom_chart()
custom_html = custom_chart.to_html(include_plotlyjs=False, div_id="custom-chart")
```

3. **Add to HTML template** in the dashboard HTML string.

### Modifying Styling

Update the CSS styles in the dashboard HTML template:
- `.header` - Dashboard header styling
- `.chart-container` - Individual chart containers
- `.insights-box` - Summary statistics boxes

## 🔍 Troubleshooting

### Common Issues

**"No data available" messages:**
- Ensure all required CSV files are present in the correct directories
- Check file naming conventions match expected patterns
- Verify data files contain expected columns

**Dashboard not loading:**
- Check browser console for JavaScript errors
- Ensure Plotly CDN is accessible
- Verify HTML file was generated without errors

**Missing visualizations:**
- Check console output for data loading errors
- Verify all input files have proper formatting
- Ensure required columns exist in source data

### Debug Mode

Enable verbose logging by adding debug prints:
```python
generator.load_all_data()
print(f"Task 1 data: {generator.task1_data is not None}")
print(f"Task 2 data: {generator.task2_data is not None}")
print(f"Geographic data: {generator.geographic_data is not None}")
```

## 📧 Output Files Description

### HTML Dashboards
- **Interactive visualizations** with hover effects and zoom capabilities
- **Responsive design** that adapts to different screen sizes
- **Export functionality** for charts (PNG, PDF, SVG)

### JSON Insights
- **Machine-readable format** for programmatic access
- **Timestamped data** for tracking analysis runs
- **Structured metrics** for integration with other tools

## 🚀 Future Enhancements

### Planned Features
- Real-time data updates
- Advanced filtering controls
- Export to additional formats (Excel, PowerPoint)
- Integration with external APIs
- Automated report generation

### Extension Points
- Plugin system for custom analyzers
- REST API for remote access
- Database integration for persistent storage
- Machine learning insights integration

## 📄 License & Credits

This dashboard system is part of the tune2travel cultural analysis project.

**Key Technologies:**
- **Plotly** - Interactive visualizations
- **Pandas** - Data processing and analysis
- **Python** - Core development platform

---

*For additional support or questions, refer to the main project documentation or contact the development team.* 