#!/usr/bin/env python3
"""
Advanced Cultural Analysis Dashboard - Enhanced Interactive Visualization

This script provides comprehensive analysis combining Task 1 and Task 2 with:
- Stacked bar charts for cultural identity categories
- Temporal trend analysis
- Comparative metrics between songs
- Interactive filtering and exploration
- Statistical summaries and insights
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

# Constants
BASE_DIR = Path(__file__).parent
OUTPUT_HTML = BASE_DIR / "advanced_cultural_analysis_dashboard.html"

class AdvancedCulturalAnalysisVisualizer:
    def __init__(self):
        self.task1_data = None
        self.task2_data = None
        self.task2_yearly = None
        
    def load_all_data(self):
        """Load both Task 1 and Task 2 data"""
        self.load_task1_data()
        self.load_task2_data()
        self.task2_yearly = self.process_task2_yearly_stats()
        
    def load_task1_data(self):
        """Load Task 1 yearly statistics data"""
        task1_file = BASE_DIR / "task_1_yearly_stats.csv"
        if task1_file.exists():
            self.task1_data = pd.read_csv(task1_file)
            print(f"✅ Loaded Task 1 data: {len(self.task1_data)} rows")
        else:
            print(f"⚠️ Warning: Task 1 file not found at {task1_file}")
            
    def load_task2_data(self):
        """Load Task 2 yearly cultural identity data"""
        task2_data = []
        
        # Load data for both songs
        for song in ['despa', 'seeyou']:
            yearly_dir = BASE_DIR / f"{song}_outputs" / f"{song}_task_2_yearly_split"
            
            if yearly_dir.exists():
                csv_files = list(yearly_dir.glob(f"{song}_*_labeled_context_comments.csv"))
                print(f"📁 Found {len(csv_files)} yearly files for {song}")
                
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
                        print(f"   📊 Loaded {len(df)} comments for {song} {year}")
                        
                    except Exception as e:
                        print(f"❌ Error loading {csv_file}: {e}")
            else:
                print(f"⚠️ Warning: Directory not found: {yearly_dir}")
        
        if task2_data:
            self.task2_data = pd.concat(task2_data, ignore_index=True)
            print(f"✅ Total Task 2 data loaded: {len(self.task2_data)} rows")
        else:
            print("⚠️ Warning: No Task 2 data loaded")
            
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
                        # Calculate percentage
                        percentage = (count / len(year_data) * 100) if len(year_data) > 0 else 0
                        stats[f"{category.replace(' ', '_')}_Percentage"] = round(percentage, 1)
                    else:
                        stats[category.replace(' ', '_')] = 0
                        stats[f"{category.replace(' ', '_')}_Percentage"] = 0
                
                yearly_stats.append(stats)
        
        return pd.DataFrame(yearly_stats)
    
    def create_enhanced_stacked_bar_chart(self):
        """Create enhanced stacked bar chart with better interactivity"""
        if self.task2_yearly is None or self.task2_yearly.empty:
            return go.Figure().add_annotation(text="No Task 2 data available", 
                                            xref="paper", yref="paper",
                                            x=0.5, y=0.5, showarrow=False)
        
        # Create subplots for each song
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Despacito - Cultural Identity Categories', 'See You Again - Cultural Identity Categories'],
            vertical_spacing=0.1,
            row_heights=[0.5, 0.5]
        )
        
        categories = ['Ethnic_Identity', 'Religious_Identity', 'Collective_Memory']
        colors = {
            'Ethnic_Identity': '#FF6B6B',     # Red
            'Religious_Identity': '#4ECDC4',   # Teal  
            'Collective_Memory': '#45B7D1'     # Blue
        }
        
        for i, song in enumerate(['Despacito', 'See You Again'], 1):
            song_data = self.task2_yearly[self.task2_yearly['Song'] == song].sort_values('Year')
            
            for category in categories:
                display_name = category.replace('_', ' ')
                
                fig.add_trace(go.Bar(
                    name=f"{display_name}",
                    x=song_data['Year'].tolist(),
                    y=song_data[category].tolist(),
                    marker_color=colors[category],
                    text=song_data[category].tolist(),
                    textposition='inside',
                    textfont=dict(color='white', size=10),
                    hovertemplate=f"<b>{song} - {display_name}</b><br>" +
                                "Year: %{x}<br>" +
                                "Comments: %{y}<br>" +
                                "Percentage: %{customdata}%<br>" +
                                "<extra></extra>",
                    customdata=song_data[f"{category}_Percentage"].tolist(),
                    showlegend=(i == 1)  # Show legend only for first subplot
                ), row=i, col=1)
        
        fig.update_layout(
            title="Task 2: Cultural Identity Categories by Year - Comparative View",
            height=800,
            barmode='stack',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5
            ),
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        return fig
    
    def create_temporal_trend_analysis(self):
        """Create temporal trend analysis chart"""
        if self.task2_yearly is None or self.task2_yearly.empty:
            return go.Figure().add_annotation(text="No data available for temporal analysis", 
                                            xref="paper", yref="paper",
                                            x=0.5, y=0.5, showarrow=False)
        
        fig = go.Figure()
        
        categories = ['Ethnic_Identity', 'Religious_Identity', 'Collective_Memory']
        colors = {'Ethnic_Identity': '#FF6B6B', 'Religious_Identity': '#4ECDC4', 'Collective_Memory': '#45B7D1'}
        
        for song in ['Despacito', 'See You Again']:
            for category in categories:
                song_data = self.task2_yearly[self.task2_yearly['Song'] == song].sort_values('Year')
                display_name = category.replace('_', ' ')
                
                line_style = dict(width=3) if song == 'Despacito' else dict(width=3, dash='dash')
                
                fig.add_trace(go.Scatter(
                    x=song_data['Year'].tolist(),
                    y=song_data[f"{category}_Percentage"].tolist(),
                    mode='lines+markers',
                    name=f"{song} - {display_name}",
                    line=line_style,
                    marker=dict(size=8, color=colors[category]),
                    hovertemplate=f"<b>{song} - {display_name}</b><br>" +
                                "Year: %{x}<br>" +
                                "Percentage: %{y}%<br>" +
                                "Comments: %{customdata}<br>" +
                                "<extra></extra>",
                    customdata=song_data[category].tolist()
                ))
        
        fig.update_layout(
            title="Temporal Trends: Cultural Identity Categories Percentage Over Time",
            xaxis_title="Year",
            yaxis_title="Percentage of Comments (%)",
            height=500,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        return fig
    
    def create_comparative_metrics_chart(self):
        """Create comparative metrics visualization"""
        if self.task2_yearly is None or self.task2_yearly.empty:
            return go.Figure().add_annotation(text="No data available for comparison", 
                                            xref="paper", yref="paper",
                                            x=0.5, y=0.5, showarrow=False)
        
        # Calculate average percentages by song and category
        categories = ['Ethnic_Identity', 'Religious_Identity', 'Collective_Memory']
        
        comparison_data = []
        for song in ['Despacito', 'See You Again']:
            song_data = self.task2_yearly[self.task2_yearly['Song'] == song]
            for category in categories:
                avg_percentage = song_data[f"{category}_Percentage"].mean()
                total_comments = song_data[category].sum()
                comparison_data.append({
                    'Song': song,
                    'Category': category.replace('_', ' '),
                    'Average_Percentage': round(avg_percentage, 1),
                    'Total_Comments': total_comments
                })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Create grouped bar chart
        fig = px.bar(
            df_comparison,
            x='Category',
            y='Average_Percentage',
            color='Song',
            barmode='group',
            color_discrete_map={'Despacito': '#764ba2', 'See You Again': '#667eea'},
            title="Average Cultural Identity Representation - Song Comparison",
            labels={'Average_Percentage': 'Average Percentage (%)', 'Category': 'Cultural Identity Category'}
        )
        
        fig.update_layout(
            height=400,
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def analyze_identity_words(self):
        """Analyze specific words within ethnic and religious identity categories"""
        if self.task2_data is None:
            return None, None
            
        # Define the keyword lists for each category
        ethnic_keywords = ['latino', 'hispanic', 'asian', 'white', 'caucasian', 'black']
        religious_keywords = ['muslim', 'islam', 'christian', 'christ']
        
        ethnic_word_counts = {}
        religious_word_counts = {}
        
        # Count occurrences by song and overall
        for song in ['Despacito', 'See You Again']:
            song_data = self.task2_data[self.task2_data['Song'] == song]
            
            # Analyze ethnic identity words
            for word in ethnic_keywords:
                # Count in comments where Ethnic Identity is not null
                ethnic_comments = song_data[song_data['Ethnic Identity'].notna()]
                word_count = ethnic_comments['original_comment'].str.lower().str.count(word).sum()
                
                if word not in ethnic_word_counts:
                    ethnic_word_counts[word] = {'Despacito': 0, 'See You Again': 0, 'Total': 0}
                ethnic_word_counts[word][song] = word_count
                ethnic_word_counts[word]['Total'] += word_count
            
            # Analyze religious identity words  
            for word in religious_keywords:
                # Count in comments where Religious Identity is not null
                religious_comments = song_data[song_data['Religious Identity'].notna()]
                word_count = religious_comments['original_comment'].str.lower().str.count(word).sum()
                
                if word not in religious_word_counts:
                    religious_word_counts[word] = {'Despacito': 0, 'See You Again': 0, 'Total': 0}
                religious_word_counts[word][song] = word_count
                religious_word_counts[word]['Total'] += word_count
        
        return ethnic_word_counts, religious_word_counts
    
    def create_ethnic_identity_word_chart(self):
        """Create bar chart showing ethnic identity word frequencies"""
        ethnic_words, _ = self.analyze_identity_words()
        
        if not ethnic_words:
            return go.Figure().add_annotation(text="No ethnic identity data available", 
                                            xref="paper", yref="paper",
                                            x=0.5, y=0.5, showarrow=False)
        
        # Prepare data for plotting
        words = list(ethnic_words.keys())
        despacito_counts = [ethnic_words[word]['Despacito'] for word in words]
        seeyou_counts = [ethnic_words[word]['See You Again'] for word in words]
        total_counts = [ethnic_words[word]['Total'] for word in words]
        
        # Create grouped bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Despacito',
            x=words,
            y=despacito_counts,
            marker_color='#FF6B6B',
            text=despacito_counts,
            textposition='outside',
            hovertemplate="<b>Despacito</b><br>Word: %{x}<br>Count: %{y}<extra></extra>"
        ))
        
        fig.add_trace(go.Bar(
            name='See You Again',
            x=words,
            y=seeyou_counts,
            marker_color='#4ECDC4',
            text=seeyou_counts,
            textposition='outside',
            hovertemplate="<b>See You Again</b><br>Word: %{x}<br>Count: %{y}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Ethnic Identity Keywords - Frequency Analysis",
            xaxis_title="Ethnic Identity Keywords",
            yaxis_title="Frequency Count",
            barmode='group',
            height=500,
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        return fig
    
    def create_religious_identity_word_chart(self):
        """Create bar chart showing religious identity word frequencies"""
        _, religious_words = self.analyze_identity_words()
        
        if not religious_words:
            return go.Figure().add_annotation(text="No religious identity data available", 
                                            xref="paper", yref="paper",
                                            x=0.5, y=0.5, showarrow=False)
        
        # Prepare data for plotting
        words = list(religious_words.keys())
        despacito_counts = [religious_words[word]['Despacito'] for word in words]
        seeyou_counts = [religious_words[word]['See You Again'] for word in words]
        
        # Create grouped bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Despacito',
            x=words,
            y=despacito_counts,
            marker_color='#4ECDC4',
            text=despacito_counts,
            textposition='outside',
            hovertemplate="<b>Despacito</b><br>Word: %{x}<br>Count: %{y}<extra></extra>"
        ))
        
        fig.add_trace(go.Bar(
            name='See You Again',
            x=words,
            y=seeyou_counts,
            marker_color='#45B7D1',
            text=seeyou_counts,
            textposition='outside',
            hovertemplate="<b>See You Again</b><br>Word: %{x}<br>Count: %{y}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Religious Identity Keywords - Frequency Analysis",
            xaxis_title="Religious Identity Keywords",
            yaxis_title="Frequency Count",
            barmode='group',
            height=500,
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        return fig
    
    def create_identity_word_comparison_chart(self):
        """Create a combined comparison chart for all identity words"""
        ethnic_words, religious_words = self.analyze_identity_words()
        
        if not ethnic_words and not religious_words:
            return go.Figure().add_annotation(text="No identity word data available", 
                                            xref="paper", yref="paper",
                                            x=0.5, y=0.5, showarrow=False)
        
        # Combine all words with their totals
        all_words = {}
        
        # Add ethnic words
        for word, counts in ethnic_words.items():
            all_words[f"{word} (ethnic)"] = counts['Total']
        
        # Add religious words
        for word, counts in religious_words.items():
            all_words[f"{word} (religious)"] = counts['Total']
        
        # Sort by frequency
        sorted_words = sorted(all_words.items(), key=lambda x: x[1], reverse=True)
        words, counts = zip(*sorted_words) if sorted_words else ([], [])
        
        # Create bar chart
        fig = go.Figure()
        
        # Color code by category
        colors = ['#FF6B6B' if '(ethnic)' in word else '#4ECDC4' for word in words]
        
        fig.add_trace(go.Bar(
            x=list(words),
            y=list(counts),
            marker_color=colors,
            text=list(counts),
            textposition='outside',
            hovertemplate="<b>%{x}</b><br>Total Count: %{y}<extra></extra>",
            showlegend=False
        ))
        
        fig.update_layout(
            title="Identity Keywords - Overall Frequency Ranking",
            xaxis_title="Keywords (Category)",
            yaxis_title="Total Frequency Count",
            height=500,
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(tickangle=45)
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        return fig
    
    def analyze_yearly_word_distribution(self):
        """Analyze yearly distribution of specific words within ethnic and religious categories"""
        if self.task2_data is None:
            return None, None
        
        # Define keyword lists
        ethnic_keywords = ['latino', 'hispanic', 'asian', 'white', 'caucasian', 'black']
        religious_keywords = ['muslim', 'islam', 'christian', 'christ']
        
        ethnic_yearly = {}
        religious_yearly = {}
        
        # Initialize structures
        for song in ['Despacito', 'See You Again']:
            ethnic_yearly[song] = {}
            religious_yearly[song] = {}
            
            song_data = self.task2_data[self.task2_data['Song'] == song]
            years = sorted(song_data['Year'].unique())
            
            for year in years:
                year_data = song_data[song_data['Year'] == year]
                
                # Count ethnic words for this year
                ethnic_yearly[song][year] = {}
                for word in ethnic_keywords:
                    # Count in comments where Ethnic Identity is not null
                    ethnic_comments = year_data[year_data['Ethnic Identity'].notna()]
                    word_count = ethnic_comments['original_comment'].str.lower().str.count(word).sum()
                    ethnic_yearly[song][year][word] = word_count
                
                # Count religious words for this year
                religious_yearly[song][year] = {}
                for word in religious_keywords:
                    # Count in comments where Religious Identity is not null
                    religious_comments = year_data[year_data['Religious Identity'].notna()]
                    word_count = religious_comments['original_comment'].str.lower().str.count(word).sum()
                    religious_yearly[song][year][word] = word_count
        
        return ethnic_yearly, religious_yearly
    
    def create_ethnic_yearly_distribution_chart(self):
        """Create stacked bar chart showing yearly distribution of ethnic identity words"""
        ethnic_yearly, _ = self.analyze_yearly_word_distribution()
        
        if not ethnic_yearly:
            return go.Figure().add_annotation(text="No ethnic identity yearly data available", 
                                            xref="paper", yref="paper",
                                            x=0.5, y=0.5, showarrow=False)
        
        # Create subplots for each song
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Despacito - Ethnic Identity Words by Year', 'See You Again - Ethnic Identity Words by Year'],
            vertical_spacing=0.1,
            row_heights=[0.5, 0.5]
        )
        
        ethnic_keywords = ['latino', 'hispanic', 'asian', 'white', 'caucasian', 'black']
        colors = {
            'latino': '#FF6B6B',
            'hispanic': '#FF8E53', 
            'asian': '#FF6B9D',
            'white': '#C44569',
            'caucasian': '#F8B500',
            'black': '#6C5CE7'
        }
        
        for i, (song, song_data) in enumerate(ethnic_yearly.items(), 1):
            years = sorted(song_data.keys())
            
            for word in ethnic_keywords:
                word_counts = [song_data[year].get(word, 0) for year in years]
                
                fig.add_trace(go.Bar(
                    name=word.capitalize(),
                    x=years,
                    y=word_counts,
                    marker_color=colors[word],
                    text=word_counts,
                    textposition='inside',
                    textfont=dict(color='white', size=9),
                    hovertemplate=f"<b>{song} - {word.capitalize()}</b><br>" +
                                "Year: %{x}<br>" +
                                "Count: %{y}<br>" +
                                "<extra></extra>",
                    showlegend=(i == 1)  # Show legend only for first subplot
                ), row=i, col=1)
        
        fig.update_layout(
            title="Ethnic Identity Keywords - Yearly Distribution Analysis",
            height=800,
            barmode='stack',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            ),
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', title_text="Year")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', title_text="Word Count")
        
        return fig
    
    def create_religious_yearly_distribution_chart(self):
        """Create stacked bar chart showing yearly distribution of religious identity words"""
        _, religious_yearly = self.analyze_yearly_word_distribution()
        
        if not religious_yearly:
            return go.Figure().add_annotation(text="No religious identity yearly data available", 
                                            xref="paper", yref="paper",
                                            x=0.5, y=0.5, showarrow=False)
        
        # Create subplots for each song
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Despacito - Religious Identity Words by Year', 'See You Again - Religious Identity Words by Year'],
            vertical_spacing=0.1,
            row_heights=[0.5, 0.5]
        )
        
        religious_keywords = ['muslim', 'islam', 'christian', 'christ']
        colors = {
            'muslim': '#4ECDC4',
            'islam': '#45B7D1',
            'christian': '#96CEB4', 
            'christ': '#FFEAA7'
        }
        
        for i, (song, song_data) in enumerate(religious_yearly.items(), 1):
            years = sorted(song_data.keys())
            
            for word in religious_keywords:
                word_counts = [song_data[year].get(word, 0) for year in years]
                
                fig.add_trace(go.Bar(
                    name=word.capitalize(),
                    x=years,
                    y=word_counts,
                    marker_color=colors[word],
                    text=word_counts,
                    textposition='inside',
                    textfont=dict(color='white', size=9),
                    hovertemplate=f"<b>{song} - {word.capitalize()}</b><br>" +
                                "Year: %{x}<br>" +
                                "Count: %{y}<br>" +
                                "<extra></extra>",
                    showlegend=(i == 1)  # Show legend only for first subplot
                ), row=i, col=1)
        
        fig.update_layout(
            title="Religious Identity Keywords - Yearly Distribution Analysis",
            height=800,
            barmode='stack',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            ),
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', title_text="Year")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', title_text="Word Count")
        
        return fig

    def create_task1_enhanced_chart(self):
        """Create enhanced Task 1 visualization"""
        if self.task1_data is None:
            return go.Figure().add_annotation(text="Task 1 data not available", 
                                            xref="paper", yref="paper",
                                            x=0.5, y=0.5, showarrow=False)
        
        # Create dual-axis chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        for video in self.task1_data['Video'].unique():
            video_data = self.task1_data[self.task1_data['Video'] == video].sort_values('Year')
            video_name = 'Despacito' if video == 'Despa' else 'See You Again'
            
            # Add comment count trace
            fig.add_trace(
                go.Scatter(
                    x=video_data['Year'].tolist(),
                    y=video_data['Total Comments'].tolist(),
                    mode='lines+markers',
                    name=f"{video_name} - Comments",
                    line=dict(width=3),
                    marker=dict(size=8)
                ),
                secondary_y=False,
            )
            
            # Add unique countries trace
            fig.add_trace(
                go.Scatter(
                    x=video_data['Year'].tolist(),
                    y=video_data['Unique Countries'].tolist(),
                    mode='lines+markers',
                    name=f"{video_name} - Countries",
                    line=dict(width=3, dash='dash'),
                    marker=dict(size=8, symbol='diamond')
                ),
                secondary_y=True,
            )
        
        # Set axis titles
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="Total Comments", secondary_y=False)
        fig.update_yaxes(title_text="Unique Countries", secondary_y=True)
        
        fig.update_layout(
            title="Task 1: Geographic Analysis - Comments and Country Diversity",
            height=500,
            font=dict(size=12),
            legend=dict(x=0.02, y=0.98),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def generate_insights_summary(self):
        """Generate insights and summary statistics"""
        insights = {
            'timestamp': datetime.now().isoformat(),
            'task1_insights': {},
            'task2_insights': {},
            'comparative_insights': {}
        }
        
        # Task 1 insights
        if self.task1_data is not None:
            peak_year_despa = None
            peak_year_seeyou = None
            
            despa_data = self.task1_data[self.task1_data['Video'] == 'Despa']
            if not despa_data.empty:
                peak_year_despa = int(despa_data.loc[despa_data['Total Comments'].idxmax(), 'Year'])
                
            seeyou_data = self.task1_data[self.task1_data['Video'] == 'Seeyou']
            if not seeyou_data.empty:
                peak_year_seeyou = int(seeyou_data.loc[seeyou_data['Total Comments'].idxmax(), 'Year'])
                
            insights['task1_insights'] = {
                'total_geographic_comments': int(self.task1_data['Total Comments'].sum()),
                'peak_year_despa': peak_year_despa,
                'peak_year_seeyou': peak_year_seeyou,
                'max_countries': int(self.task1_data['Unique Countries'].max())
            }
        
        # Task 2 insights
        if self.task2_yearly is not None and not self.task2_yearly.empty:
            insights['task2_insights'] = {
                'total_cultural_comments': int(self.task2_data.shape[0]) if self.task2_data is not None else 0,
                'dominant_category_despa': self.get_dominant_category('Despacito'),
                'dominant_category_seeyou': self.get_dominant_category('See You Again'),
                'yearly_coverage': [int(year) for year in sorted(self.task2_yearly['Year'].unique())],
                'category_totals': {
                    'ethnic_identity': int(self.task2_yearly['Ethnic_Identity'].sum()),
                    'religious_identity': int(self.task2_yearly['Religious_Identity'].sum()),
                    'collective_memory': int(self.task2_yearly['Collective_Memory'].sum())
                }
            }
        
        return insights
    
    def get_dominant_category(self, song):
        """Get the dominant cultural category for a song"""
        if self.task2_yearly is None:
            return None
            
        song_data = self.task2_yearly[self.task2_yearly['Song'] == song]
        if song_data.empty:
            return None
            
        categories = ['Ethnic_Identity', 'Religious_Identity', 'Collective_Memory']
        totals = {cat: int(song_data[cat].sum()) for cat in categories}
        
        dominant = max(totals, key=totals.get)
        return {
            'category': dominant.replace('_', ' '),
            'total': int(totals[dominant]),
            'average_percentage': float(round(song_data[f"{dominant}_Percentage"].mean(), 1))
        }
    
    def create_advanced_dashboard(self):
        """Create the advanced interactive dashboard"""
        print("🚀 Creating Advanced Cultural Analysis Dashboard...")
        
        # Load all data
        self.load_all_data()
        
        # Generate insights
        insights = self.generate_insights_summary()
        
        # Create all charts
        enhanced_stacked = self.create_enhanced_stacked_bar_chart()
        temporal_trends = self.create_temporal_trend_analysis()
        comparative_metrics = self.create_comparative_metrics_chart()
        task1_enhanced = self.create_task1_enhanced_chart()
        
        # Create new identity word analysis charts
        ethnic_word_chart = self.create_ethnic_identity_word_chart()
        religious_word_chart = self.create_religious_identity_word_chart()
        word_comparison_chart = self.create_identity_word_comparison_chart()
        
        # Create yearly distribution charts
        ethnic_yearly_chart = self.create_ethnic_yearly_distribution_chart()
        religious_yearly_chart = self.create_religious_yearly_distribution_chart()
        
        # Calculate summary statistics
        if self.task2_yearly is not None:
            total_task2_comments = self.task2_yearly[['Ethnic_Identity', 'Religious_Identity', 'Collective_Memory']].sum().sum()
            years_covered = len(self.task2_yearly['Year'].unique())
        else:
            total_task2_comments = 0
            years_covered = 0
            
        if self.task1_data is not None:
            total_task1_comments = self.task1_data['Total Comments'].sum()
            unique_countries = self.task1_data['Unique Countries'].max()
        else:
            total_task1_comments = 0
            unique_countries = 0
        
        # Generate chart HTML content for embedding with explicit config to avoid binary encoding
        import plotly.offline as pyo
        import plotly.io as pio
        
        # Configure plotly to not use binary encoding
        config = {
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'cultural_analysis_chart',
                'height': 500,
                'width': 700,
                'scale': 1
            },
            'displayModeBar': False
        }
        
        # Generate individual chart HTML content with explicit config
        task1_html = pyo.plot(task1_enhanced, output_type='div', include_plotlyjs=False, config=config)
        stacked_html = pyo.plot(enhanced_stacked, output_type='div', include_plotlyjs=False, config=config)
        temporal_html = pyo.plot(temporal_trends, output_type='div', include_plotlyjs=False, config=config)
        comparative_html = pyo.plot(comparative_metrics, output_type='div', include_plotlyjs=False, config=config)
        
        # Generate new identity word charts HTML
        ethnic_word_html = pyo.plot(ethnic_word_chart, output_type='div', include_plotlyjs=False, config=config)
        religious_word_html = pyo.plot(religious_word_chart, output_type='div', include_plotlyjs=False, config=config)
        word_comparison_html = pyo.plot(word_comparison_chart, output_type='div', include_plotlyjs=False, config=config)
        
        # Generate yearly distribution charts HTML
        ethnic_yearly_html = pyo.plot(ethnic_yearly_chart, output_type='div', include_plotlyjs=False, config=config)
        religious_yearly_html = pyo.plot(religious_yearly_chart, output_type='div', include_plotlyjs=False, config=config)
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Advanced Cultural Analysis Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
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
        .dashboard-subtitle {{
            opacity: 0.9;
            font-size: 18px;
            margin-top: 10px;
        }}
        .chart-container {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border: 1px solid rgba(255,255,255,0.2);
        }}
        .chart-title {{
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 20px;
            color: #2d3748;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 10px;
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
        .info-panel {{
            background: linear-gradient(135deg, #e6fffa 0%, #f0fff4 100%);
            border-left: 5px solid #38b2ac;
            padding: 20px;
            margin: 25px 0;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
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
        .methodology-panel {{
            background: linear-gradient(135deg, #f0f4ff 0%, #e6f3ff 100%);
            border-left: 5px solid #4299e1;
            padding: 20px;
            margin: 25px 0;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1>🎵 Advanced Cultural Analysis Dashboard</h1>
        <div class="dashboard-subtitle">
            Comprehensive Analysis of Cultural Identity in YouTube Comments<br>
            <em>"Despacito" vs "See You Again" - Cross-Cultural Comparative Study</em>
        </div>
    </div>
    
    <div class="info-panel">
        <h3>📊 Dashboard Overview</h3>
        <p><strong>Task 1:</strong> Geographic distribution analysis identifying comments with country mentions, city references, and flag emojis</p>
        <p><strong>Task 2:</strong> Cultural identity categorization across three dimensions:</p>
        <div class="insights-grid">
            <div class="insight-item">
                <strong>🌍 Ethnic Identity:</strong> latino, hispanic, asian, white, caucasian, black
            </div>
            <div class="insight-item">
                <strong>🕊️ Religious Identity:</strong> muslim, islam, christian, christ
            </div>
            <div class="insight-item">
                <strong>🏠 Collective Memory:</strong> my country, our country, in my culture, homeland, diaspora, immigrant
            </div>
        </div>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-number">{total_task1_comments:,}</div>
            <div class="stat-label">Geographic Comments<br>Task 1 Analysis</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{total_task2_comments:,}</div>
            <div class="stat-label">Cultural Identity Mentions<br>Task 2 Analysis</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{unique_countries}</div>
            <div class="stat-label">Unique Countries<br>Identified</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{years_covered}</div>
            <div class="stat-label">Years Analyzed<br>Temporal Coverage</div>
        </div>
    </div>
    
    <div class="chart-container">
        <div class="chart-title">📈 Task 1: Enhanced Geographic Analysis</div>
        {task1_html}
    </div>
    
    <div class="chart-container">
        <div class="chart-title">📊 Task 2: Cultural Identity Categories - Comparative Stacked Analysis</div>
        {stacked_html}
    </div>
    
    <div class="chart-container">
        <div class="chart-title">📉 Temporal Trends: Cultural Identity Evolution</div>
        {temporal_html}
    </div>
    
    <div class="chart-container">
        <div class="chart-title">⚖️ Comparative Metrics: Song-wise Cultural Representation</div>
        {comparative_html}
    </div>
    
    <div class="chart-container">
        <div class="chart-title">🎯 Ethnic Identity Keywords - Word Frequency Analysis</div>
        {ethnic_word_html}
    </div>
    
    <div class="chart-container">
        <div class="chart-title">🙏 Religious Identity Keywords - Word Frequency Analysis</div>
        {religious_word_html}
    </div>
    
    <div class="chart-container">
        <div class="chart-title">📈 Identity Keywords - Overall Frequency Ranking</div>
        {word_comparison_html}
    </div>
    
    <div class="chart-container">
        <div class="chart-title">📊 Ethnic Identity Keywords - Yearly Distribution (Stacked)</div>
        {ethnic_yearly_html}
    </div>
    
    <div class="chart-container">
        <div class="chart-title">🙏 Religious Identity Keywords - Yearly Distribution (Stacked)</div>
        {religious_yearly_html}
    </div>
    
    <div class="insights-panel">
        <h3>💡 Key Insights</h3>
        <div class="insights-grid">"""
        
        # Add insights based on data
        if 'task1_insights' in insights and insights['task1_insights']:
            t1 = insights['task1_insights']
            html_content += f"""
            <div class="insight-item">
                <strong>🌍 Geographic Engagement:</strong><br>
                Total comments with geographic references: {t1.get('total_geographic_comments', 'N/A'):,}<br>
                Maximum countries in a year: {t1.get('max_countries', 'N/A')}
            </div>"""
        
        if 'task2_insights' in insights and insights['task2_insights']:
            t2 = insights['task2_insights']
            html_content += f"""
            <div class="insight-item">
                <strong>🏛️ Cultural Categories:</strong><br>
                Ethnic Identity: {t2.get('category_totals', {}).get('ethnic_identity', 'N/A'):,} mentions<br>
                Religious Identity: {t2.get('category_totals', {}).get('religious_identity', 'N/A'):,} mentions<br>
                Collective Memory: {t2.get('category_totals', {}).get('collective_memory', 'N/A'):,} mentions
            </div>"""
            
            # Add dominant categories
            if t2.get('dominant_category_despa'):
                dom_despa = t2['dominant_category_despa']
                html_content += f"""
            <div class="insight-item">
                <strong>🎵 Despacito Dominant Category:</strong><br>
                {dom_despa['category']} ({dom_despa['total']:,} mentions)<br>
                Average: {dom_despa['average_percentage']}% per year
            </div>"""
                
            if t2.get('dominant_category_seeyou'):
                dom_seeyou = t2['dominant_category_seeyou']
                html_content += f"""
            <div class="insight-item">
                <strong>🎭 See You Again Dominant Category:</strong><br>
                {dom_seeyou['category']} ({dom_seeyou['total']:,} mentions)<br>
                Average: {dom_seeyou['average_percentage']}% per year
            </div>"""
        
        html_content += f"""
        </div>
    </div>
    
    <div class="methodology-panel">
        <h3>🔬 Methodology & Data Processing</h3>
        <p><strong>Data Sources:</strong></p>
        <ul style="margin: 10px 0; padding-left: 30px;">
            <li><strong>Task 1:</strong> Geographic extraction using regex patterns for countries, cities, flag emoji detection, and location keywords</li>
            <li><strong>Task 2:</strong> Cultural identity keyword matching with context filtering across three predefined categories</li>
            <li><strong>Temporal Analysis:</strong> Year-based aggregation with percentage calculations for trend identification</li>
        </ul>
        <p><strong>Processing Notes:</strong></p>
        <ul style="margin: 10px 0; padding-left: 30px;">
            <li>Comments can belong to multiple cultural categories simultaneously</li>
            <li>Yearly statistics calculated as both absolute counts and percentages</li>
            <li>Geographic data includes country mentions, flag emojis, and location-based keywords</li>
            <li>Analysis covers years {min(insights.get('task2_insights', {}).get('yearly_coverage', [2015]))}-{max(insights.get('task2_insights', {}).get('yearly_coverage', [2025]))} based on available data</li>
        </ul>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    

    
</body>
</html>"""
        
        # Save the enhanced dashboard
        with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Save insights as JSON
        insights_file = BASE_DIR / "cultural_analysis_insights.json"
        with open(insights_file, 'w', encoding='utf-8') as f:
            json.dump(insights, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Advanced dashboard saved to: {OUTPUT_HTML}")
        print(f"📊 Insights summary saved to: {insights_file}")
        
        return str(OUTPUT_HTML), str(insights_file)

def main():
    """Main execution function"""
    print("🚀 Advanced Cultural Analysis Dashboard Generator")
    print("=" * 70)
    
    # Create visualizer instance
    visualizer = AdvancedCulturalAnalysisVisualizer()
    
    try:
        # Generate advanced dashboard
        dashboard_path, insights_path = visualizer.create_advanced_dashboard()
        
        print("\n" + "=" * 70)
        print("✅ ADVANCED ANALYSIS COMPLETE!")
        print(f"📊 Interactive Dashboard: {dashboard_path}")
        print(f"💡 Insights Summary: {insights_path}")
        print("\n🌟 Features included:")
        print("  • Enhanced stacked bar charts with comparative view")
        print("  • Temporal trend analysis with percentage tracking")
        print("  • Comparative metrics across songs")
        print("  • Geographic analysis with dual-axis visualization")
        print("  • Key insights and statistical summaries")
        print("  • Comprehensive methodology documentation")
        print(f"\n🌟 Open the dashboard file in your browser to explore!")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main() 