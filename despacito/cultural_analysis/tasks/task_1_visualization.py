import subprocess
import sys
import re
import os
from tqdm import tqdm
import json
import logging
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import emoji
import pycountry
from typing import Tuple

def install(package):
    """Installs a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# --- Install required packages ---
try:
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    import emoji
    import pycountry
except ImportError:
    print("One or more required packages are not installed. Installing now...")
    install("pandas")
    install("plotly")
    install("emoji")
    install("pycountry")
    install("tqdm")
    print("Packages installed successfully. Please re-run the script.")
    sys.exit()

# Initialize tqdm for pandas
tqdm.pandas(desc="Processing")

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PREFIXES = ["despa", "seeyou"]
OUTPUT_UNIFIED_CSV = os.path.join(BASE_DIR, "task_1_unified_countries.csv")
OUTPUT_STATS_CSV = os.path.join(BASE_DIR, "task_1_yearly_stats.csv")

# Create a list of country names for regex matching
COUNTRY_NAMES = [country.name for country in pycountry.countries]
COUNTRY_REGEX = re.compile(r'\b(' + '|'.join(re.escape(name) for name in COUNTRY_NAMES) + r')\b', re.IGNORECASE)

# Manual mapping for edge cases where pycountry names differ from common usage
COUNTRY_NAME_MAPPING = {
    "United States": "United States",
    "USA": "United States",
    "U.S.A": "United States",
    "U.S.": "United States",
    "United Kingdom": "United Kingdom",
    "UK": "United Kingdom",
    "U.K.": "United Kingdom",
    "England": "United Kingdom",
    "UAE": "United Arab Emirates",
    "Bosnia": "Bosnia and Herzegovina",
}

# --- Data Processing Functions ---

def get_country_from_emoji(flag_emoji):
    """Converts a flag emoji to a standardized country name."""
    if not isinstance(flag_emoji, str):
        return None
    try:
        # The emoji library converts '🇺🇸' to ':United_States:'
        demojized = emoji.demojize(flag_emoji)
        # Extract the name between the colons
        match = re.search(r':([a-zA-Z_]+):', demojized)
        if match:
            country_name_guess = match.group(1).replace('_', ' ')
            # Use pycountry to find the official name
            try:
                country = pycountry.countries.get(name=country_name_guess)
                if country:
                    return country.name
                # If direct lookup fails, try searching
                results = pycountry.countries.search_fuzzy(country_name_guess)
                if results:
                    return results[0].name
            except (KeyError, AttributeError):
                 # Fallback for fuzzy search if official name not found
                results = pycountry.countries.search_fuzzy(country_name_guess)
                if results:
                    return results[0].name
    except Exception:
        return None
    return None

def extract_country_from_text(text):
    """Extracts a country name from a line of text using regex."""
    if not isinstance(text, str):
        return None
    
    # First, check for manual mappings like "UK" or "USA"
    for alias, standard_name in COUNTRY_NAME_MAPPING.items():
        if re.search(r'\b' + re.escape(alias) + r'\b', text, re.IGNORECASE):
            return standard_name

    # If no manual mapping found, use the comprehensive regex
    match = COUNTRY_REGEX.search(text)
    if match:
        matched_name = match.group(1)
        try:
            # Verify the found name with pycountry to get the standard name
            country = pycountry.countries.get(name=matched_name)
            if country:
                return country.name
            else: # If get() returns None, try fuzzy search
                results = pycountry.countries.search_fuzzy(matched_name)
                if results:
                    return results[0].name
        except Exception:
            # Fallback for any error during lookup (e.g., non-standard names)
            try:
                results = pycountry.countries.search_fuzzy(matched_name)
                if results:
                    return results[0].name
            except Exception:
                # If fuzzy search also fails, return None
                return None
    return None


print("Functions for country extraction are defined.")

def process_and_unify_data():
    """Loads, processes, and unifies the data from all source CSVs."""
    all_video_dfs = []

    for prefix in FILE_PREFIXES:
        print(f"--- Processing data for '{prefix}' ---")
        
        # Define file paths
        base_output_dir = os.path.join(BASE_DIR, f"{prefix}_outputs")
        flag_file = os.path.join(base_output_dir, f"{prefix}_task_1_flag_emoji_extracted_comments.csv")
        keyword_file = os.path.join(base_output_dir, f"{prefix}_task_1_keyword_extracted_comments.csv")
        location_file = os.path.join(base_output_dir, f"{prefix}_task_1_location_extracted_comments.csv")

        try:
            # Load all three files
            df_flag = pd.read_csv(flag_file)
            df_keyword = pd.read_csv(keyword_file)
            df_location = pd.read_csv(location_file)

            # Create the boolean 'is_spam' column and process emotion_label for each DataFrame
            for df in [df_flag, df_keyword, df_location]:
                # Handle spam label
                if 'spam_label' in df.columns:
                    df['is_spam'] = (df['spam_label'].astype(str).str.lower() == 'spam')
                elif 'is_spam' in df.columns:
                    if df['is_spam'].dtype == object:
                        df['is_spam'] = df['is_spam'].str.lower().isin(['true', '1', 't', 'y', 'yes', 'spam'])
                    else:
                        df['is_spam'] = df['is_spam'].astype(bool)
                else:
                    print(f"Warning: 'spam_label' or 'is_spam' column not found for {prefix}. Assuming non-spam.")
                    df['is_spam'] = False
                
                # Handle emotion label
                if 'emotion_label' not in df.columns:
                    print(f"Warning: 'emotion_label' column not found for {prefix}. Defaulting to 'unknown'.")
                    df['emotion_label'] = 'unknown'
                else:
                    df['emotion_label'] = df['emotion_label'].fillna('unknown')

            # Process each and add the 'Country' column
            print("Extracting countries...")
            df_flag['Country'] = df_flag['original_comment'].progress_apply(get_country_from_emoji)
            df_keyword['Country'] = df_keyword['original_comment'].progress_apply(extract_country_from_text)
            df_location['Country'] = df_location['original_comment'].progress_apply(extract_country_from_text)

            # Combine the dataframes, ensuring 'is_spam' is kept
            combined_df = pd.concat([
                df_flag.dropna(subset=['Country']),
                df_keyword.dropna(subset=['Country']),
                df_location.dropna(subset=['Country'])
            ], ignore_index=True).drop_duplicates(subset=['original_comment'])

            if combined_df.empty:
                print(f"Warning: No country data found for '{prefix}'.")
                continue

            # Standardize data and ensure all columns are present
            combined_df['Source Video'] = 'Despacito' if prefix == 'despa' else 'See You Again'
            combined_df['published_at'] = pd.to_datetime(combined_df['published_at'], errors='coerce')
            combined_df.dropna(subset=['published_at'], inplace=True)
            combined_df['Year'] = combined_df['published_at'].dt.year
            combined_df.rename(columns={'original_comment': 'Comment Text', 'published_at': 'Comment Published Date', 'emotion_label': 'Emotion'}, inplace=True)

            # Final check and selection of columns
            final_columns = ['Comment Text', 'Comment Published Date', 'is_spam', 'Emotion', 'Year', 'Country', 'Source Video']
            all_video_dfs.append(combined_df[final_columns])
            print(f"Successfully processed {len(combined_df)} comments for '{prefix}'.")

        except Exception as e:
            print(f"An unexpected error occurred during file processing for prefix '{prefix}': {e}")
            continue

    if not all_video_dfs:
        print("Error: No data was processed successfully. Exiting.")
        sys.exit()
        
    # --- Final Combination ---
    final_df = pd.concat(all_video_dfs, ignore_index=True)
    print(f"\nTotal unified comments: {len(final_df)}")

    # Save the unified data
    final_df.to_csv(OUTPUT_UNIFIED_CSV, index=False)
    print(f"Unified data saved to '{OUTPUT_UNIFIED_CSV}'")
    
    return final_df

def get_iso_alpha(country_name):
    """Converts a country name to its ISO alpha-3 code."""
    try:
        return pycountry.countries.get(name=country_name).alpha_3
    except AttributeError:
        # Try a fuzzy search for names that don't match exactly
        try:
            return pycountry.countries.search_fuzzy(country_name)[0].alpha_3
        except (LookupError, AttributeError):
            return None # Return None if no country is found

def get_emoji_from_country(country_name):
    """Converts a country name to its flag emoji."""
    try:
        # First, get the 2-letter country code
        country_code = pycountry.countries.get(name=country_name).alpha_2
        # Convert the code to regional indicator symbols
        emoji_str = "".join([chr(ord(c) - ord('A') + 0x1F1E6) for c in country_code.upper()])
        return emoji_str
    except (AttributeError, LookupError):
         # Try a fuzzy search if the first lookup fails
        try:
            country_code = pycountry.countries.search_fuzzy(country_name)[0].alpha_2
            emoji_str = "".join([chr(ord(c) - ord('A') + 0x1F1E6) for c in country_code.upper()])
            return emoji_str
        except (AttributeError, LookupError):
            return "🏳️" # Return a default white flag if not found

def generate_yearly_top5_figures(df: pd.DataFrame) -> dict:
    """
    Generates interactive yearly top 5 country bar charts for each song.
    Returns a dictionary containing the figure, controls, and script for each song.
    """
    if df is None or df.empty:
        logging.warning("Input DataFrame for yearly top 5 generation is empty.")
        return {}

    # This helper function is defined inside to capture the scope of this function
    def _generate_filter_html_and_js(plot_div_id, trace_map, spam_modes, emotion_modes, total_traces):
        """Generates the HTML and JavaScript for the interactive filters."""
        trace_map_json = json.dumps(trace_map)
        spam_modes_json = json.dumps(spam_modes)
        emotion_modes_json = json.dumps(emotion_modes)

        controls_html = f"""
        <div class="filter-controls">
            <h4>Spam Filter</h4>
            <div id="{plot_div_id}-spam-filters" class="filter-group"></div>
            <h4>Emotion Filter</h4>
            <div id="{plot_div_id}-emotion-filters" class="filter-group"></div>
        </div>
        <style>
            .filter-controls {{ margin: 20px; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: #f9f9f9; }}
            .filter-controls h4 {{ margin-top: 0; margin-bottom: 5px; font-family: sans-serif; }}
            .filter-group div {{ display: inline-block; margin-right: 15px; }}
            .filter-group label {{ font-family: sans-serif; font-size: 14px; }}
        </style>
        """

        script_js = f"""
        <script>
            window.addEventListener('load', function() {{
                const plotDiv = document.getElementById('{plot_div_id}');
                if (!plotDiv) {{
                    console.error('Plotly div not found: {plot_div_id}');
                    return;
                }}

                const traceMap = {trace_map_json};
                const spamModes = {spam_modes_json};
                const emotionModes = {emotion_modes_json};
                const totalTraces = {total_traces};

                function createRadioCheckboxes(containerId, groupName, labels, changeCallback) {{
                    const container = document.getElementById(containerId);
                    labels.forEach((label, index) => {{
                        const wrapper = document.createElement('div');
                        const checkbox = document.createElement('input');
                        checkbox.type = 'checkbox';
                        checkbox.id = `filter-${{groupName}}-${{label.replace(/\\s+/g, '-')}}-${{index}}`;
                        checkbox.name = groupName;
                        checkbox.value = index;
                        if (index === 0) checkbox.checked = true;

                        const labelEl = document.createElement('label');
                        labelEl.htmlFor = checkbox.id;
                        labelEl.textContent = ` ${{label}}`;

                        checkbox.addEventListener('change', () => {{
                            document.getElementsByName(groupName).forEach(cb => {{
                                if (cb !== checkbox) cb.checked = false;
                            }});
                            if (!Array.from(document.getElementsByName(groupName)).some(cb => cb.checked)) {{
                                checkbox.checked = true;
                            }}
                            changeCallback();
                        }});
                        
                        wrapper.appendChild(checkbox);
                        wrapper.appendChild(labelEl);
                        container.appendChild(wrapper);
                    }});
                }}

                function updatePlotVisibility() {{
                    const selectedSpamIndex = parseInt(document.querySelector(`input[name="{plot_div_id}-spam-filter"]:checked`).value);
                    const selectedEmotionIndex = parseInt(document.querySelector(`input[name="{plot_div_id}-emotion-filter"]:checked`).value);

                    const visibility = new Array(totalTraces).fill(false);
                    let visibleTraceCount = 0;
                    
                    traceMap.forEach(mapInfo => {{
                        if (mapInfo.spam_idx === selectedSpamIndex && mapInfo.emotion_idx === selectedEmotionIndex) {{
                            for (let i = 0; i < mapInfo.count; i++) {{
                                visibility[mapInfo.start + i] = true;
                            }}
                            visibleTraceCount = mapInfo.count;
                        }}
                    }});
                    
                    Plotly.restyle(plotDiv, {{ visible: visibility }});

                    const spamLabel = spamModes[selectedSpamIndex];
                    const emotionLabel = emotionModes[selectedEmotionIndex];
                    const originalTitle = plotDiv.layout.title.text.split(' (')[0];
                    const newTitle = `${{originalTitle}} (${{spamLabel}} / ${{emotionLabel}})`;
                    Plotly.relayout(plotDiv, 'title.text', newTitle);
                }}
                
                createRadioCheckboxes('{plot_div_id}-spam-filters', `{plot_div_id}-spam-filter`, spamModes, updatePlotVisibility);
                createRadioCheckboxes('{plot_div_id}-emotion-filters', `{plot_div_id}-emotion-filter`, emotionModes, updatePlotVisibility);
            }});
        </script>
        """
        return controls_html, script_js

    results = {}
    spam_modes = ['All', 'Non-Spam', 'Spam']
    unique_emotions = df['Emotion'].unique().tolist()
    emotion_modes = ['All'] + sorted([e for e in unique_emotions if e != 'unknown'])

    for video in df['Source Video'].unique():
        video_df = df[df['Source Video'] == video].copy()
        fig_bar = go.Figure()
        bar_trace_map = []
        bar_current_trace_idx = 0

        for s_idx, spam_mode in enumerate(spam_modes):
            for e_idx, emotion_mode in enumerate(emotion_modes):
                mode_df = video_df.copy()
                if spam_mode == 'Non-Spam': mode_df = mode_df[mode_df['is_spam'] == False]
                elif spam_mode == 'Spam': mode_df = mode_df[mode_df['is_spam'] == True]
                if emotion_mode != 'All': mode_df = mode_df[mode_df['Emotion'] == emotion_mode]

                top5_per_year = mode_df.groupby(['Year', 'Country']).size().reset_index(name='counts')
                # Get the actual top 5 countries by count for each year
                top5_per_year = top5_per_year.groupby('Year').apply(lambda x: x.nlargest(5, 'counts')).reset_index(drop=True)
                pivot_df = top5_per_year.pivot_table(index='Year', columns='Country', values='counts', aggfunc='sum').fillna(0)
                
                # Debug logging for the first few combinations
                if s_idx <= 1 and e_idx <= 1:
                    logging.info(f"Video: {video}, Spam: {spam_mode}, Emotion: {emotion_mode}")
                    logging.info(f"Mode DF shape: {mode_df.shape}")
                    logging.info(f"Top5 per year shape: {top5_per_year.shape}")
                    logging.info(f"Pivot DF shape: {pivot_df.shape}")
                    if not pivot_df.empty:
                        logging.info(f"Pivot DF columns: {list(pivot_df.columns)}")
                        logging.info(f"Pivot DF index: {list(pivot_df.index)}")
                
                num_traces_for_view = len(pivot_df.columns)
                bar_trace_map.append({"spam_idx": s_idx, "emotion_idx": e_idx, "start": bar_current_trace_idx, "count": num_traces_for_view})
                
                is_visible = (s_idx == 0) and (e_idx == 0)  # First combination (All/All) should be visible
                for country in pivot_df.columns:
                    emoji_flag = get_emoji_from_country(country)
                    fig_bar.add_trace(go.Bar(x=pivot_df.index, y=pivot_df[country], name=f"{emoji_flag} {country}", visible=is_visible))
                
                bar_current_trace_idx += num_traces_for_view

        fig_bar.update_layout(barmode='stack', title_text=f'Top 5 Countries by Comment Volume per Year (All / All)', xaxis_title='Year', yaxis_title='Number of Comments', legend_title='Country')
        
        plot_id = f'plotly-bar-{video.lower().replace(" ", "-")}'
        controls, script = _generate_filter_html_and_js(plot_id, bar_trace_map, spam_modes, emotion_modes, bar_current_trace_idx)
        
        results[video] = {
            "figure": fig_bar,
            "controls": controls,
            "script": script,
            "plot_id": plot_id
        }
    return results

def generate_heatmap_figures(df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    """
    Generates two interactive heatmap figures from the dataframe, one for each song.
    """
    if df is None or df.empty:
        logging.warning("Input DataFrame for heatmap generation is empty.")
        return go.Figure(), go.Figure()

    figs = {}
    for song_name in ['Despacito', 'See You Again']:
        song_df = df[df['Source Video'] == song_name].copy()
        
        if song_df.empty:
            logging.warning(f"No data for {song_name}, skipping heatmap generation.")
            figs[song_name] = go.Figure().add_annotation(text=f"No data for {song_name}", showarrow=False)
            continue

        # Convert country names to ISO alpha-3 codes for Plotly
        song_df['iso_alpha'] = song_df['Country'].apply(get_iso_alpha)
        song_df.dropna(subset=['iso_alpha'], inplace=True)
        
        # Aggregate data by country
        country_counts = song_df.groupby(['iso_alpha', 'Country']).size().reset_index(name='Comment Count')
        
        fig = px.choropleth(
            country_counts,
            locations="iso_alpha",
            color="Comment Count",
            hover_name="Country",
            color_continuous_scale=px.colors.sequential.Plasma,
            title=f"Geographic Distribution of Comments for {song_name}"
        )
        figs[song_name] = fig
        
    return figs.get('Despacito', go.Figure()), figs.get('See You Again', go.Figure())


def analyze_and_visualize(df):
    """Performs yearly analysis and generates visualizations with interactive checkbox filters."""
    print("\n--- Starting Analysis and Visualization ---")

    # Define all possible filter modes
    spam_modes = ['All', 'Non-Spam', 'Spam']
    unique_emotions = df['Emotion'].unique().tolist()
    if 'unknown' in unique_emotions:
        unique_emotions.remove('unknown')
    unique_emotions.sort()
    emotion_modes = ['All'] + unique_emotions

    def _generate_filter_html_and_js(plot_div_id, trace_map, spam_modes, emotion_modes, total_traces):
        """Generates the HTML and JavaScript for the interactive filters."""
        trace_map_json = json.dumps(trace_map)
        spam_modes_json = json.dumps(spam_modes)
        emotion_modes_json = json.dumps(emotion_modes)

        controls_html = f"""
        <div class="filter-controls">
            <h4>Spam Filter</h4>
            <div id="{plot_div_id}-spam-filters" class="filter-group"></div>
            <h4>Emotion Filter</h4>
            <div id="{plot_div_id}-emotion-filters" class="filter-group"></div>
        </div>
        <style>
            .filter-controls {{ margin: 20px; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: #f9f9f9; }}
            .filter-controls h4 {{ margin-top: 0; margin-bottom: 5px; font-family: sans-serif; }}
            .filter-group div {{ display: inline-block; margin-right: 15px; }}
            .filter-group label {{ font-family: sans-serif; font-size: 14px; }}
        </style>
        """

        script_js = f"""
        <script>
            window.addEventListener('load', function() {{
                const plotDiv = document.getElementById('{plot_div_id}');
                if (!plotDiv) {{
                    console.error('Plotly div not found: {plot_div_id}');
                    return;
                }}

                const traceMap = {trace_map_json};
                const spamModes = {spam_modes_json};
                const emotionModes = {emotion_modes_json};
                const totalTraces = {total_traces};

                function createRadioCheckboxes(containerId, groupName, labels, changeCallback) {{
                    const container = document.getElementById(containerId);
                    labels.forEach((label, index) => {{
                        const wrapper = document.createElement('div');
                        const checkbox = document.createElement('input');
                        checkbox.type = 'checkbox';
                        checkbox.id = `filter-${{groupName}}-${{label.replace(/\\s+/g, '-')}}-${{index}}`;
                        checkbox.name = groupName;
                        checkbox.value = index;
                        if (index === 0) checkbox.checked = true;

                        const labelEl = document.createElement('label');
                        labelEl.htmlFor = checkbox.id;
                        labelEl.textContent = ` ${{label}}`;

                        checkbox.addEventListener('change', () => {{
                            document.getElementsByName(groupName).forEach(cb => {{
                                if (cb !== checkbox) cb.checked = false;
                            }});
                            if (!Array.from(document.getElementsByName(groupName)).some(cb => cb.checked)) {{
                                checkbox.checked = true;
                            }}
                            changeCallback();
                        }});
                        
                        wrapper.appendChild(checkbox);
                        wrapper.appendChild(labelEl);
                        container.appendChild(wrapper);
                    }});
                }}

                function updatePlotVisibility() {{
                    const selectedSpamIndex = parseInt(document.querySelector(`input[name="{plot_div_id}-spam-filter"]:checked`).value);
                    const selectedEmotionIndex = parseInt(document.querySelector(`input[name="{plot_div_id}-emotion-filter"]:checked`).value);

                    const visibility = new Array(totalTraces).fill(false);
                    let visibleTraceCount = 0;
                    
                    traceMap.forEach(mapInfo => {{
                        if (mapInfo.spam_idx === selectedSpamIndex && mapInfo.emotion_idx === selectedEmotionIndex) {{
                            for (let i = 0; i < mapInfo.count; i++) {{
                                visibility[mapInfo.start + i] = true;
                            }}
                            visibleTraceCount = mapInfo.count;
                        }}
                    }});
                    
                    Plotly.restyle(plotDiv, {{ visible: visibility }});

                    const spamLabel = spamModes[selectedSpamIndex];
                    const emotionLabel = emotionModes[selectedEmotionIndex];
                    const originalTitle = plotDiv.layout.title.text.split(' (')[0];
                    const newTitle = `${{originalTitle}} (${{spamLabel}} / ${{emotionLabel}})`;
                    Plotly.relayout(plotDiv, 'title.text', newTitle);
                }}
                
                createRadioCheckboxes('{plot_div_id}-spam-filters', `{plot_div_id}-spam-filter`, spamModes, updatePlotVisibility);
                createRadioCheckboxes('{plot_div_id}-emotion-filters', `{plot_div_id}-emotion-filter`, emotionModes, updatePlotVisibility);
            }});
        </script>
        """
        return controls_html, script_js

    for video in df['Source Video'].unique():
        print(f"\nAnalyzing data for '{video}'...")
        video_df = df[df['Source Video'] == video].copy()

        # --- Stacked Bar Chart ---
        print(f"Generating interactive stacked bar chart for '{video}'...")
        fig_bar = go.Figure()
        bar_trace_map = []
        bar_current_trace_idx = 0

        for s_idx, spam_mode in enumerate(spam_modes):
            for e_idx, emotion_mode in enumerate(emotion_modes):
                mode_df = video_df.copy()
                if spam_mode == 'Non-Spam': mode_df = mode_df[mode_df['is_spam'] == False]
                elif spam_mode == 'Spam': mode_df = mode_df[mode_df['is_spam'] == True]
                if emotion_mode != 'All': mode_df = mode_df[mode_df['Emotion'] == emotion_mode]

                top5_per_year = mode_df.groupby(['Year', 'Country']).size().reset_index(name='counts')
                # Get the actual top 5 countries by count for each year
                top5_per_year = top5_per_year.groupby('Year').apply(lambda x: x.nlargest(5, 'counts')).reset_index(drop=True)
                pivot_df = top5_per_year.pivot_table(index='Year', columns='Country', values='counts', aggfunc='sum').fillna(0)
                
                # Debug logging for the first few combinations
                if s_idx <= 1 and e_idx <= 1:
                    logging.info(f"Video: {video}, Spam: {spam_mode}, Emotion: {emotion_mode}")
                    logging.info(f"Mode DF shape: {mode_df.shape}")
                    logging.info(f"Top5 per year shape: {top5_per_year.shape}")
                    logging.info(f"Pivot DF shape: {pivot_df.shape}")
                    if not pivot_df.empty:
                        logging.info(f"Pivot DF columns: {list(pivot_df.columns)}")
                        logging.info(f"Pivot DF index: {list(pivot_df.index)}")
                
                num_traces_for_view = len(pivot_df.columns)
                bar_trace_map.append({"spam_idx": s_idx, "emotion_idx": e_idx, "start": bar_current_trace_idx, "count": num_traces_for_view})
                
                is_visible = (spam_mode == 'All') and (emotion_mode == 'All')
                for country in pivot_df.columns:
                    emoji_flag = get_emoji_from_country(country)
                    fig_bar.add_trace(go.Bar(x=pivot_df.index, y=pivot_df[country], name=f"{emoji_flag} {country}", visible=is_visible))
                
                bar_current_trace_idx += num_traces_for_view

        fig_bar.update_layout(barmode='stack', title_text=f'Top 5 Countries by Comment Volume per Year (All / All)', xaxis_title='Year', yaxis_title='Number of Comments', legend_title='Country')
        
        bar_plot_id = f'plotly-bar-{video.lower().replace(" ", "-")}'
        bar_html_str = fig_bar.to_html(full_html=True, include_plotlyjs='cdn', div_id=bar_plot_id)
        bar_controls, bar_script = _generate_filter_html_and_js(bar_plot_id, bar_trace_map, spam_modes, emotion_modes, bar_current_trace_idx)

        body_tag_start = bar_html_str.find('<body>') + len('<body>')
        final_bar_html = bar_html_str[:body_tag_start] + bar_controls + bar_html_str[body_tag_start:]
        body_tag_end = final_bar_html.rfind('</body>')
        final_bar_html = final_bar_html[:body_tag_end] + bar_script + final_bar_html[body_tag_end:]

        bar_chart_path = os.path.join(BASE_DIR, f"{video.lower()}_task_1_yearly_top5_countries_interactive.html")
        with open(bar_chart_path, 'w', encoding='utf-8') as f:
            f.write(final_bar_html)
        print(f"Saved interactive bar chart to '{bar_chart_path}'")
        
        # --- World Map Heatmap ---
        print(f"Generating interactive world map heatmap for '{video}'...")
        fig_map = go.Figure()
        map_trace_map = []
        map_current_trace_idx = 0

        for s_idx, spam_mode in enumerate(spam_modes):
            for e_idx, emotion_mode in enumerate(emotion_modes):
                mode_df = video_df.copy()
                if spam_mode == 'Non-Spam': mode_df = mode_df[mode_df['is_spam'] == False]
                elif spam_mode == 'Spam': mode_df = mode_df[mode_df['is_spam'] == True]
                if emotion_mode != 'All': mode_df = mode_df[mode_df['Emotion'] == emotion_mode]

                country_totals = mode_df['Country'].value_counts().reset_index()
                country_totals.columns = ['Country', 'Total Comments']
                country_totals['ISO_Alpha'] = country_totals['Country'].progress_apply(get_iso_alpha)
                country_totals.dropna(subset=['ISO_Alpha'], inplace=True)
                
                is_visible = (spam_mode == 'All') and (emotion_mode == 'All')
                map_trace_map.append({"spam_idx": s_idx, "emotion_idx": e_idx, "start": map_current_trace_idx, "count": 1})
                fig_map.add_trace(go.Choropleth(
                    locations=country_totals['ISO_Alpha'], z=country_totals['Total Comments'], text=country_totals['Country'],
                    colorscale='Viridis', reversescale=True, marker_line_color='darkgray', marker_line_width=0.5,
                    colorbar_title='Total Comments', visible=is_visible))
                map_current_trace_idx += 1

        fig_map.update_layout(title_text=f'Total Geographic Distribution of Comments (All / All)', geo=dict(showframe=False, showcoastlines=False, projection_type='equirectangular'))
        
        map_plot_id = f'plotly-map-{video.lower().replace(" ", "-")}'
        map_html_str = fig_map.to_html(full_html=True, include_plotlyjs='cdn', div_id=map_plot_id)
        map_controls, map_script = _generate_filter_html_and_js(map_plot_id, map_trace_map, spam_modes, emotion_modes, map_current_trace_idx)

        body_tag_start = map_html_str.find('<body>') + len('<body>')
        final_map_html = map_html_str[:body_tag_start] + map_controls + map_html_str[body_tag_start:]
        body_tag_end = final_map_html.rfind('</body>')
        final_map_html = final_map_html[:body_tag_end] + map_script + final_map_html[body_tag_end:]

        map_path = os.path.join(BASE_DIR, f"{video.lower()}_task_1_world_heatmap_interactive.html")
        with open(map_path, 'w', encoding='utf-8') as f:
            f.write(final_map_html)
        print(f"Saved interactive world map to '{map_path}'")

    # The stats CSV is no longer representative.
    # stats_df.to_csv(OUTPUT_STATS_CSV, index=False)
    # print(f"\nYearly statistics saved to '{OUTPUT_STATS_CSV}'")


# --- Main Execution ---
if __name__ == "__main__":
    unified_df = process_and_unify_data()
    if not unified_df.empty:
        analyze_and_visualize(unified_df)
        print("\nScript finished successfully.")
    else:
        print("\nScript finished, but no data was generated to visualize.") 