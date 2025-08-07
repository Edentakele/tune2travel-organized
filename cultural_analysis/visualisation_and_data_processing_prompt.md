# New Analysis & Visualization Prompt

This document outlines the requirements for the next phase of the cultural analysis project, focusing on data processing, unification, and visualization for Task 1, and an expansion of keyword-based analysis for Task 2.

## Task 1: Yearly Geographic Distribution Visualization

**Objective:** To process, unify, and visualize the geographic distribution of comments over the years for both "Despacito" and "See You Again".

**Deadline:** Visualizations for Task 1 are requested by this Wednesday.

### Input Data
*   `[prefix]task_1_flag_emoji_extracted_comments.csv`
*   `[prefix]task_1_keyword_extracted_comments.csv` (containing "Hi from", etc.)
*   `[prefix]task_1_location_extracted_comments.csv`

### Data Processing & Unification Steps
1.  **Unify Location Data:**
    *   **Flag Emoji to Country Name Conversion:** The flag emojis from `..._flag_emoji_extracted_comments.csv` must be converted into their corresponding country names.
    *   **Country Name Extraction:** Country names need to be reliably extracted from the comments in the other two CSV files.
2.  **Combine Data:**
    *   After unifying the location data (i.e., having a clear "country" column), the processed data from all three input sources (`flag_emoji`, `keyword`, `location`) for both "Despacito" and "See You Again" should be stacked into a single, unified CSV file.
    *   This master CSV should contain at least the following columns: `comment`, `year`, `country`, `source_video` (e.g., 'Despacito' or 'See You Again').
3.  **Temporal Analysis:**
    *   The analysis must be grouped by year.

### Required Outputs
The following outputs are required for **each song separately** ("Despacito" and "See You Again"):

#### 1. Data Analysis
*   **Top 5 Countries List (per year):** For each year in the dataset, produce a list of the top 5 countries with the highest number of comments.
*   **Unique Country/City Count (per year):** For each year, calculate the total number of unique countries/cities from which comments were received.

#### 2. Visualizations
*   **Stacked Bar Chart:**
    *   **X-axis:** Year.
    *   **Y-axis:** Total number of comments.
    *   **Bars:** Each year will have one bar.
    *   **Stacking:** Each bar will be a stacked chart, with colored sections representing the comment counts from the top 5 countries for that year.
    *   **Legend:** A legend should clearly identify the country corresponding to each color.
*   **World Map Heatmap:**
    *   A world map showing the total comment distribution per country across all years. The intensity of the color on the map should correspond to the total number of comments from that country.

---

## Task 2: Enhanced Context-Based Analysis

**Objective:** To expand the keyword list for context-based filtering, assign specific context labels to the extracted comments, and structure the output by year.

### Data Processing & Labeling Steps
1.  **Expand Keyword List:**
    *   The list of keywords for context filtering needs to be expanded. The full list should now include:
        *   **Existing:** `Hispanic`, `Asian`, `Muslim`, `Latino`, `in my country`, `where I'm from`.
        *   **New:** `Caucasian`, `Christ`, `Christian`, `Diaspora`, `Immigrant`, `Black`, `White`, `our country`, `in my culture`, `homeland`.
2.  **Assign Context Labels:**
    *   Each comment that is filtered should be assigned a context label in the output CSV based on the keyword that triggered the match.
    *   The categories for labeling are:
        *   **`Ethnic Identity`**: For keywords like `latino`, `hispanic`, `asian`, `white`, `caucasian`, `black`, etc.
        *   **`Religious Identity`**: For keywords like `muslim`, `islam`, `christian`, `christ`, etc.
        *   **`Collective Memory`**: For keywords like `my country`, `our country`, `in my culture`, `homeland`, `diaspora`, `immigrant`, etc.
3.  **Special Handling:**
    *   Comments matching `Black` and `White` should be clearly identifiable, possibly with a specific `sub_label`, as they are expected to contain significant noise.
4.  **Structure Output:**
    *   The final output CSV must be structured to include the year of the comment.

### Output Requirements
*   A new, processed CSV file (or one for each source video) containing the filtered comments.
*   This CSV must include the following columns: `original_comment`, `year`, `matched_keyword`, and `context_label` (`Ethnic Identity`, `Religious Identity`, or `Collective Memory`).

---

## General Notes
*   Task 3 (Linguistic Analysis) is on hold for now.
*   The primary focus is on generating the visualizations for Task 1 first, followed by the data processing for Task 2. 