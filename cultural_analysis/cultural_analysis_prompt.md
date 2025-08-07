# Cultural Comparison Analysis Prompt
**Subject:** Analysis for Cultural Comparison Section in Project

To prepare the cultural comparison section for our project, we need to extract signals related to country and cultural affiliation from the comment data. 

## Analysis Tasks:

### 1. Regex and Keyword-Based Extraction

*   Extract comments that directly mention country or city names (e.g., "Hi from India", "Love from Turkey").
*   Utilize the following keywords and patterns:
    *   "Hi from"
    *   "Love from"
    *   "Greetings from"
    *   "Watching from"
    *   Flag emojis (e.g., 🇮🇳, 🇹🇷)
*   Analyze the temporal distribution of these extracted comments (refer to "w kısmındaki dağılıma").

### 2. Keyword + Context-Based Filtering

*   Identify comments that may contain cultural references even if no specific country name is mentioned. (I'm not entirely sure, but we might be able to achieve this).
*   Consider using keywords such as:
    *   "Muslim"
    *   "Latino"
    *   "in my country"
    *   "where I'm from"
    *   "Asian"
*   Manually sample and analyze 15–20 representative comments from this group for a deeper understanding.

### 3. NLP for Linguistic Structure Analysis (Optional - "Maybe")

*   If it can be done quickly, explore the possibility of developing a rough "language score" for comments based on:
    *   Linguistic fluency
    *   Grammatical errors
    *   (This is prompted by the observation of many poorly written comments.)

