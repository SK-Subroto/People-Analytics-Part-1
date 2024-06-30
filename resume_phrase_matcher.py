import os
import fitz  # PyMuPDF
import spacy
import pandas as pd
from collections import Counter
from spacy.matcher import PhraseMatcher
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")

# Path to resumes and skills CSV
mypath = r'Resume'  # Enter your path here where you saved the resumes
pdfdocs = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
skills_csv_path = r'data\skills.csv'

# Function to extract text from PDFs
def pdfextract2(resume):
    doc = fitz.open(resume)
    text = []
    for page in doc:
        t = page.get_text()
        text.append(t)
    return " ".join(text)

# Function to create skill profile from resume
def create_profile(resume):
    text = pdfextract2(resume)
    text = text.lower().replace("\\n", " ")

    # Read skills from CSV
    keyword_dict = pd.read_csv(skills_csv_path, encoding='ansi')
    
    # Define categories and their abbreviations
    categories = {
        'Statistics': 'Stats',
        'Mathematics': 'Math',
        'Artificial Intelligence': 'AI',
        'Programming': 'Prog',
        'Cloud Computing': 'CloudComp',
        'Digital Transformation Manager': 'DTManager'
    }
    
    matcher = PhraseMatcher(nlp.vocab)
    
    for category, abbreviation in categories.items():
        words = [nlp(text) for text in keyword_dict[category].dropna()]
        matcher.add(abbreviation, None, *words)
    
    doc = nlp(text)
    matches = matcher(doc)
    
    # Collect matched keywords and their categories
    matched_keywords = [(nlp.vocab.strings[match_id], span.text) for match_id, start, end in matches for span in [doc[start:end]]]
    
    # Create a DataFrame from matched keywords
    keywords_counter = Counter(matched_keywords)
    
    if not keywords_counter:
        return pd.DataFrame(columns=['Employee Name', 'Category', 'Keyword', 'Count'])
    
    keywords_df = pd.DataFrame(keywords_counter.items(), columns=['Category_Keyword', 'Count'])
    
    # Split Category_Keyword into separate columns
    keywords_df[['Category', 'Keyword']] = keywords_df['Category_Keyword'].apply(lambda x: pd.Series(x))
    keywords_df.drop(columns=['Category_Keyword'], inplace=True)
    
    # Extract employee name from filename
    filename = os.path.basename(resume)
    employee_name = filename.split('_')[0].lower()
    keywords_df['Employee Name'] = employee_name
    
    return keywords_df[['Employee Name', 'Category', 'Keyword', 'Count']]

# Create profiles for all resumes
text_content = pd.DataFrame()

for resume in pdfdocs:
    dat = create_profile(resume)
    text_content = pd.concat([text_content, dat], ignore_index=True)

# Count words under each category and visualize with Matplotlib
text_content2 = text_content.groupby(['Employee Name', 'Category'])['Keyword'].count().unstack(fill_value=0).reset_index()
new_data = text_content2.set_index('Employee Name')

plt.rcParams.update({'font.size': 10})
ax = new_data.plot.barh(title="Skills per category", stacked=True, figsize=(25, 7))

labels = []
for column in new_data.columns:
    for value in new_data[column]:
        labels.append(f"{column}: {value}")

patches = ax.patches
for label, rect in zip(labels, patches):
    width = rect.get_width()
    if width > 0:
        x = rect.get_x() + width / 2.
        y = rect.get_y() + rect.get_height() / 2.
        ax.text(x, y, label, ha='center', va='center')

plt.show()
