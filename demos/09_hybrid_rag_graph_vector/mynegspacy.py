import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import spacy
from spacy.language import Language
from negspacy.negation import Negex

# 1. Load base model
nlp = spacy.load("en_core_web_sm")

# 2. Load your custom food model
nlp_food = spacy.load("model")

# 3. Inject the custom NER model into the base pipeline, renaming it to avoid collisions
nlp.add_pipe("ner", name="food_ner", source=nlp_food, after="ner")

# 4. Define the custom pipeline component to resolve overlapping/false entities
@Language.component("food_false_positive_filter")
def food_false_positive_filter(doc):
    filtered_ents = []
    
    for ent in doc.ents:
        if ent.label_ == "FOOD":
            # Check POS tags within the span
            valid_pos = all(token.pos_ in ["NOUN"] for token in ent)
            if valid_pos:
                filtered_ents.append(ent)
        else:
            # Keep all non-food entities found by the base 'ner'
            filtered_ents.append(ent)
            
    # Reassign the filtered entities back to the document
    try:
        from spacy.util import filter_spans
        doc.ents = filter_spans(filtered_ents)
    except Exception as e:
        print(f"Error filtering spans: {e}")
        
    return doc

# 5. Add the false-positive filter
nlp.add_pipe("food_false_positive_filter", last=True)

# 6. Initialize Negex AFTER the filter. 
# We explicitly configure it to target our custom "FOOD" label.
nlp.add_pipe(
    "negex", 
    after="food_false_positive_filter", 
    # config={"ent_types": ["FOOD", "PRODUCT", "ORG"]} # Add any other base entities you want negated
    config={"ent_types": ["FOOD"]} # Add any other base entities you want negated
)

# --- Execution & Testing ---
text = "I would like a recipe with noodle in it."
doc = nlp(text)

print(f"Analyzing text: '{text}'\n")
print(f"{'Entity':<15} | {'Label':<10} | {'Negated (Negex)':<15}")
print("-" * 45)

for ent in doc.ents:
    print(f"{ent.text:<15} | {ent.label_:<10} | {ent._.negex}")


print("\n\n")

text = "But I want a recipe without soba in it."
doc = nlp(text)

print(f"Analyzing text: '{text}'\n")
print(f"{'Entity':<15} | {'Label':<10} | {'Negated (Negex)':<15}")
print("-" * 45)

for ent in doc.ents:
    print(f"{ent.text:<15} | {ent.label_:<10} | {ent._.negex}")