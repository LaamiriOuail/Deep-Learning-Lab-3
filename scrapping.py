import pandas as pd
import random

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('arabic_text_data.csv')

# Generate random scores between 0 and 10
random_scores = [random.randint(0, 10) for _ in range(len(df))]

# Update the 'Score' column with the random scores
df['Score'] = random_scores

# Save the updated DataFrame to a new CSV file
df.to_csv('updated_csv_file.csv', index=False)
