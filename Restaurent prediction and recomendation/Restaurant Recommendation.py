import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
import os

file_path = 'zomato.csv'

if not os.path.exists(file_path):
    print(f"Error: The file '{file_path}' was not found in the project directory.")
else:
    df = pd.read_csv(file_path)

    print("\nSample Data Preview:")
    print(df.head())

    # Fill any missing Cuisines with the mode value
    df['Cuisines'] = df['Cuisines'].fillna(df['Cuisines'].mode()[0])

    # Select relevant columns and remove any rows with missing data
    data = df[['Restaurant Name', 'Cuisines', 'Average Cost for two', 'Aggregate rating']]
    data = data.dropna()

    # Initialize OneHotEncoder and apply it to the 'Cuisines' column
    encoder = OneHotEncoder(sparse_output=False)  # Use sparse_output=False instead of sparse=False
    cuisine_encoded = encoder.fit_transform(data[['Cuisines']])

    # Get the feature names from the encoder
    cuisine_columns = encoder.get_feature_names_out(['Cuisines'])

    # Create a DataFrame with the encoded cuisines
    cuisine_df = pd.DataFrame(cuisine_encoded, columns=cuisine_columns)

    # Combine the original data with the encoded cuisine data
    final_data = pd.concat([data.reset_index(drop=True), cuisine_df], axis=1)

    # User preferences (customize as needed)
    user_cuisine_preference = 'Italian'
    user_price_preference = 2
    user_rating_preference = 4.0

    # Filter the data based on price and rating preferences
    filtered_data = final_data[
        (final_data['Average Cost for two'] <= user_price_preference * 250) &  # Multiply by 250 for cost filter
        (final_data['Aggregate rating'] >= user_rating_preference)  # Minimum rating filter
    ]

    if filtered_data.empty:
        print("\nSorry, no restaurants match your preferences. Please adjust your filters!")
    else:
        # Extract cuisine features from the filtered data
        cuisine_features = filtered_data[cuisine_columns]

        # Create a user vector based on the cuisine preference
        user_vector = [1 if col.split('_')[-1] == user_cuisine_preference else 0 for col in cuisine_features.columns]

        # Calculate cosine similarities between the user preferences and available restaurants
        similarities = cosine_similarity([user_vector], cuisine_features.values)[0]

        # Add the similarity score to the filtered data
        filtered_data.loc[:, 'similarity_score'] = similarities  # Avoid SettingWithCopyWarning

        # Sort the filtered data by similarity score and rating in descending order
        recommendations = filtered_data.sort_values(by=['similarity_score', 'Aggregate rating'], ascending=[False, False])

        # Show the top 5 recommended restaurants
        print("\nRecommended Restaurants based on your preferences:")
        print(recommendations[['Restaurant Name', 'Cuisines', 'Average Cost for two', 'Aggregate rating']].head(5))
