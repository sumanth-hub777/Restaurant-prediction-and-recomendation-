import pandas as pd
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'zomato.csv'
df = pd.read_csv(file_path)

# Check for missing latitude and longitude values
if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
    print("Error: The dataset does not contain 'Latitude' and 'Longitude' columns.")
else:
    # Data preview
    print("Sample Data Preview:")
    print(df[['Restaurant Name', 'City', 'Latitude', 'Longitude', 'Average Cost for two', 'Aggregate rating']].head())

    # Filter out rows with missing geographic coordinates
    df = df.dropna(subset=['Latitude', 'Longitude'])

    # Create a base map centered around a central location (e.g., the average latitude and longitude)
    map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
    restaurant_map = folium.Map(location=map_center, zoom_start=12)

    # Add MarkerCluster for better performance when plotting many points
    marker_cluster = MarkerCluster().add_to(restaurant_map)

    # Plot all restaurants on the map
    for _, row in df.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"{row['Restaurant Name']} - {row['City']}",
            icon=folium.Icon(color='blue')
        ).add_to(marker_cluster)

    # Save the map to an HTML file
    restaurant_map.save('restaurant_map.html')

    print("Map has been saved as 'restaurant_map.html'. Open this file to explore the restaurant locations.")

    # Group restaurants by city/locality and calculate statistics
    city_group = df.groupby('City').agg(
        num_restaurants=('Restaurant Name', 'count'),
        avg_rating=('Aggregate rating', 'mean'),
        avg_cost=('Average Cost for two', 'mean')
    ).reset_index()

    print("\nCity-level Statistics (Number of Restaurants, Average Rating, Average Cost):")
    print(city_group.sort_values(by='num_restaurants', ascending=False).head())

    # Plot city-level statistics: Number of Restaurants and Average Rating
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.barplot(x='num_restaurants', y='City', data=city_group.sort_values('num_restaurants', ascending=False), ax=axes[0])
    axes[0].set_title('Number of Restaurants by City')
    axes[0].set_xlabel('Number of Restaurants')

    sns.barplot(x='avg_rating', y='City', data=city_group.sort_values('avg_rating', ascending=False), ax=axes[1])
    axes[1].set_title('Average Rating by City')
    axes[1].set_xlabel('Average Rating')

    plt.tight_layout()
    plt.show()

    # Calculate the concentration of restaurants in different areas by city
    city_concentration = city_group['num_restaurants'].sum()
    print(f"\nTotal number of restaurants in all cities: {city_concentration}")

    # Identify cities with the highest average ratings and lowest average cost
    best_rated_cities = city_group.sort_values(by='avg_rating', ascending=False).head(5)
    most_affordable_cities = city_group.sort_values(by='avg_cost', ascending=True).head(5)

    print("\nTop 5 Best Rated Cities:")
    print(best_rated_cities[['City', 'avg_rating']])

    print("\nTop 5 Most Affordable Cities:")
    print(most_affordable_cities[['City', 'avg_cost']])

