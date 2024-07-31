data['date'] = pd.to_datetime(data['date'])
data['expiration'] = pd.to_datetime(data['expiration'])

# Encode the 'type' column: 'call' as 0, 'put' as 1
data['type'] = data['type'].map({'call': 0, 'put': 1})

# Calculate the number of days until expiration
data['days_until_expiration'] = (data['expiration'] - data['date']).dt.days
# Separate training and testing data
train_data = data[data['date'] <= '2024-06-21']
test_data = data[data['date'] > '2024-06-21']

# Drop unnecessary columns
train_data = train_data.drop(columns=['contractID', 'symbol', 'expiration', 'strike', 'date'])
test_data = test_data.drop(columns=['contractID', 'symbol', 'expiration', 'strike', 'date'])

# Drop unnecessary columns
train_data = train_data.drop(columns=['contractID', 'symbol', 'expiration', 'strike', 'date'])
test_data = test_data.drop(columns=['contractID', 'symbol', 'expiration', 'strike', 'date'])

# Separate features and target
features = train_data.drop(columns=['last'])
target = train_data['last']

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% of variance
reduced_features = pca.fit_transform(scaled_features)

# Create a new DataFrame with the reduced features
reduced_features_df = pd.DataFrame(reduced_features)

# Add the target back to the DataFrame
reduced_features_df['last'] = target.values

# Split the data into training and testing sets
train_data = reduced_features_df
test_features = test_data.drop(columns=['last'])
test_target = test_data['last']

# Standardize the test features
scaled_test_features = scaler.transform(test_features)

# Apply PCA to test features
reduced_test_features = pca.transform(scaled_test_features)

# Create a new DataFrame with the reduced test features
reduced_test_features_df = pd.DataFrame(reduced_test_features)

# Add the target back to the DataFrame
reduced_test_features_df['last'] = test_target.values
