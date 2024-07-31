data['date'] = pd.to_datetime(data['date'])
data['expiration'] = pd.to_datetime(data['expiration'])
data['type'] = data['type'].map({'call': 0, 'put': 1})
data['days_until_expiration'] = (data['expiration'] - data['date']).dt.days

train_data = data[data['date'] <= '2024-06-21']
test_data = data[data['date'] > '2024-06-21']
train_data = train_data.drop(columns=['contractID', 'symbol', 'expiration', 'strike', 'date'])
test_data = test_data.drop(columns=['contractID', 'symbol', 'expiration', 'strike', 'date'])


features = train_data.drop(columns=['last'])
target = train_data['last']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

pca = PCA(n_components=0.95) 
reduced_features = pca.fit_transform(scaled_features)

reduced_features_df = pd.DataFrame(reduced_features)
reduced_features_df['last'] = target.values
train_data = reduced_features_df
test_features = test_data.drop(columns=['last'])
test_target = test_data['last']

scaled_test_features = scaler.transform(test_features)
reduced_test_features = pca.transform(scaled_test_features)
reduced_test_features_df = pd.DataFrame(reduced_test_features)
reduced_test_features_df['last'] = test_target.values
