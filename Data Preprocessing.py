# Split the data
train_data = data[data['date'] <= '2024-06-14']
validation_data = data[(data['date'] > '2024-06-14') & (data['date'] <= '2024-06-21')]
test_data = data[data['date'] > '2024-06-21']

# Preprocess the training data
train_data = train_data.drop(columns=['contractID', 'symbol', 'expiration', 'strike', 'date'])
features_train = train_data.drop(columns=['last', 'bid', 'ask'])
target_train = train_data['last']

scaler = StandardScaler()
scaled_features_train = scaler.fit_transform(features_train)

train_data = pd.DataFrame(scaled_features_train, columns=features_train.columns)
train_data['last'] = target_train.values
train_data['bid'] = train_data['bid'].values
train_data['ask'] = train_data['ask'].values

# Preprocess the validation data
validation_data = validation_data.drop(columns=['contractID', 'symbol', 'expiration', 'strike', 'date'])
features_validation = validation_data.drop(columns=['last', 'bid', 'ask'])
target_validation = validation_data['last']

scaled_features_validation = scaler.transform(features_validation)

validation_data = pd.DataFrame(scaled_features_validation, columns=features_validation.columns)
validation_data['last'] = target_validation.values
validation_data['bid'] = validation_data['bid'].values
validation_data['ask'] = validation_data['ask'].values

# Preprocess the test data
test_data = test_data.drop(columns=['contractID', 'symbol', 'expiration', 'strike', 'date'])
features_test = test_data.drop(columns=['last', 'bid', 'ask'])
target_test = test_data['last']

scaled_features_test = scaler.transform(features_test)

test_data = pd.DataFrame(scaled_features_test, columns=features_test.columns)
test_data['last'] = target_test.values
test_data['bid'] = test_data['bid'].values
test_data['ask'] = test_data['ask'].values
