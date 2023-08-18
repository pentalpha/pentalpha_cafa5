import tensorflow as tf
import numpy as np

def makeMultiClassifierModel(train_x, train_y,
                        batch_size, n_layers, 
                        size_factors, hidden_activations, 
                        optimizer, epochs):
    n_features = len(train_x[0])
    first = tf.keras.layers.BatchNormalization(input_shape=[n_features])
    last = tf.keras.layers.Dense(units=len(train_y[0]), activation='sigmoid')
    hidden_sizes = [n_features*size_factor for size_factor in size_factors]
    hidden_layers = [tf.keras.layers.Dense(units=hidden_sizes[param_i], 
                                           activation=hidden_activations[param_i])
                    for param_i in range(n_layers) if hidden_sizes[param_i] > 0.0]
    model = tf.keras.Sequential([first] + hidden_layers + [last])
    
    # Compile model
    '''self.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=''
    )'''
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy'
    )
    
    history = model.fit(
        train_x, train_y,
        batch_size=batch_size,
        epochs=epochs
    )

    '''history_df = pd.DataFrame(history.history)
    history_df.loc[:, ['mean_absolute_percentage_error']].plot(title="mean_absolute_percentage_error")
    history_df.loc[:, ['val_mean_absolute_percentage_error']].plot(title="Val mean_absolute_percentage_error")'''
    
    return model

keras_classification = {
    'genes': ['batch_size', 'learning_rate', 'epochs', 'hidden1', 'hidden2', 'hidden3', 'n_layers'],
    'discrete_genes': ['optimizer', 'activator1', 'activator2', 'activator3'],
    'gene_values': {
        'batch_size': [1200, 9000], 
        'learning_rate': [0.0003, 0.06], 
        'epochs': [8, 12],
        'hidden1': [0.2, 1.75],
        'hidden2': [0.2, 1.75],
        'hidden3': [0.2, 1.75],
        'n_layers': [2,3]},
    'discrete_gene_values': {
        'optimizer': ['AdamOptimizer'],
        'activator1': ['relu'],
        'activator2': ['relu', 'softmax'],
        'activator3': ['relu', 'softmax']},
    'gene_types': {'batch_size': int, 'learning_rate': float, 'epochs': int, 'hidden1': float, 
        'hidden2': float, 'hidden3': float, 'n_layers': int}
}