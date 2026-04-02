import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.metrics import Precision, Recall, AUC
import yaml


class FraudDetectionModel:
    def __init__(self, input_dim, config_path="config.yaml"):
        """
        Fraud Detection Model Builder
        
        Args:
            input_dim: Input feature dimension
            config_path: Path to YAML config file
        """
        self.input_dim = input_dim
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path):
        """Config dosyasını yükle"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Default config
            return {
                'model': {
                    'hidden_layers': [128, 64, 32, 16],
                    'dropout_rate': 0.3,
                    'use_batch_norm': True,
                    'activation': 'relu',
                    'output_activation': 'sigmoid'
                },
                'training': {
                    'learning_rate': 0.001,
                    'optimizer': 'adam',
                    'loss': 'binary_crossentropy'
                }
            }

    def build_model(self):
        """
        Gelişmiş model mimarisi oluştur
        
        Özellikler:
        - Dropout (overfitting önleme)
        - BatchNormalization (eğitim stabilizasyonu)
        - Çoklu metrik (precision, recall, AUC)
        - Yapılandırılabilir mimari
        """
        model_config = self.config.get('model', {})
        
        # Hyperparameters
        hidden_layers = model_config.get('hidden_layers', [128, 64, 32, 16])
        dropout_rate = model_config.get('dropout_rate', 0.3)
        use_batch_norm = model_config.get('use_batch_norm', True)
        activation = model_config.get('activation', 'relu')
        output_activation = model_config.get('output_activation', 'sigmoid')
        
        # Training config
        training_config = self.config.get('training', {})
        learning_rate = training_config.get('learning_rate', 0.001)
        optimizer_name = training_config.get('optimizer', 'adam')
        loss = training_config.get('loss', 'binary_crossentropy')
        
        # Build layers
        layers = [Input(shape=(self.input_dim,))]
        
        for i, units in enumerate(hidden_layers):
            layers.append(Dense(units, activation=activation, name=f'dense_{i+1}'))
            
            if use_batch_norm:
                layers.append(BatchNormalization(name=f'batch_norm_{i+1}'))
            
            if dropout_rate > 0:
                layers.append(Dropout(dropout_rate, name=f'dropout_{i+1}'))
        
        # Output layer
        layers.append(Dense(1, activation=output_activation, name='output'))
        
        model = tf.keras.Sequential(layers)
        
        # Optimizer
        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Compile with multiple metrics
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                'accuracy',
                Precision(name='precision'),
                Recall(name='recall'),
                AUC(name='auc')
            ]
        )
        
        return model
    
    def build_model_with_class_weights(self, y_train):
        """
        Class weights ile model oluştur
        
        Args:
            y_train: Training labels (for calculating class weights)
        
        Returns:
            model: Compiled model
            class_weights: Dict of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        
        # Calculate class weights
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        
        # Build model
        model = self.build_model()
        
        return model, class_weights


class FraudDetectionModelAdvanced:
    """Gelişmiş model - Residual connections ile"""
    
    def __init__(self, input_dim, config_path="config.yaml"):
        self.input_dim = input_dim
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {'model': {'dropout_rate': 0.3}}
    
    def build_model(self):
        """
        Residual block'lu gelişmiş mimari
        
        Avantajları:
        - Gradient flow daha iyi
        - Derin mimarilerde daha stabil
        - Daha iyi feature extraction
        """
        inputs = Input(shape=(self.input_dim,))
        
        # First block
        x = Dense(128, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Residual block 1
        residual = x
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        # Add residual (need to match dimensions)
        residual_proj = Dense(64)(residual)
        x = tf.keras.layers.Add()([x, residual_proj])
        x = Dropout(0.2)(x)
        
        # Output block
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)
        x = Dropout(0.1)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                Precision(name='precision'),
                Recall(name='recall'),
                AUC(name='auc')
            ]
        )
        
        return model
