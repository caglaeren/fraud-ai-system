import tensorflow as tf
import os
import yaml
from datetime import datetime


class Trainer:
    def __init__(self, model, config_path="config.yaml"):
        """
        Model Eğitim Sınıfı
        
        Args:
            model: Eğitilmemiş TensorFlow modeli
            config_path: Config dosyası yolu
        """
        self.model = model
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path):
        """Config dosyasını yükle"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {
                'training': {
                    'epochs': 30,
                    'batch_size': 2048,
                    'early_stopping': {'enabled': True, 'patience': 5},
                    'learning_rate_scheduler': {'enabled': True, 'factor': 0.5, 'patience': 3}
                },
                'checkpoint': {'save_dir': 'models/', 'save_best_only': True}
            }
    
    def train(self, X_train, y_train, X_test, y_test, epochs=None, batch_size=None, class_weights=None):
        """
        Modeli eğit
        
        Args:
            X_train: Eğitim özellikleri
            y_train: Eğitim etiketleri
            X_test: Test özellikleri
            y_test: Test etiketleri
            epochs: Epoch sayısı (config'den alınır eğer None ise)
            batch_size: Batch boyutu (config'den alınır eğer None ise)
            class_weights: Class ağırlıkları (imbalanced data için)
        
        Returns:
            history: Eğitim geçmişi
        """
        training_config = self.config.get('training', {})
        
        if epochs is None:
            epochs = training_config.get('epochs', 30)
        if batch_size is None:
            batch_size = training_config.get('batch_size', 2048)
        
        # Callbacks oluştur
        callbacks = self._build_callbacks()
        
        # Eğitim
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        return history
    
    def _build_callbacks(self):
        """
        Eğitim callback'lerini oluştur
        
        Returns:
            list: Callback listesi
        """
        callbacks = []
        training_config = self.config.get('training', {})
        checkpoint_config = self.config.get('checkpoint', {})
        
        # Early Stopping
        early_stopping_config = training_config.get('early_stopping', {})
        if early_stopping_config.get('enabled', True):
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor=early_stopping_config.get('monitor', 'val_loss'),
                patience=early_stopping_config.get('patience', 5),
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
        
        # Learning Rate Scheduler
        lr_config = training_config.get('learning_rate_scheduler', {})
        if lr_config.get('enabled', True):
            lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=lr_config.get('factor', 0.5),
                patience=lr_config.get('patience', 3),
                min_lr=1e-6,
                verbose=1
            )
            callbacks.append(lr_scheduler)
        
        # Model Checkpoint
        save_dir = checkpoint_config.get('save_dir', 'models/')
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = os.path.join(save_dir, f'model_{timestamp}.keras')
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=checkpoint_config.get('monitor', 'val_loss'),
            save_best_only=checkpoint_config.get('save_best_only', True),
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # TensorBoard (opsiyonel)
        logging_config = self.config.get('logging', {})
        if logging_config.get('tensorboard', False):
            log_dir = logging_config.get('log_dir', 'logs/')
            os.makedirs(log_dir, exist_ok=True)
            tensorboard = tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(log_dir, timestamp),
                histogram_freq=1
            )
            callbacks.append(tensorboard)
        
        return callbacks
    
    def save_model(self, path=None):
        """
        Modeli kaydet
        
        Args:
            path: Kayıt yolu (None ise otomatik oluştur)
        """
        if path is None:
            save_dir = self.config.get('checkpoint', {}).get('save_dir', 'models/')
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(save_dir, f'fraud_model_{timestamp}.keras')
        
        self.model.save(path)
        print(f"Model saved to: {path}")
        return path
