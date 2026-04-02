import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from utils.metrics import calculate_classification_metrics, get_classification_report, confusion_matrix_val, calculate_roc_auc, find_best_threshold_by_f1


class Evaluator:
    def __init__(self, model):
        """
        Model Değerlendirme Sınıfı
        
        Args:
            model: Eğitilmiş TensorFlow modeli
        """
        self.model = model
    
    def evaluate_model(self, X_test, y_test):
        """
        Modelin test performansını değerlendir
        
        Returns:
            tuple: (test_loss, test_accuracy)
        """
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        
        return test_loss, test_acc
    
    def get_predictions(self, X_test, threshold=0.5):
        """
        Tahminleri al
        
        Args:
            X_test: Test features
            threshold: Sınıflandırma eşiği (0-1 arası)
        
        Not:
            - threshold düşerse recall artar, precision düşebilir
            - threshold yükselirse precision artar, recall düşebilir
        
        Returns:
            tuple: (y_prob, y_pred)
        """
        y_prob = self.model.predict(X_test)
        y_pred = (y_prob > threshold).astype(int)
        
        return y_prob, y_pred
    
    def print_classification_metrics(self, y_test, y_pred):
        """
        Sınıflandırma metriklerini yazdır
        
        Args:
            y_test: Gerçek etiketler
            y_pred: Tahmin edilen etiketler
        """
        metrics = calculate_classification_metrics(y_test, y_pred)
        
        print("\nCLASSIFICATION METRICS")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        print("\nCLASSIFICATION REPORT")
        print(get_classification_report(y_test, y_pred))
    
    def confusion_matrix_plt(self, y_test, y_pred, title="Confusion Matrix"):
        """
        Confusion Matrix görselleştir
        
        Args:
            y_test: Gerçek etiketler
            y_pred: Tahmin edilen etiketler
            title: Grafik başlığı
        
        Matris yapısı:
                        Predicted Normal    Predicted Fraud
        Actual Normal   True Negative (TN)  False Positive (FP)
        Actual Fraud    False Negative (FN)  True Positive (TP)
        """
        confusion_mat = confusion_matrix_val(y_test, y_pred)
        
        plt.figure(figsize=(9, 6))
        plt.imshow(confusion_mat, cmap='Blues', interpolation='nearest')
        plt.title(title)
        plt.colorbar()
        plt.xticks([0, 1], ["Normal", "Fraud"])
        plt.yticks([0, 1], ["Normal", "Fraud"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        
        # Sayıları grafiğe ekle
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(confusion_mat[i][j]),
                        ha="center", va="center",
                        color="white" if confusion_mat[i][j] > confusion_mat.max()/2 else "black",
                        fontsize=16)
        
        plt.tight_layout()
        plt.savefig("assets/model/confusion_matrix.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def roc_curve_plt(self, y_test, y_prob):
        """
        ROC Curve çiz
        
        Args:
            y_test: Gerçek etiketler
            y_prob: Tahmin olasılıkları
        """
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        auc_score = calculate_roc_auc(y_test, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', label=f'Model (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'r--', label='Random (AUC = 0.5)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("assets/model/roc_curve.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def threshold_tuning(self, y_test, y_prob):
        """
        En iyi threshold'u bul
        
        Args:
            y_test: Gerçek etiketler
            y_prob: Tahmin olasılıkları
        
        Returns:
            float: En iyi F1 skoru veren threshold
        """
        best_threshold = find_best_threshold_by_f1(y_test, y_prob)
        print(f"\nBest Threshold: {best_threshold:.2f}")
        return best_threshold
    
    def threshold_metrics_plot(self, y_test, y_prob):
        """
        Farklı threshold değerleri için metrikleri çiz
        
        Args:
            y_test: Gerçek etiketler
            y_prob: Tahmin olasılıkları
        """
        thresholds = np.arange(0.1, 1.0, 0.05)
        precisions, recalls, f1s = [], [], []
        
        for t in thresholds:
            y_pred = (y_prob > t).astype(int)
            metrics = calculate_classification_metrics(y_test, y_pred)
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1s.append(metrics['f1_score'])
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
        plt.plot(thresholds, recalls, 'r-', label='Recall', linewidth=2)
        plt.plot(thresholds, f1s, 'g-', label='F1 Score', linewidth=2)
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Metrics vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("assets/model/threshold_metrics.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def risk_score_distribution_plot(self, y_test, y_prob):
        """
        Risk skoru dağılımını çiz
        
        Args:
            y_test: Gerçek etiketler
            y_prob: Tahmin olasılıkları
        """
        plt.figure(figsize=(10, 6))
        plt.hist(y_prob[y_test == 0], bins=50, alpha=0.5, label='Normal', color='green')
        plt.hist(y_prob[y_test == 1], bins=50, alpha=0.5, label='Fraud', color='red')
        plt.xlabel('Risk Score')
        plt.ylabel('Count')
        plt.title('Risk Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("assets/model/risk_score_distribution.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def predict_risk_score(self, transaction):
        """
        Tek bir işlem için risk skoru hesapla
        
        Args:
            transaction: Tek işlem verisi (DataFrame)
        
        Returns:
            float: Risk skoru (0-1 arası)
        """
        risk_score = self.model.predict(transaction, verbose=0)[0][0]
        return risk_score
    
    def predict_batch(self, transactions, threshold=0.5):
        """
        Toplu tahmin yap
        
        Args:
            transactions: İşlem verileri (DataFrame)
            threshold: Sınıflandırma eşiği
        
        Returns:
            dict: Risk skorları ve tahminler
        """
        probabilities = self.model.predict(transactions, verbose=0)
        predictions = (probabilities > threshold).astype(int)
        
        results = []
        for i in range(len(probabilities)):
            results.append({
                'index': i,
                'risk_score': float(probabilities[i][0]),
                'is_fraud': bool(predictions[i][0]),
                'confidence': float(max(probabilities[i][0], 1 - probabilities[i][0]))
            })
        
        return results
