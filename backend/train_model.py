"""
Run this ONCE to train the ML model before starting the server.
Usage: python train_model.py
"""
import sys
sys.path.append('.')
from agents.transaction_scorer import TransactionScorer

if __name__ == '__main__':
    print('=' * 60)
    print('SyndicateAI — Training ML Fraud Scorer')
    print('=' * 60)

    scorer = TransactionScorer()
    metrics = scorer.train('data/creditcard.csv')

    print('\n' + '=' * 60)
    print('TRAINING COMPLETE')
    print('=' * 60)
    print(f'  AUC-ROC:    {metrics["auc_roc"]}')
    print(f'  Precision:  {metrics["precision"]}')
    print(f'  Recall:     {metrics["recall"]}')
    print(f'  F1 Score:   {metrics["f1"]}')
    print(f'  Model Type: {metrics["model_type"]}')
    print(f'  Features:   {metrics["n_features"]}')
    print(f'  Train Size: {metrics["n_train"]:,}')
    print('\nModel saved to models/. Start the server: python main.py')