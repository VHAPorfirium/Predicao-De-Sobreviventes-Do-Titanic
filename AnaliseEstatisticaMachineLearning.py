import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ===============================
# ETAPA 1: Importação dos Dados
# ===============================

# Carrega o dataset do Titanic
df = sns.load_dataset("titanic")

print("Primeiras 5 linhas do dataset:")
print(df.head())

print("\nInformações do dataset:")
print(df.info())
print("\nDescrição estatística:")
print(df.describe())

# ==========================================
# ETAPA 2: Exploração e Limpeza dos Dados
# ==========================================

# 1. Tratamento dos valores ausentes:
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# 2. Remoção de colunas irrelevantes
colunas_remover = ['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male', 'alone']
df = df.drop(columns=colunas_remover)

# 3. Converter colunas categóricas para numéricas
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Exibe o dataset após as transformações
print("\nDataset após a limpeza e pré-processamento:")
print(df.head())
print("\nInformações do dataset após limpeza:")
print(df.info())

# Seleciona apenas as colunas numéricas para calcular a correlação
numeric_df = df.select_dtypes(include=['number'])

# Calcula a correlação entre as colunas
correlation = numeric_df.corr()

# Exibe um heatmap das correlações
plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlação entre as Features e a Sobrevivência")
plt.show()

# =====================================================
# ETAPA 3: Seleção de Features e Divisão dos Dados
# =====================================================

# Seleciona as colunas que serão as features
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
X = df[features]

y = df['survived']

# Divide os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nDivisão dos dados realizada:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# =======================================
# ETAPA 4: Treinamento dos Modelos
# =======================================

# Modelo 1: Regressão Logística
modelo_logreg = LogisticRegression(max_iter=1000)
modelo_logreg.fit(X_train, y_train)

# Modelo 2: Random Forest
modelo_rf = RandomForestClassifier(random_state=42)
modelo_rf.fit(X_train, y_train)

# =======================================
# ETAPA 5: Avaliação dos Modelos
# =======================================

# Previsões nos dados de teste para cada modelo
y_pred_logreg = modelo_logreg.predict(X_test)
y_pred_rf = modelo_rf.predict(X_test)

# Função para exibir as métricas de avaliação
def exibir_metricas(modelo_nome, y_true, y_pred):
    print(f"\nMétricas para o modelo: {modelo_nome}")
    print("Acurácia:", accuracy_score(y_true, y_pred))
    print("Matriz de Confusão:\n", confusion_matrix(y_true, y_pred))
    print("Relatório de Classificação:\n", classification_report(y_true, y_pred))

exibir_metricas("Regressão Logística", y_test, y_pred_logreg)
exibir_metricas("Random Forest", y_test, y_pred_rf)

# =======================================
# ETAPA 6: Visualização de Resultados
# =======================================

# 1. Importância das Features
importances = modelo_rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 6))
plt.title("Importância das Features - Random Forest")
plt.bar(range(len(features)), importances[indices], align='center')
plt.xticks(range(len(features)), [features[i] for i in indices])
plt.ylabel("Importância")
plt.show()

# 2. Distribuição de Sobreviventes por Sexo e Classe
plt.figure(figsize=(10, 4))

# Distribuição por Sexo
plt.subplot(1, 2, 1)
sns.countplot(x='sex', hue='survived', data=df)
plt.title("Sobrevivência por Sexo")
plt.xlabel("Sexo (0: female, 1: male)")

# Distribuição por Classe
plt.subplot(1, 2, 2)
sns.countplot(x='pclass', hue='survived', data=df)
plt.title("Sobrevivência por Classe")
plt.xlabel("Classe")
plt.show()

# =======================================
# ETAPA 7: Predição para Novos Passageiros
# =======================================

def prever_sobrevivencia(pclass, sex, age, sibsp, parch, fare, embarked):
    # Cria um DataFrame com as informações do passageiro
    novo_passageiro = pd.DataFrame({
        'pclass': [pclass],
        'sex': [sex],
        'age': [age],
        'sibsp': [sibsp],
        'parch': [parch],
        'fare': [fare],
        'embarked': [embarked]
    })

    predicao = modelo_logreg.predict(novo_passageiro)
    return predicao[0]

resultado = prever_sobrevivencia(3, 1, 25, 0, 0, 10, 2)
print("\nPredição para o novo passageiro:", "Sobreviveu" if resultado == 1 else "Não Sobreviveu")
