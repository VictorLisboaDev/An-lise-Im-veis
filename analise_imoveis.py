import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# 1. Carregar os dados
# Substitua 'housing.csv' pelo caminho correto do seu dataset
housing = pd.read_csv('housing.csv')

# 2. Análise Exploratória de Dados (EDA)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=housing['median_income'], y=housing['median_house_value'])
plt.xlabel('Renda Mediana')
plt.ylabel('Valor Mediano das Casas')
plt.title('Relação entre Renda Mediana e Valor das Casas')
plt.show()

# 3. Divisão dos dados em treino (70%) e teste (30%)
train_set, test_set = train_test_split(housing, test_size=0.3, random_state=42)

# 4. Análise de outliers
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=train_set['median_income'])
plt.title('Distribuição da Renda Mediana')

plt.subplot(1, 2, 2)
sns.boxplot(y=train_set['median_house_value'])
plt.title('Distribuição do Valor Mediano das Casas')
plt.show()

# 5. Seleção de variáveis para modelo reduzido
train_features_reduced = train_set[['median_income', 'housing_median_age', 'total_rooms']]
test_features_reduced = test_set[['median_income', 'housing_median_age', 'total_rooms']]
train_labels = train_set['median_house_value']
test_labels = test_set['median_house_value']

# 6. Treinar modelo de Regressão Linear
lin_reg = LinearRegression()
lin_reg.fit(train_features_reduced, train_labels)
predictions_lr = lin_reg.predict(test_features_reduced)
mae_lr = mean_absolute_error(test_labels, predictions_lr)
print(f'MAE - Regressão Linear: {mae_lr:.2f}')

# 7. Implementação de uma Árvore de Decisão
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(train_features_reduced, train_labels)
predictions_tree = tree_reg.predict(test_features_reduced)
mae_tree = mean_absolute_error(test_labels, predictions_tree)
print(f'MAE - Árvore de Decisão: {mae_tree:.2f}')

# 8. Comparação dos modelos
if mae_tree < mae_lr:
    print("A Árvore de Decisão teve melhor desempenho.")
else:
    print("A Regressão Linear teve melhor desempenho.")