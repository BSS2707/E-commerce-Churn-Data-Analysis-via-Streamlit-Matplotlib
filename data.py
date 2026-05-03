import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib as jb


d = pd.read_csv("ecommerce_churn.csv")


print(d.head())
print(d.info())
print(d.describe())

fillna = d['Tenure'].mean()
d.fillna({'Tenure': fillna}, inplace=True)



gp = d.groupby('Churn')['CashbackAmount'].agg(
    mean='mean',
    median='median',
    sum='sum'
).reset_index()

print(gp)

sns.barplot(x='Churn', y='mean', data=gp)
plt.title('Average Cashback Amount by Churn Status')
plt.xlabel('Churn Status')
plt.ylabel('Average Cashback Amount')
plt.show()


plt.pie(gp['mean'], labels=gp['Churn'], autopct='%1.1f%%', startangle=140)
plt.title('Proportion of Average Cashback Amount by Churn Status')
plt.axis('equal')
plt.show()

sns.boxplot(x='Churn', y='CashbackAmount', data=d)
plt.title('Boxplot of Cashback Amount by Churn Status')
plt.xlabel('Churn Status')
plt.ylabel('Cashback Amount')
plt.show()

sns.scatterplot(x='Tenure', y='CashbackAmount', hue='Churn', data=d)
plt.title('Scatter Plot of Tenure vs Cashback Amount by Churn Status')
plt.xlabel('Tenure')
plt.ylabel('Cashback Amount')
plt.show()

sns.countplot(x='Churn', data=d)
plt.title('Count of Churn Status')
plt.xlabel('Churn Status')
plt.ylabel('Count')
plt.show()

sns.heatmap(d.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

sns.histplot(d['CashbackAmount'], kde=True)
plt.title('Distribution of Cashback Amount')
plt.xlabel('Cashback Amount')
plt.ylabel('Frequency')
plt.show()

jb.dump(d, 'ecommerce_churn_data.joblib')
 