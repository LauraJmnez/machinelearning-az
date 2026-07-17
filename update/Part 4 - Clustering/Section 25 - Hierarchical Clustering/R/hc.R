# Clustering Jerárquico

# Importing the dataset
dataset = read.csv('mall.csv')
X = dataset[4:5]

# Utilizar el dendrogrma para encontrar el número óptimo de clusters
dendrogram = hclust(d = dist(X, method = 'euclidean'),
                    method = 'ward.D')
plot(dendrogram,
     main = paste('Dendrogram'),
     xlab = 'Customers',
     ylab = 'Euclidean distances')

# Ajustar el Clustering Jerárquico a nuestro dataset
hc = hclust(d = dist(X, method = 'euclidean'),
            method = 'ward.D')
y_hc = cutree(hc, 5)

# Visualizar clusters 2D
library(cluster)
clusplot(x = X,
         clus = y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels= 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clustering de clientes'),
         xlab = 'Ingresos anuales',
         ylab = 'Puntuación')
