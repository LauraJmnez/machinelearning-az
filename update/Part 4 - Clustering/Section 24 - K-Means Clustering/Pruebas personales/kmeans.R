# Clustering K-means

# Importar los datos
dataset = read.csv("Mall_Customers.csv")

X = dataset[,4:5]

# Método del Codo para generar el método óptimo de clústers
set.seed(6)
wcss = vector()
for (i in 1:10){
  wcss[i] <- sum(kmeans(X, i)$withinss)
}
plot(1:10, wcss, type = "b", main = "Método del Codo",
     xlab =  "Número de clústers (k)", ylab = "WCSS(k)")

#El número óptimo es k=5

# Aplicar el algoritmo de k-means con k=5
set.seed(29)
kmeans <- kmeans(X, 5, iter.max = 300, nstart = 10)

# Visualización de los clústers
library(cluster)
clusplot(X, kmeans$cluster, 
         lines = 0, 
         shade = T,
         color = T,
         labels = 2,#si ponemos 2, nos sale los nombres de los clientes en vez de puntos
         plotchar = F,
         span = T,
         main = "Clúster de clientes",
         xlab = "Ingresos anuales (miles $)",
         ylab = "Puntuación (1-100)")
