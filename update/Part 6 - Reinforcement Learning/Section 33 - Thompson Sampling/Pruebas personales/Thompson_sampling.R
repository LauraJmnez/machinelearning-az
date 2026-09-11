### Muestreo de Thompson

dataset = read.csv(file = "Ads_CTR_Optimisation.csv")

## Algoritmo

# Paso 1:
N = 1000
d = 10
number_of_rewards_1 = integer(d)
number_of_rewards_0 = integer(d)

# Paso 2:
ads_selected = integer(d)
total_reward = 0
for (n in 1:N) {
   max_random = 0
   ad = 0
   for (i in 1:d) {
      random_beta = rbeta(n = 1, shape1 = number_of_rewards_1[i]+1, shape2 = number_of_rewards_0[i]+1)
      if(random_beta > max_random){
        max_random = random_beta
        ad = i
      }
   }
# Paso 3:
   ads_selected = append(ads_selected, ad)
   reward = dataset[n, ad]
   if (reward == 1) {
      number_of_rewards_1[ad] = number_of_rewards_1[ad] + 1
   }else {
      number_of_rewards_0[ad] = number_of_rewards_0[ad] + 1
   }
   total_reward = total_reward + reward
}

## Visualización de resultados
hist(ads_selected,
col = "lightblue",
main = "Histograma de los Anuncios",
xlab = "Id del Anuncio",
ylab = "Frecuencia absoluta del anuncio")

'
Confirma que el mejor anuncio es el 5 con una mejor eficiencia (max_reward es muy superior que con el UCB)
'

