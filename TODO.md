# Analiza skupień recenzji

## 1. Dane
- Połączymy dane na _id_; Powstaną recenzje, które mają dodatkowe dane informacje o filmie, którego dotyczą.
- Wyrzucamy wybrakowane dane?
- Mamy jeszcze `'originalScore'`, ale jego skala jest **x/10**, **x/5** lub **x/4**

## 2. Cechy

- Wybraliśmy cechy numeryczne: `'audienceScore', 'tomatoMeter', 'runtimeMinutes'`.
- Jakościowe `genre`
- Możemy zrobić jeszcze `'scoreSentiment', 'originalLanguage', 'director'`. Jak to wykorzystać?

## 3. Klasteryzacja

**Są dwie koncepcje**
1. Łączymy recenzje z recenzentami i tworzymy tabelkę ze średnimi ocenami dla każdego gatunku.
   
   - Do tego dorobimy tabelę z globalnie najwyższą i najniższą oceną oraz medianą.
   - Klasteryzacja po recenzentach. Zobaczymy, jacy recenzenci lubią jakie gatunki filmów

2. Łączymy oceny `'audienceScore', 'tomatoMeter'` z gatunkiem 
    
   - Film z najwyższą i najniższą oceną
   - Klasteryzacja zależności pomiędzy gatunkiem a otrzymaną oceną
