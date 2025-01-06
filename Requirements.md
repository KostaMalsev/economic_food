Updated requirements:

**Graphs**

1. **c3 vs Food Actual and Food Norm**
   - X-axis: c3
   - Y-axis: Food Actual and Food Norm separately
   - Plot their difference

2. **c3 vs c3 and c^3**
   - X-axis: c3
   - Y-axis: c3 and c^3
   - Plot their difference

3. **Food Actual vs Food Actual and Food Norm**
   - X-axis: Food Actual
   - Y-axis: Food Actual and Food Norm
   - Plot their difference

4. **Per Capita Graphs for Sedentary and Active Lifestyles**
   - Repeat all above graphs with per capita metrics for:
     - Sedentary lifestyle
     - Active lifestyle

5. **ZU as a Function of ZL and Food Norm**
   - Check if a 3x relationship exists

6. **Household Distribution Graph**
   - X-axis: Food Actual - Food Norm
   - This normalizes household food deprivation
   - Orders households from most deprived (most negative, "poorest") to least deprived (most positive, "richest")
   - Zero point indicates Food Actual = Food Norm (Ravallion's upper poverty line ZU)
   - Households below zero are considered poor

**Issues**

1. **Per Capita Graphs**
   - Check the plotted graph for deprivation and others

**Requirements**

**For Active and Sedentary Households**

1. **Total Expenditure vs Food Expenditure**
   - Graph Title: "Total Expenditure vs Food Expenditure (Active/Sedentary Households)"
   - [x][y] = [c3][FoodActualY] for all households as active/sedentary.

2. **Total Expenditure vs Non-Food Expenditure**
   - Graph Title: "Total Expenditure vs Non-Food Expenditure (Active/Sedentary Households)"
   - [x][y] = [c3][c3 - FoodActualY] for all households as active.

3. **Actual vs Estimated Expenditure**
   - Graph Title: "'Actual vs Estimated Expenditure (Active/Sedentary Households)'"
   - [x][y] = [c3][c3^] with overlay: [c3][c3^-c3] for all households as active/sendetary.

4. **Food Expenditure vs Food Norm**
   - Graph Title: "Food Expenditure vs Food Norm (Active Households) Acive/Sendetary"
   - [x][y] = [FoodActualX][FoodActualY] 
   - with overlay: [FoodActualX][FoodNorm], 
   - and with overlay: [FoodActualX][FoodNorm - FoodActualY] for all households active and sendetary.

6. **ZL vs ZU**
   - Graph Title: "ZL vs ZU"
   - [x][y] = [ZL][ZU] for all households active and sendetary.

7. **ZL vs Food Norm**
   - Graph Title: "ZL vs Food Norm (Active/Sendetary Households)"
   - [x][y] = [ZL][FoodNorm] for all households as active and sendetary. Check if 3X bigger. (overlay y = 3x)

8. **FoodActual - FoodNorm for Households**
   - Graph Title: "FoodActual - FoodNorm for households active/sendetary"
   - [x][y] = [c3][FoodActualY-FoodNorm] active and sendetary

10. **Per Person Graphs**
   - Do 1-8 graphs for persons instead of households. By normalizing the values by the number of persons in the household. i.e: FoodNorm/(#of persons in household), c3/(#..), Zl/(#..), ZU/(#..). FoodActual/(#..)

**Glossary**

1. **c3**: Total expenses from field "c3"
2. **FoodActualY**: From field "food_actual-active" for active and "food_actual-sendetary" for sendetary
3. **FoodActualX**: The same as c3
4. **FoodNorm**: Values for active in field "FoodNorm-active" and for sendetary in field: "FoodNorm-sendetary"
5. **ZU**: c3^
6. **ZL**: Need to calculate for group and for each person (which is normalized by family size).

**Regression**

9. **Identify Family Types Below FoodNorm**
   - Perform regression to identify which family types are below the FoodNorm

**Available Data Parameters**

1. **FoodActualY**: c30+c31 [food_actual]
2. **FoodActualX**: c3
3. **c3**: ['expenses'] = ["F_norm Active cheapest - סך הוצאה נמוכה למספר הסלים"]
4. **c3^**: ZU

5. **ZL**: Need to calculate for group and for each person (which is normalized by family size).
6. **ZU**: Need to calculate for each person instead of group.

7. **FoodNorm**: ['food_norm'] = row["F_norm Active cheapest - סך הוצאה נמוכה למספר הסלים"]

8. **Missing Values for Sedentary**: Need to calculate c3^, FoodActualX/Y. c3, ZL, and ZU can be calculated (TODO)
   - Need to do regression to get coefficients for c3^

