import pandas as pd
from sklearn.linear_model import LinearRegression



house_data = pd.DataFrame(
    dict(
        age = [2, 30, 8, 5],
        rooms = [7, 3, 6, 12],
        bedrooms = [3, 1, 2, 4],
        area = [4200, 2500, 3000, 5500],
        price = [500000, 200000, 250000, 750000],
    )
)
print(house_data)
model = LinearRegression()
model.fit(house_data[["age", "rooms", "bedrooms", "area"]], house_data["price"])
new_house_parameters = dict(age=4, rooms=5, bedrooms=2, area=3300)
new_house_parameters = [list(new_house_parameters.values())]
predicted_price = model.predict(new_house_parameters)

print("${:,.2f}".format(float(predicted_price)))
