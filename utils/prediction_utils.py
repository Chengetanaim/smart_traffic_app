import random

def predict_traffic(df, origin, destination):
    recent = df[df['location'] == origin].sort_values(by="timestamp").iloc[-1]
    return {
        "congestion_level": str(recent['congestion_level']),
        "accident_risk": f"{recent['accident_probability']*100:.1f}%",
        "fuel_consumption": f"{recent['fuel_consumption_l_per_100km']} L/100km"
    }
