"""Settings shared by every module."""

DATA_PATH = "data/chicago_crimes_sample.csv"
OUTPUT_DIR = "outputs"

SEED = 42            # fixed seed -> reproducible results
TEST_SIZE = 0.30     # 70 % train / 30 % held-out test
ALPHA = 0.05         # significance level
BOOTSTRAP = 2000     # bootstrap resamples for the CI of an accuracy difference
RISK_QUANTILE = 0.75 # a place-time cell is "high risk" if its count is in the top 25 %

TIME_SLOTS = ["Night (0-6)", "Morning (6-12)", "Afternoon (12-18)", "Evening (18-24)"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# 30 crime types -> 5 categories (used for the pattern analysis)
CATEGORIES = ["PROPERTY", "VIOLENT", "FRAUD", "DRUGS_WEAPONS", "OTHER"]
CATEGORY_MAP = {
    "PROPERTY": ["THEFT", "BURGLARY", "MOTOR VEHICLE THEFT", "CRIMINAL DAMAGE", "ARSON"],
    "VIOLENT": ["BATTERY", "ASSAULT", "ROBBERY", "HOMICIDE", "CRIM SEXUAL ASSAULT",
                "KIDNAPPING", "INTIMIDATION", "STALKING"],
    "FRAUD": ["DECEPTIVE PRACTICE"],
    "DRUGS_WEAPONS": ["NARCOTICS", "WEAPONS VIOLATION", "CONCEALED CARRY LICENSE VIOLATION",
                      "PROSTITUTION", "GAMBLING", "LIQUOR LAW VIOLATION", "PUBLIC INDECENCY"],
}

# Official names of Chicago's 77 community areas (the "places" a user can choose)
COMMUNITY_AREAS = {
    1: "Rogers Park", 2: "West Ridge", 3: "Uptown", 4: "Lincoln Square", 5: "North Center",
    6: "Lake View", 7: "Lincoln Park", 8: "Near North Side", 9: "Edison Park",
    10: "Norwood Park", 11: "Jefferson Park", 12: "Forest Glen", 13: "North Park",
    14: "Albany Park", 15: "Portage Park", 16: "Irving Park", 17: "Dunning", 18: "Montclare",
    19: "Belmont Cragin", 20: "Hermosa", 21: "Avondale", 22: "Logan Square",
    23: "Humboldt Park", 24: "West Town", 25: "Austin", 26: "West Garfield Park",
    27: "East Garfield Park", 28: "Near West Side", 29: "North Lawndale", 30: "South Lawndale",
    31: "Lower West Side", 32: "Loop", 33: "Near South Side", 34: "Armour Square",
    35: "Douglas", 36: "Oakland", 37: "Fuller Park", 38: "Grand Boulevard", 39: "Kenwood",
    40: "Washington Park", 41: "Hyde Park", 42: "Woodlawn", 43: "South Shore", 44: "Chatham",
    45: "Avalon Park", 46: "South Chicago", 47: "Burnside", 48: "Calumet Heights",
    49: "Roseland", 50: "Pullman", 51: "South Deering", 52: "East Side", 53: "West Pullman",
    54: "Riverdale", 55: "Hegewisch", 56: "Garfield Ridge", 57: "Archer Heights",
    58: "Brighton Park", 59: "McKinley Park", 60: "Bridgeport", 61: "New City",
    62: "West Elsdon", 63: "Gage Park", 64: "Clearing", 65: "West Lawn", 66: "Chicago Lawn",
    67: "West Englewood", 68: "Englewood", 69: "Greater Grand Crossing", 70: "Ashburn",
    71: "Auburn Gresham", 72: "Beverly", 73: "Washington Heights", 74: "Mount Greenwood",
    75: "Morgan Park", 76: "O'Hare", 77: "Edgewater",
}

# Colours
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
MODEL_COLORS = {"Logistic Regression": GREY, "Decision Tree": ORANGE, "Random Forest": BLUE}
